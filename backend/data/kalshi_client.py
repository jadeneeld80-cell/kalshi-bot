
"""Kalshi API client with RSA-PSS authentication and public market access."""

import asyncio
import base64
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from backend.config import settings

logger = logging.getLogger("trading_bot")

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

_MAX_RATE_LIMIT_RETRIES = 3
_BASE_BACKOFF_SECONDS = 0.5


class KalshiClient:
    """Async Kalshi API client with public and authenticated requests."""

    def __init__(self):
        self._private_key = None

    def _load_private_key(self):
        if self._private_key is not None:
            return self._private_key

        key_path = settings.KALSHI_PRIVATE_KEY_PATH

        if not key_path:
            raise ValueError(
                "KALSHI_PRIVATE_KEY_PATH not configured"
            )

        pem_data = Path(key_path).expanduser().read_bytes()

        self._private_key = serialization.load_pem_private_key(
            pem_data,
            password=None
        )

        return self._private_key

    def _sign_request(self, method: str, path: str) -> Dict[str, str]:
        """Create RSA-PSS authentication headers."""

        timestamp_ms = str(int(time.time() * 1000))

        clean_path = path.split("?", 1)[0]
        full_path = f"/trade-api/v2{clean_path}"

        message = f"{timestamp_ms}{method.upper()}{full_path}"

        private_key = self._load_private_key()

        signature = private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return {
            "KALSHI-ACCESS-KEY": settings.KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(
                signature
            ).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "Content-Type": "application/json"
        }

    def _is_public_endpoint(self, method: str, path: str) -> bool:
        """Identify public read-only market endpoints."""

        if method.upper() != "GET":
            return False

        clean_path = path.split("?", 1)[0]

        public_prefixes = (
            "/markets",
            "/events",
            "/series"
        )

        return any(
            clean_path == prefix
            or clean_path.startswith(prefix + "/")
            for prefix in public_prefixes
        )

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Send requests with public access and rate-limit retries."""

        url = f"{BASE_URL}{path}"

        is_public = self._is_public_endpoint(method, path)

        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):

            if is_public:
                headers = {
                    "Accept": "application/json"
                }
            else:
                headers = self._sign_request(method, path)

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs
                )

            if response.status_code != 429:
                response.raise_for_status()
                return response

            if attempt == _MAX_RATE_LIMIT_RETRIES:
                logger.warning(
                    "Kalshi 429 rate limit on %s %s",
                    method,
                    path
                )
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")

            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = (
                        _BASE_BACKOFF_SECONDS * (2 ** attempt)
                    )
            else:
                delay = (
                    _BASE_BACKOFF_SECONDS * (2 ** attempt)
                    + random.uniform(0, 0.25)
                )

            logger.warning(
                "Kalshi rate limit: retry %d/%d after %.2fs",
                attempt + 1,
                _MAX_RATE_LIMIT_RETRIES,
                delay
            )

            await asyncio.sleep(max(0, delay))

        raise RuntimeError("Kalshi request retries exhausted")

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None
    ) -> dict:

        response = await self._request_with_retry(
            "GET",
            path,
            params=params
        )

        return response.json()

    async def post(self, path: str, body: dict) -> dict:

        response = await self._request_with_retry(
            "POST",
            path,
            json=body
        )

        return response.json()

    async def delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None
    ) -> dict:

        response = await self._request_with_retry(
            "DELETE",
            path,
            params=params
        )

        return response.json()

    async def get_markets(
        self,
        params: Optional[Dict[str, Any]] = None
    ) -> dict:

        return await self.get("/markets", params=params)

    async def get_market(self, ticker: str) -> dict:

        return await self.get(f"/markets/{ticker}")

    async def get_balance(self) -> dict:

        return await self.get("/portfolio/balance")

    async def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        client_order_id: Optional[str] = None
    ) -> dict:
        """Submit an authenticated limit order.

        This method is for live trading only.
        Paper trading should use a separate simulator.
        """

        if side not in ("yes", "no"):
            raise ValueError("side must be 'yes' or 'no'")

        if count <= 0:
            raise ValueError("count must be positive")

        if not 1 <= price_cents <= 99:
            raise ValueError("price_cents must be 1-99")

        if side == "yes":
            book_side = "bid"
            yes_price = price_cents / 100.0
        else:
            book_side = "ask"
            yes_price = 1.0 - (price_cents / 100.0)

        body = {
            "ticker": ticker,
            "side": book_side,
            "count": str(count),
            "price": f"{yes_price:.4f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "client_order_id": (
                client_order_id or str(uuid.uuid4())
            )
        }

        logger.info(
            "Placing Kalshi order: %s %s x%d @ %dc",
            ticker,
            side.upper(),
            count,
            price_cents
        )

        return await self.post("/portfolio/orders", body)

    async def cancel_order(self, order_id: str) -> dict:

        return await self.delete(
            f"/portfolio/events/orders/{order_id}"
        )

    async def get_order(self, order_id: str) -> dict:

        return await self.get(
            f"/portfolio/orders/{order_id}"
        )

    async def get_orders(
        self,
        *,
        ticker: Optional[str] = None,
        status: Optional[str] = None
    ) -> dict:

        params: Dict[str, Any] = {}

        if ticker:
            params["ticker"] = ticker

        if status:
            params["status"] = status

        return await self.get(
            "/portfolio/orders",
            params=params
        )


def kalshi_credentials_present() -> bool:
    """Check whether authenticated API settings exist."""

    return bool(
        settings.KALSHI_API_KEY_ID
        and settings.KALSHI_PRIVATE_KEY_PATH
    )
