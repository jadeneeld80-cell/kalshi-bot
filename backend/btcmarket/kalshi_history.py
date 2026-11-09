"""
Real historical backfill for KXBTC15M using Kalshi's own market-candlesticks
endpoint (GET /series/{series}/markets/{ticker}/candlesticks).

Each KXBTC15M window is its own market ticker that only exists for ~15
minutes, so there's no single call that returns a multi-day history. Instead:
list every settled market in the series over the lookback window, then fetch
each one's 1-minute candlesticks and stitch them into one continuous series.
This is genuine historical Kalshi pricing (yes_bid/yes_ask/traded price) --
not something we have to wait to record live.
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from backend.config import settings
from backend.models.database import SessionLocal, KxBtcSnapshot, KxBtcCandle, BtcCandle
from backend.scanner.kalshi_api import KalshiScannerAPI

logger = logging.getLogger("trading_bot")

_api = KalshiScannerAPI()


def _dollars(field: Optional[dict], key: str) -> Optional[float]:
    if not field:
        return None
    val = field.get(key)
    return float(val) if val is not None else None


async def _list_settled_markets(min_close_ts: int, max_close_ts: int) -> List[dict]:
    markets: List[dict] = []
    cursor = None
    for _ in range(50):
        data = await _api.get_markets(
            series_ticker=settings.KXBTC15M_SERIES_TICKER,
            status="settled",
            min_close_ts=min_close_ts,
            max_close_ts=max_close_ts,
            limit=200,
            cursor=cursor,
        )
        page = data.get("markets", [])
        markets.extend(page)
        cursor = data.get("cursor")
        if not cursor or not page:
            break
    return markets


def _nearest_btc_price(db, at: datetime) -> Optional[float]:
    row = (
        db.query(BtcCandle)
        .filter(BtcCandle.granularity_seconds == 60, BtcCandle.open_time <= at)
        .order_by(BtcCandle.open_time.desc())
        .first()
    )
    return row.close if row else None


def backfill_missing_btc_price() -> int:
    """
    One-off repair: kxbtc_snapshots.btc_price was recorded at write-time from
    whatever btc_candles(60) held *then*. The 1-minute candle cache didn't
    exist for the snapshot poller's first ~3 weeks, so most early rows got
    btc_price=NULL permanently. Once btc_candles(60) has been backfilled
    further back (e.g. by the training pipeline), this fills those NULLs in
    using the same nearest-candle lookup, retroactively.
    """
    db = SessionLocal()
    try:
        rows = db.query(KxBtcSnapshot).filter(KxBtcSnapshot.btc_price.is_(None)).all()
        fixed = 0
        for row in rows:
            price = _nearest_btc_price(db, row.timestamp)
            if price is not None:
                row.btc_price = price
                fixed += 1
        db.commit()
        return fixed
    finally:
        db.close()


def get_kxbtc_candles(lookback: timedelta) -> List[dict]:
    """Real OHLC of the KXBTC15M YES price, stitched across rotating windows -- for the candlestick chart."""
    db = SessionLocal()
    try:
        since = datetime.utcnow() - lookback
        rows = (
            db.query(KxBtcCandle)
            .filter(KxBtcCandle.open_time >= since)
            .order_by(KxBtcCandle.open_time.asc())
            .all()
        )
        return [
            {
                "time": int(r.open_time.replace(tzinfo=timezone.utc).timestamp()),
                "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    finally:
        db.close()


async def backfill_kxbtc_history(lookback: timedelta) -> int:
    """
    Fetch every settled KXBTC15M market in the lookback window and stitch
    their 1-minute candlesticks into KxBtcSnapshot rows. Safe to re-run --
    skips tickers whose snapshots are already recorded.
    """
    now = datetime.utcnow()
    min_close_ts = int((now - lookback).timestamp())
    max_close_ts = int(now.timestamp())

    try:
        markets = await _list_settled_markets(min_close_ts, max_close_ts)
    except Exception as e:
        logger.warning(f"KXBTC15M history: failed to list settled markets: {e}")
        return 0

    if not markets:
        return 0

    db = SessionLocal()
    added = 0
    try:
        already_backfilled = {
            row.ticker for row in db.query(KxBtcSnapshot.ticker).distinct().all()
        }
        existing_candle_times = {row.open_time for row in db.query(KxBtcCandle.open_time).all()}

        for market in markets[:20]:
            ticker = market.get("ticker", "")
            if not ticker or ticker in already_backfilled:
                continue

            floor_strike = market.get("floor_strike")
            open_time = market.get("open_time")
            close_time = market.get("close_time")
            start_ts = int(datetime.fromisoformat(open_time.replace("Z", "+00:00")).timestamp()) if open_time else min_close_ts
            end_ts = int(datetime.fromisoformat(close_time.replace("Z", "+00:00")).timestamp()) if close_time else max_close_ts

            try:
                data = await _api.get_market_candlesticks(
                    settings.KXBTC15M_SERIES_TICKER, ticker,
                    start_ts=start_ts, end_ts=end_ts, period_interval=1,
                )
            except Exception as e:
                logger.debug(f"KXBTC15M history: candlesticks failed for {ticker}: {e}")
                continue

            for c in data.get("candlesticks", []):
                end_period_ts = c.get("end_period_ts")
                if not end_period_ts:
                    continue
                ts = datetime.fromtimestamp(end_period_ts, tz=timezone.utc).replace(tzinfo=None)

                yes_bid = _dollars(c.get("yes_bid"), "close_dollars")
                yes_ask = _dollars(c.get("yes_ask"), "close_dollars")
                if yes_bid is None and yes_ask is None:
                    continue
                yes_bid = yes_bid if yes_bid is not None else yes_ask
                yes_ask = yes_ask if yes_ask is not None else yes_bid

                db.add(KxBtcSnapshot(
                    ticker=ticker,
                    event_ticker=market.get("event_ticker", ""),
                    timestamp=ts,
                    floor_strike=floor_strike,
                    open_time=datetime.fromisoformat(open_time.replace("Z", "+00:00")).replace(tzinfo=None) if open_time else None,
                    close_time=datetime.fromisoformat(close_time.replace("Z", "+00:00")).replace(tzinfo=None) if close_time else None,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=1.0 - yes_ask,
                    no_ask=1.0 - yes_bid,
                    btc_price=_nearest_btc_price(db, ts),
                ))

                # Real OHLC of the YES contract price for the candlestick chart --
                # prefer actually-traded price, fall back to the bid/ask midpoint
                # range when no trade happened in that minute.
                price = c.get("price") or {}
                bid_dist, ask_dist = c.get("yes_bid") or {}, c.get("yes_ask") or {}
                o = _dollars(price, "open_dollars")
                h = _dollars(price, "high_dollars")
                l = _dollars(price, "low_dollars")
                cl = _dollars(price, "close_dollars")
                if o is None or h is None or l is None or cl is None:
                    bid_o, bid_c = _dollars(bid_dist, "open_dollars"), _dollars(bid_dist, "close_dollars")
                    ask_o, ask_c = _dollars(ask_dist, "open_dollars"), _dollars(ask_dist, "close_dollars")
                    o = ((bid_o or yes_bid) + (ask_o or yes_ask)) / 2
                    cl = ((bid_c or yes_bid) + (ask_c or yes_ask)) / 2
                    h, l = max(o, cl), min(o, cl)

                candle_open_time = ts - timedelta(minutes=1)
                if candle_open_time not in existing_candle_times:
                    db.add(KxBtcCandle(
                        open_time=candle_open_time, open=o, high=h, low=l, close=cl,
                        volume=float(c.get("volume_fp", 0) or c.get("volume", 0) or 0),
                    ))
                    existing_candle_times.add(candle_open_time)
                added += 1

            already_backfilled.add(ticker)
            db.commit()

        if added:
            logger.info(f"KXBTC15M history: backfilled {added} real candlestick snapshots across {len(markets)} settled markets")
        return added
    finally:
        db.close()
