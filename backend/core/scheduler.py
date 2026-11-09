"""Background scheduler for Kalshi weather temperature trading."""
import asyncio
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import func
import logging

from backend.config import settings
from backend.models.database import SessionLocal, Trade, BotState, Signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trading_bot")

# Persist logs to bot.log so you can review what happened after the fact
_file_handler = logging.FileHandler("bot.log", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
logger.addHandler(_file_handler)

scheduler: Optional[AsyncIOScheduler] = None

event_log: List[dict] = []
MAX_LOG_SIZE = 200

# For jobs that run at 1Hz, logging every failure at WARNING risks up to 3600
# lines/hour during a real outage. This throttles: always log the first
# failure immediately (visibility), then at most once per _THROTTLE_SECONDS
# while it keeps failing, so a persistent break is still visible without
# flooding bot.log.
_THROTTLE_SECONDS = 60.0
_last_warned_at: Dict[str, float] = {}


def _log_throttled_warning(key: str, message: str) -> None:
    import time
    now = time.monotonic()
    last = _last_warned_at.get(key)
    if last is None or (now - last) >= _THROTTLE_SECONDS:
        _last_warned_at[key] = now
        logger.warning(message)
        logger.exception(f"Full traceback for {key}")

# ---------------------------------------------------------------------------
# Signal confirmation tracking
#
# We require a signal to survive CONFIRM_MINUTES before we trade it.
# This prevents acting on a transient spike and gives us time to sanity-check.
#
# _signal_first_seen: market_id -> datetime when the signal first exceeded threshold
# _scan_count: how many scans have completed (never trade on scan #1)
# ---------------------------------------------------------------------------
_signal_first_seen: Dict[str, datetime] = {}
_scan_count: int = 0


def log_event(event_type: str, message: str, data: dict = None):
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": event_type,
        "message": message,
        "data": data or {}
    }
    event_log.append(event)
    while len(event_log) > MAX_LOG_SIZE:
        event_log.pop(0)

    log_func = {
        "error": logger.error,
        "warning": logger.warning,
        "success": logger.info,
        "trade": logger.info,
    }.get(event_type, logger.info)
    log_func(f"[{event_type.upper()}] {message}")


def get_recent_events(limit: int = 50) -> List[dict]:
    return event_log[-limit:]


async def _send_discord(title: str, description: str, color: int = 0x5865F2, fields: list = None):
    """Post a rich embed to Discord. Silent no-op if DISCORD_WEBHOOK_URL is not set."""
    url = settings.DISCORD_WEBHOOK_URL
    if not url:
        return
    try:
        import httpx
        embed: dict = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if fields:
            embed["fields"] = fields
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json={"embeds": [embed]})
    except Exception as e:
        logger.debug(f"Discord notification failed: {e}")


async def _place_kalshi_order(
    ticker: str,
    side: str,
    size_usd: float,
    price: float,
) -> Optional[str]:
    """
    Submit a real limit order to Kalshi.
    price is the YES/NO ask as a fraction (0-1); converted to cents internally.
    count = floor(size_usd / price) — each contract pays $1 at settlement.
    """
    try:
        from backend.data.kalshi_client import KalshiClient
        price_cents = max(1, min(99, round(price * 100)))
        count = max(1, math.floor(size_usd / (price_cents / 100)))

        client = KalshiClient()
        response = await client.place_order(
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
        )
        order = response.get("order", response)
        order_id = order.get("order_id") or order.get("id") or str(order)
        logger.info(f"Order placed: {order_id} | {ticker} {side.upper()} x{count} @ {price_cents}c")
        return order_id
    except Exception as e:
        logger.error(f"Order placement failed for {ticker}: {e}")
        return None


async def scan_and_trade_job():
    """
    Main scan job. On each run:
      1. Fetch Kalshi markets + generate ensemble signals
      2. For signals that pass the edge threshold:
           - First time seen → mark "watching", log reasoning, do NOT trade yet
           - Seen for CONFIRM_MINUTES+ → trade (or log "would trade" in sim)
      3. Remove stale confirmations (signal dropped below threshold)
    """
    if not settings.WEATHER_BOT_ENABLED:
        return

    global _scan_count
    _scan_count += 1

    mode = "SIM" if settings.SIMULATION_MODE else "LIVE"
    log_event("info", f"[{mode}] Scan #{_scan_count} — fetching Kalshi weather markets...")

    try:
        from backend.core.weather_signals import scan_for_weather_signals
        signals = await scan_for_weather_signals()
        actionable = [s for s in signals if s.passes_threshold]

        log_event("data",
            f"Scan #{_scan_count}: {len(signals)} markets | {len(actionable)} above threshold",
            {"scan": _scan_count, "total": len(signals), "actionable": len(actionable)}
        )

        now = datetime.utcnow()
        confirm_delta = timedelta(minutes=settings.CONFIRM_MINUTES)

        # Update confirmation tracking
        current_actionable_ids = {s.market.market_id for s in actionable}

        # Register newly actionable signals
        for s in actionable:
            mid = s.market.market_id
            if mid not in _signal_first_seen:
                _signal_first_seen[mid] = now
                wait_mins = settings.CONFIRM_MINUTES
                log_event("info",
                    f"New signal: {s.market.city_name} high {s.market.direction} "
                    f"{s.market.threshold_f:.0f}F | edge {s.edge:+.1%} | "
                    f"model {s.model_probability:.0%} vs market {s.market_probability:.0%} | "
                    f"Watching for {wait_mins}m before trading",
                    {
                        "ticker": mid,
                        "edge": s.edge,
                        "direction": s.direction,
                        "city": s.market.city_name,
                        "first_seen": now.isoformat(),
                    }
                )
                await _send_discord(
                    title=f"👀 Watching: {s.market.city_name}",
                    description=(
                        f"High **{s.market.direction}** {s.market.threshold_f:.0f}°F "
                        f"on {s.market.target_date}\n"
                        f"Model: **{s.model_probability:.0%}** vs Market: **{s.market_probability:.0%}**\n"
                        f"Edge: **{s.edge:+.1%}** — will trade in **{wait_mins}m** if signal holds"
                    ),
                    color=0xFEE75C,
                )

        # Drop signals that fell below threshold
        stale = [mid for mid in list(_signal_first_seen) if mid not in current_actionable_ids]
        for mid in stale:
            del _signal_first_seen[mid]
            log_event("info", f"Signal dropped below threshold, removed from watch: {mid}")

        # Determine which confirmed signals are ready to trade
        confirmed = [
            s for s in actionable
            if (now - _signal_first_seen.get(s.market.market_id, now)) >= confirm_delta
        ]

        if not confirmed:
            watching = len(_signal_first_seen) - len(confirmed)
            if watching > 0:
                oldest = min(_signal_first_seen.values())
                secs_remaining = max(0, (confirm_delta - (now - oldest)).total_seconds())
                log_event("info",
                    f"Watching {watching} signal(s) — oldest confirmed in {secs_remaining/60:.1f}m"
                )
            else:
                log_event("info", "No confirmed signals ready to trade")
            return

        log_event("info", f"{len(confirmed)} signal(s) confirmed — evaluating for execution")

        db = SessionLocal()
        try:
            state = db.query(BotState).first()
            if not state or not state.is_running:
                log_event("info", "Bot paused — skipping execution")
                return

            # Daily loss circuit breaker
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            daily_pnl = db.query(func.coalesce(func.sum(Trade.pnl), 0.0)).filter(
                Trade.settled == True,
                Trade.settlement_time >= today_start
            ).scalar()

            if daily_pnl <= -settings.DAILY_LOSS_LIMIT:
                log_event("warning", f"Daily loss limit hit (${daily_pnl:.2f}) — halting all trades today")
                await _send_discord(
                    title="🚨 Daily Loss Limit Hit — Bot Halted",
                    description=(
                        f"Today's settled P&L: **${daily_pnl:.2f}**\n"
                        f"Limit: **-${settings.DAILY_LOSS_LIMIT:.0f}**\n"
                        f"No more trades will be placed today."
                    ),
                    color=0xED4245,
                )
                return

            # Total exposure check
            weather_exposure = db.query(func.coalesce(func.sum(Trade.size), 0.0)).filter(
                Trade.settled == False
            ).scalar()

            if weather_exposure >= settings.MAX_WEATHER_ALLOCATION:
                log_event("info",
                    f"Max allocation reached: ${weather_exposure:.0f}/${settings.MAX_WEATHER_ALLOCATION:.0f}"
                )
                return

            total_pending = db.query(Trade).filter(Trade.settled == False).count()
            if total_pending >= settings.MAX_TOTAL_PENDING_TRADES:
                log_event("info", f"Max pending trades reached ({total_pending})")
                return

            trades_executed = 0
            for signal in confirmed[:settings.MAX_TRADES_PER_SCAN]:
                mid = signal.market.market_id

                # Skip if already holding this market
                existing = db.query(Trade).filter(
                    Trade.market_ticker == mid,
                    Trade.settled == False,
                ).first()
                if existing:
                    log_event("info", f"Already holding {mid} — skipping")
                    continue

                trade_size = min(signal.suggested_size, settings.MAX_TRADE_SIZE)
                if trade_size < 1.0:
                    log_event("info",
                        f"Trade size ${trade_size:.2f} too small for {mid} — skipping "
                        f"(bankroll=${state.bankroll:.2f}, edge={signal.edge:.1%})"
                    )
                    continue

                if state.bankroll < trade_size:
                    log_event("warning", f"Bankroll ${state.bankroll:.2f} below trade size ${trade_size:.2f}")
                    break

                entry_price = signal.market.yes_price if signal.direction == "yes" else signal.market.no_price
                confirm_age_mins = (now - _signal_first_seen[mid]).total_seconds() / 60

                # Pre-trade reasoning log (always, even in sim)
                log_event("info",
                    f"PRE-TRADE: {signal.market.city_name} high {signal.market.direction} "
                    f"{signal.market.threshold_f:.0f}F on {signal.market.target_date} | "
                    f"Buy {signal.direction.upper()} @ {entry_price:.0%} | "
                    f"Model {signal.model_probability:.0%} vs market {signal.market_probability:.0%} | "
                    f"Edge {signal.edge:+.1%} | Kelly ${trade_size:.2f} | "
                    f"Ensemble: {signal.ensemble_mean:.1f}F ±{signal.ensemble_std:.1f}F "
                    f"({signal.ensemble_members} members, {signal.confidence:.0%} agreement) | "
                    f"Signal age: {confirm_age_mins:.1f}m",
                    {
                        "ticker": mid,
                        "city": signal.market.city_name,
                        "direction": signal.direction,
                        "entry_price": entry_price,
                        "edge": signal.edge,
                        "model_prob": signal.model_probability,
                        "market_prob": signal.market_probability,
                        "size": trade_size,
                        "ensemble_mean": signal.ensemble_mean,
                        "ensemble_members": signal.ensemble_members,
                        "confirm_age_mins": confirm_age_mins,
                    }
                )

                order_id = None
                if not settings.SIMULATION_MODE:
                    order_id = await _place_kalshi_order(
                        ticker=mid,
                        side=signal.direction,
                        size_usd=trade_size,
                        price=entry_price,
                    )
                    if order_id is None:
                        log_event("error", f"Order placement failed for {mid} — skipping trade")
                        continue

                trade = Trade(
                    market_ticker=mid,
                    platform="kalshi",
                    event_slug=signal.market.slug,
                    direction=signal.direction,
                    entry_price=entry_price,
                    size=trade_size,
                    order_id=order_id,
                    order_status="resting" if order_id else "simulated",
                    model_probability=signal.model_probability,
                    market_price_at_entry=signal.market_probability,
                    edge_at_entry=signal.edge,
                )
                db.add(trade)
                db.flush()

                # Link to signal record for calibration tracking
                matching_signal = db.query(Signal).filter(
                    Signal.market_ticker == mid,
                    Signal.executed == False,
                ).order_by(Signal.timestamp.desc()).first()
                if matching_signal:
                    matching_signal.executed = True
                    trade.signal_id = matching_signal.id

                state.total_trades += 1
                trades_executed += 1

                # Remove from confirmation watch — we've acted on it
                _signal_first_seen.pop(mid, None)

                live_tag = "[SIM] " if settings.SIMULATION_MODE else "[LIVE] "
                log_event("trade",
                    f"{live_tag}{signal.market.city_name}: "
                    f"{signal.direction.upper()} ${trade_size:.2f} @ {entry_price:.0%} | "
                    f"high {signal.market.direction} {signal.market.threshold_f:.0f}F | "
                    f"edge {signal.edge:+.1%}",
                    {
                        "ticker": mid,
                        "city": signal.market.city_name,
                        "direction": signal.direction,
                        "size": trade_size,
                        "edge": signal.edge,
                        "order_id": order_id,
                        "simulation": settings.SIMULATION_MODE,
                    }
                )
                mode_icon = "🔵" if settings.SIMULATION_MODE else "⚡"
                mode_label = "SIM" if settings.SIMULATION_MODE else "LIVE"
                await _send_discord(
                    title=f"{mode_icon} [{mode_label}] Trade: {signal.market.city_name}",
                    description=(
                        f"Bought **{signal.direction.upper()}** on "
                        f"high **{signal.market.direction}** {signal.market.threshold_f:.0f}°F "
                        f"on {signal.market.target_date}"
                    ),
                    color=0x57F287 if not settings.SIMULATION_MODE else 0x5865F2,
                    fields=[
                        {"name": "Size", "value": f"${trade_size:.2f}", "inline": True},
                        {"name": "Entry Price", "value": f"{entry_price:.0%}", "inline": True},
                        {"name": "Edge", "value": f"{signal.edge:+.1%}", "inline": True},
                        {"name": "Model Prob", "value": f"{signal.model_probability:.0%}", "inline": True},
                        {"name": "Market Price", "value": f"{signal.market_probability:.0%}", "inline": True},
                        {"name": "Signal Age", "value": f"{confirm_age_mins:.1f}m", "inline": True},
                        {
                            "name": "Ensemble Forecast",
                            "value": (
                                f"{signal.ensemble_mean:.1f}°F ±{signal.ensemble_std:.1f}°F "
                                f"({signal.ensemble_members} members, {signal.confidence:.0%} agreement)"
                            ),
                            "inline": False,
                        },
                    ],
                )

            state.last_run = datetime.utcnow()
            db.commit()

            if trades_executed > 0:
                log_event("success", f"Executed {trades_executed} trade(s) this scan")
            else:
                log_event("info", "No new trades executed")

        finally:
            db.close()

    except Exception as e:
        log_event("error", f"Scan error: {e}")
        logger.exception("Error in scan_and_trade_job")


async def whale_scan_job():
    """Poll Kalshi's trade firehose for fills over MIN_WHALE_TRADE_USD."""
    if not settings.WHALE_SCANNER_ENABLED:
        return
    try:
        from backend.scanner.whale_scanner import scan_for_whales
        whales = await scan_for_whales()
        for w in whales:
            log_event("data",
                f"🐋 Whale trade: {w.title} | {w.taker_side.upper()} x{w.count} @ {w.price_cents}c "
                f"= ${w.notional_usd:,.0f} | {w.link}",
                {
                    "ticker": w.ticker,
                    "title": w.title,
                    "side": w.taker_side,
                    "count": w.count,
                    "price_cents": w.price_cents,
                    "notional_usd": w.notional_usd,
                    "link": w.link,
                }
            )
    except Exception as e:
        logger.debug(f"Whale scan error: {e}")


async def btc_poll_job():
    """Poll the live KXBTC15M market + record a snapshot every KXBTC_POLL_INTERVAL_SECONDS (1Hz)."""
    if not settings.BTC_MARKET_ENABLED:
        return
    try:
        from backend.btcmarket.kalshi_poll import poll_and_record
        await poll_and_record()
    except Exception as e:
        # Throttled warning, not silent debug -- runs at 1Hz, so an
        # unthrottled warning-per-failure would flood bot.log during a real
        # outage. This still surfaces a persistent failure (first occurrence
        # immediately, then at most once/minute) instead of hiding it.
        _log_throttled_warning("btc_market_poll", f"BTC market poll error: {e}")


async def btc_ticker_poll_job():
    """Poll Coinbase's latest trade price every BTC_TICKER_POLL_SECONDS (1Hz) to keep the forming candle live."""
    if not settings.BTC_MARKET_ENABLED:
        return
    try:
        from backend.btcmarket.candles import poll_live_price
        await poll_live_price()
    except Exception as e:
        # Same throttled-warning rationale as btc_poll_job above (1Hz cadence).
        _log_throttled_warning("btc_ticker_poll", f"BTC ticker poll error: {e}")


async def btc_orderbook_poll_job():
    """Record top-N orderbook depth for the open KXBTC15M window (own cadence, separate from the 1s top-of-book poll)."""
    if not settings.BTC_MARKET_ENABLED:
        return
    try:
        from backend.btcmarket.orderbook import record_orderbook_snapshot
        await record_orderbook_snapshot()
    except Exception as e:
        # WARNING not debug -- runs every 5s (not 1Hz), so a persistent failure
        # surfaces promptly without 1Hz-scale log spam. Same rationale as the
        # settlement/retrain job fixes above.
        logger.warning(f"BTC orderbook depth poll error: {e}")
        logger.exception("Full traceback for BTC orderbook depth poll error")


async def btc_history_refresh_job():
    """Backfill any newly-settled KXBTC15M windows into real historical candlestick data."""
    if not settings.BTC_MARKET_ENABLED:
        return
    try:
        from backend.btcmarket.kalshi_history import backfill_kxbtc_history
        from datetime import timedelta
        await backfill_kxbtc_history(timedelta(hours=1))
    except Exception as e:
        # WARNING not debug -- runs every 5min, safe frequency for visible logging.
        logger.warning(f"BTC history refresh error: {e}")
        logger.exception("Full traceback for BTC history refresh error")


async def btc_paper_signal_job():
    """Check the current KXBTC15M window against the trained model; journal a simulated trade if there's a real edge."""
    if not settings.BTC_MARKET_ENABLED or not settings.BTC_PAPER_TRADING_ENABLED:
        return
    try:
        from backend.core.btc_paper_trading import generate_paper_trade
        trade = await generate_paper_trade()
        if trade:
            log_event("trade",
                f"[PAPER] BTC {trade['ticker']}: {trade['direction'].upper()} ${trade['size']:.2f} "
                f"@ {trade['entry_price']:.0%} | edge {trade['edge']:+.1%}",
                trade,
            )
    except Exception as e:
        # WARNING not debug -- runs every 15s, safe frequency. A silently
        # failing signal job means the bot simply stops trading with no
        # visible reason, the exact failure mode this project has been
        # burned by before.
        logger.warning(f"BTC paper signal error: {e}")
        logger.exception("Full traceback for BTC paper signal error")


async def btc_paper_settlement_job():
    """Settle any paper trades whose window has closed, against Kalshi's real result."""
    if not settings.BTC_MARKET_ENABLED or not settings.BTC_PAPER_TRADING_ENABLED:
        return
    try:
        from backend.core.btc_paper_trading import settle_paper_trades
        settled = await settle_paper_trades()
        if settled:
            wins = sum(1 for t in settled if t["result"] == "win")
            pnl = sum(t["pnl"] for t in settled if t["pnl"] is not None)
            log_event("data", f"[PAPER] Settled {len(settled)} BTC paper trade(s): {wins}W/{len(settled)-wins}L, P&L ${pnl:+.2f}")
    except Exception as e:
        # WARNING, not debug -- this exact "broad except swallows a real error
        # at an invisible log level" pattern has hidden real bugs twice already
        # this session (a DetachedInstanceError in this same settlement path,
        # and a similar one in orderbook depth recording). A settlement job
        # that silently does nothing for 25+ minutes should never be invisible.
        logger.warning(f"BTC paper settlement error: {e}")
        logger.exception("Full traceback for BTC paper settlement error")


async def btc_retrain_job():
    """Walk-forward retrain the KXBTC15M model on all settled windows recorded so far."""
    if not settings.BTC_MARKET_ENABLED:
        return
    try:
        from backend.core.btc_model_training import train_btc_model, fit_live_recalibration
        report = await train_btc_model()
        if report.get("status") == "ok":
            log_event("success", f"BTC model retrained: {report['note']}")
        else:
            log_event("info", f"BTC model retrain skipped: {report.get('note', report.get('status'))}")

        # Refresh the secondary live-outcome recalibration each retrain cycle too,
        # so it keeps adapting as more paper-trading data accumulates rather than
        # being a one-time fit that goes stale. No-op (returns "insufficient_data")
        # until enough live trades exist -- see fit_live_recalibration's docstring.
        live_calib_report = fit_live_recalibration()
        if live_calib_report.get("status") == "ok":
            log_event("success", f"BTC live recalibration refreshed: n={live_calib_report['n']} live trades")
        else:
            log_event("info", f"BTC live recalibration skipped: {live_calib_report.get('note', live_calib_report.get('status'))}")
    except Exception as e:
        # WARNING, not debug -- same "broad except swallows a real error at an
        # invisible log level" pattern already found and fixed in
        # btc_paper_settlement_job. A retrain that silently fails every 6h
        # should never be invisible.
        logger.warning(f"BTC retrain error: {e}")
        logger.exception("Full traceback for BTC retrain error")


async def btc_feature_parity_job():
    """Check whether live-computed features are drifting from what the offline pipeline would compute for the same trades -- catches the next train/serve skew bug automatically."""
    if not settings.BTC_MARKET_ENABLED or not settings.BTC_PAPER_TRADING_ENABLED:
        return
    try:
        from backend.core.btc_feature_parity import check_feature_parity
        report = await check_feature_parity(lookback_hours=48)
        if report.get("status") != "ok":
            return
        if report["flagged_features"]:
            log_event("warning",
                f"BTC feature parity: {report['flagged_features']} exceed tolerance across "
                f"{report['trades_checked']} trades -- possible live/offline skew, check /api/btc/feature-parity",
                {"flagged": report["flagged_features"]},
            )
        else:
            log_event("info", f"BTC feature parity OK across {report['trades_checked']} trades ({report['trades_skipped_no_match']} skipped, no offline match)")
    except Exception as e:
        # WARNING not debug -- runs every 3h, this is the safety net that
        # catches train/serve skew bugs; if it silently breaks, that safety
        # net is silently gone.
        logger.warning(f"BTC feature parity check error: {e}")
        logger.exception("Full traceback for BTC feature parity check error")


async def btc_hourly_analysis_job():
    """Every hour, report the paper-trading bot's own performance -- win rate, PnL, calibration -- to logs + Discord."""
    if not settings.BTC_MARKET_ENABLED or not settings.BTC_PAPER_TRADING_ENABLED:
        return
    try:
        from backend.core.btc_paper_trading import paper_trading_report
        hour = paper_trading_report(lookback_hours=1)
        life = paper_trading_report(lookback_hours=None)

        log_event("data",
            f"[BTC PAPER HOURLY] Past hour: {hour['trades_settled']} settled "
            f"({hour['wins']}W/{hour['losses']}L, win rate {hour['win_rate']}), P&L ${hour['total_pnl']:+.2f} | "
            f"Lifetime: {life['lifetime_trades']} trades, ${life['lifetime_pnl']:+.2f}, bankroll ${life['bankroll']:.2f}, "
            f"{life['trades_pending']} pending",
            {"hour": hour, "lifetime": life},
        )
        await _send_discord(
            title="BTC Paper-Trading Hourly Report",
            description=f"Bankroll: **${life['bankroll']:.2f}** | Lifetime P&L: **${life['lifetime_pnl']:+.2f}**",
            color=0x57F287 if hour["total_pnl"] >= 0 else 0xED4245,
            fields=[
                {"name": "Past Hour", "value": f"{hour['trades_settled']} settled, {hour['win_rate'] or 0:.0%} win rate, ${hour['total_pnl']:+.2f}", "inline": False},
                {"name": "Pending", "value": str(hour["trades_pending"]), "inline": True},
                {"name": "Model Calibration (Brier, lifetime)", "value": str(life["model_brier"]), "inline": True},
            ],
        )
    except Exception as e:
        logger.debug(f"BTC hourly analysis error: {e}")


async def settlement_job():
    """Check and settle pending Kalshi trades."""
    log_event("info", "Checking trade settlements...")

    try:
        from backend.core.settlement import settle_pending_trades, update_bot_state_with_settlements

        db = SessionLocal()
        try:
            pending_count = db.query(Trade).filter(Trade.settled == False).count()
            if pending_count == 0:
                log_event("data", "No pending trades to settle")
                return

            log_event("data", f"Checking {pending_count} pending trades")
            settled = await settle_pending_trades(db)

            if settled:
                await update_bot_state_with_settlements(db, settled)
                wins = sum(1 for t in settled if t.result == "win")
                losses = sum(1 for t in settled if t.result == "loss")
                pnl = sum(t.pnl for t in settled if t.pnl is not None)
                log_event("success",
                    f"Settled {len(settled)} trades: {wins}W/{losses}L, P&L ${pnl:+.2f}",
                    {"wins": wins, "losses": losses, "pnl": pnl}
                )
                result_icon = "✅" if pnl >= 0 else "❌"
                details = "\n".join(
                    f"• {t.market_ticker}: **{t.result.upper()}** ${t.pnl:+.2f}"
                    for t in settled if t.pnl is not None
                )
                await _send_discord(
                    title=f"{result_icon} {len(settled)} Trade(s) Settled",
                    description=f"**{wins}W / {losses}L** | Net P&L: **${pnl:+.2f}**\n\n{details}",
                    color=0x57F287 if pnl >= 0 else 0xED4245,
                )
            else:
                log_event("info", "No trades ready for settlement yet")
        finally:
            db.close()

    except Exception as e:
        log_event("error", f"Settlement error: {e}")
        logger.exception("Error in settlement_job")


async def heartbeat_job():
    db = None
    try:
        db = SessionLocal()
        state = db.query(BotState).first()
        pending = db.query(Trade).filter(Trade.settled == False).count()
        watching = len(_signal_first_seen)
        if state:
            mode = "SIM" if settings.SIMULATION_MODE else "LIVE"
            log_event("data",
                f"[{mode}] Bankroll ${state.bankroll:.2f} | "
                f"Pending {pending} trades | Watching {watching} signals",
                {
                    "bankroll": state.bankroll,
                    "pending": pending,
                    "watching": watching,
                    "scan_count": _scan_count,
                    "is_running": state.is_running,
                }
            )
    except Exception as e:
        log_event("warning", f"Heartbeat failed: {e}")
    finally:
        if db:
            db.close()


async def hourly_summary_job():
    """Send a Discord summary every hour — what the bot is watching and why it isn't trading."""
    db = None
    try:
        db = SessionLocal()
        state = db.query(BotState).first()
        pending = db.query(Trade).filter(Trade.settled == False).count()
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_pnl = db.query(func.coalesce(func.sum(Trade.pnl), 0.0)).filter(
            Trade.settled == True,
            Trade.settlement_time >= today_start,
        ).scalar()
        bankroll = state.bankroll if state else settings.INITIAL_BANKROLL
        mode = "SIM" if settings.SIMULATION_MODE else "LIVE"

        watching_lines = []
        now = datetime.utcnow()
        for mid, first_seen in _signal_first_seen.items():
            age_mins = (now - first_seen).total_seconds() / 60
            watching_lines.append(f"• `{mid}` — watching {age_mins:.0f}m")

        if not watching_lines:
            watching_text = "No signals above threshold right now"
        else:
            watching_text = "\n".join(watching_lines)

        reasons = []
        if pending >= settings.MAX_TOTAL_PENDING_TRADES:
            reasons.append(f"Max pending trades reached ({pending}/{settings.MAX_TOTAL_PENDING_TRADES})")
        exposure = db.query(func.coalesce(func.sum(Trade.size), 0.0)).filter(Trade.settled == False).scalar()
        if exposure >= settings.MAX_WEATHER_ALLOCATION:
            reasons.append(f"Max exposure reached (${exposure:.0f}/${settings.MAX_WEATHER_ALLOCATION:.0f})")
        if not _signal_first_seen:
            reasons.append(f"No edge ≥ {settings.MIN_EDGE_THRESHOLD:.0%} in current scan")
        elif all(
            (now - ts).total_seconds() / 60 < settings.CONFIRM_MINUTES
            for ts in _signal_first_seen.values()
        ):
            reasons.append(f"Signals still in {settings.CONFIRM_MINUTES}m confirmation window")

        reason_text = "\n".join(f"• {r}" for r in reasons) if reasons else "Waiting for next scan"

        await _send_discord(
            title=f"📊 [{mode}] Hourly Summary",
            description=(
                f"**Scan #{_scan_count}** | Bankroll: **${bankroll:.2f}** | "
                f"Daily P&L: **${daily_pnl:+.2f}** | Pending: **{pending}**"
            ),
            color=0x5865F2,
            fields=[
                {"name": "Currently Watching", "value": watching_text, "inline": False},
                {"name": "Why No New Trades", "value": reason_text, "inline": False},
            ],
        )
    except Exception as e:
        logger.debug(f"Hourly summary failed: {e}")
    finally:
        if db:
            db.close()


async def _startup_api_check():
    """Verify all required APIs are reachable before the first scan."""
    log_event("info", "Running startup API check...")

    # 1. Kalshi auth + balance
    try:
        from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present
        if not kalshi_credentials_present():
            log_event("warning", "Kalshi credentials missing — set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        else:
            client = KalshiClient()
            bal = await client.get_balance()
            available_cents = bal.get("balance", 0)
            available = available_cents / 100
            log_event("success",
                f"Kalshi auth OK | Available: ${available:.2f}",
                {"available_usd": available}
            )
    except Exception as e:
        log_event("error", f"Kalshi API check failed: {e}")

    # 2. Open-Meteo ensemble (spot-check NYC)
    try:
        from backend.data.weather import fetch_ensemble_forecast
        from datetime import date
        forecast = await fetch_ensemble_forecast("nyc", date.today())
        if forecast:
            log_event("success",
                f"Open-Meteo OK | NYC today: {forecast.mean_high:.1f}F ±{forecast.std_high:.1f}F "
                f"({forecast.num_members} members)",
                {"mean_high": forecast.mean_high, "members": forecast.num_members}
            )
        else:
            log_event("warning", "Open-Meteo returned no data for NYC — check network")
    except Exception as e:
        log_event("error", f"Open-Meteo check failed: {e}")

    # 3. Kalshi market availability
    try:
        from backend.data.kalshi_markets import fetch_kalshi_weather_markets
        city_keys = [c.strip() for c in settings.WEATHER_CITIES.split(",") if c.strip()]
        markets = await fetch_kalshi_weather_markets(city_keys)
        log_event("success",
            f"Kalshi markets OK | {len(markets)} open markets across {len(city_keys)} cities",
            {"market_count": len(markets), "cities": city_keys}
        )
    except Exception as e:
        log_event("error", f"Kalshi market fetch failed: {e}")

    log_event("info",
        f"Startup check complete | Mode: {'SIMULATION' if settings.SIMULATION_MODE else 'LIVE'} | "
        f"Confirm window: {settings.CONFIRM_MINUTES}m | "
        f"Edge threshold: {settings.MIN_EDGE_THRESHOLD:.0%}"
    )


def start_scheduler():
    global scheduler

    if scheduler is not None and scheduler.running:
        return

    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        scan_and_trade_job,
        IntervalTrigger(seconds=settings.SCAN_INTERVAL_SECONDS),
        id="weather_scan",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        settlement_job,
        IntervalTrigger(seconds=settings.SETTLEMENT_INTERVAL_SECONDS),
        id="settlement_check",
        replace_existing=True,
        max_instances=1,
    )

    if settings.WHALE_SCANNER_ENABLED:
        scheduler.add_job(
            whale_scan_job,
            IntervalTrigger(seconds=settings.WHALE_SCAN_INTERVAL_SECONDS),
            id="whale_scan",
            replace_existing=True,
            max_instances=1,
        )

    if settings.BTC_MARKET_ENABLED:
        scheduler.add_job(
            btc_poll_job,
            IntervalTrigger(seconds=settings.KXBTC_POLL_INTERVAL_SECONDS),
            id="btc_market_poll",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=2,
        )
        scheduler.add_job(
            btc_ticker_poll_job,
            IntervalTrigger(seconds=settings.BTC_TICKER_POLL_SECONDS),
            id="btc_ticker_poll",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=2,
        )
        scheduler.add_job(
            btc_orderbook_poll_job,
            IntervalTrigger(seconds=settings.KXBTC_ORDERBOOK_POLL_INTERVAL_SECONDS),
            id="btc_orderbook_poll",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=5,
        )
        scheduler.add_job(
            btc_history_refresh_job,
            IntervalTrigger(seconds=settings.KXBTC_HISTORY_REFRESH_SECONDS),
            id="btc_history_refresh",
            replace_existing=True,
            max_instances=1,
        )

    if settings.BTC_MARKET_ENABLED and settings.BTC_PAPER_TRADING_ENABLED:
        scheduler.add_job(
            btc_paper_signal_job,
            IntervalTrigger(seconds=settings.BTC_PAPER_SIGNAL_INTERVAL_SECONDS),
            id="btc_paper_signal",
            replace_existing=True,
            max_instances=1,
        )
        scheduler.add_job(
            btc_paper_settlement_job,
            IntervalTrigger(seconds=settings.BTC_PAPER_SETTLEMENT_INTERVAL_SECONDS),
            id="btc_paper_settlement",
            replace_existing=True,
            max_instances=1,
        )
        scheduler.add_job(
            btc_hourly_analysis_job,
            IntervalTrigger(seconds=settings.BTC_HOURLY_ANALYSIS_INTERVAL_SECONDS),
            id="btc_hourly_analysis",
            replace_existing=True,
            max_instances=1,
        )

    if settings.BTC_MARKET_ENABLED:
        scheduler.add_job(
            btc_retrain_job,
            IntervalTrigger(seconds=settings.BTC_RETRAIN_INTERVAL_SECONDS),
            id="btc_retrain",
            replace_existing=True,
            max_instances=1,
        )

    if settings.BTC_MARKET_ENABLED and settings.BTC_PAPER_TRADING_ENABLED:
        scheduler.add_job(
            btc_feature_parity_job,
            IntervalTrigger(hours=3),
            id="btc_feature_parity",
            replace_existing=True,
            max_instances=1,
        )

    scheduler.add_job(
        heartbeat_job,
        IntervalTrigger(minutes=1),
        id="heartbeat",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        hourly_summary_job,
        IntervalTrigger(hours=1),
        id="hourly_summary",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.start()

    mode = "SIMULATION" if settings.SIMULATION_MODE else "*** LIVE TRADING ***"
    log_event("success", f"Scheduler started [{mode}] — scan every {settings.SCAN_INTERVAL_SECONDS}s, confirm signals for {settings.CONFIRM_MINUTES}m before trading", {
        "scan_interval_s": settings.SCAN_INTERVAL_SECONDS,
        "confirm_minutes": settings.CONFIRM_MINUTES,
        "min_edge": f"{settings.MIN_EDGE_THRESHOLD:.0%}",
        "simulation": settings.SIMULATION_MODE,
    })

    # Run API check then first scan — no trades on first scan since nothing is confirmed yet
    async def _startup_sequence():
        await _startup_api_check()
        if settings.BTC_MARKET_ENABLED:
            try:
                from backend.btcmarket.candles import ensure_recent_candles
                from datetime import timedelta
                added_1m = await ensure_recent_candles(60, timedelta(days=1))
                added_1h = await ensure_recent_candles(3600, timedelta(days=30))
                log_event("success", f"BTC candles backfilled: +{added_1m} 1m, +{added_1h} 1h")
            except Exception as e:
                log_event("warning", f"BTC candle backfill failed: {e}")

            try:
                from backend.btcmarket.kalshi_history import backfill_kxbtc_history
                from datetime import timedelta
                added_snapshots = await backfill_kxbtc_history(timedelta(hours=6))
                log_event("success", f"KXBTC15M real history backfilled: +{added_snapshots} snapshots")
            except Exception as e:
                log_event("warning", f"KXBTC15M history backfill failed: {e}")

            try:
                from backend.btcmarket.kalshi_history import backfill_missing_btc_price
                fixed = backfill_missing_btc_price()
                if fixed:
                    log_event("success", f"KXBTC15M snapshots: repaired {fixed} missing btc_price values")
            except Exception as e:
                log_event("warning", f"btc_price repair failed: {e}")

            if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "btcmarket", "artifacts", "kxbtc15m_xgb.json")):
                try:
                    await btc_retrain_job()
                except Exception as e:
                    log_event("warning", f"Initial BTC model training failed: {e}")
        await scan_and_trade_job()

    asyncio.create_task(_startup_sequence())


def stop_scheduler():
    global scheduler
    if scheduler is None or not scheduler.running:
        return
    scheduler.shutdown(wait=False)
    scheduler = None
    log_event("info", "Scheduler stopped")


def is_scheduler_running() -> bool:
    return scheduler is not None and scheduler.running


async def run_manual_scan():
    log_event("info", "Manual scan triggered")
    await scan_and_trade_job()


async def run_manual_settlement():
    log_event("info", "Manual settlement triggered")
    await settlement_job()


async def run_manual_whale_scan():
    log_event("info", "Manual whale scan triggered")
    await whale_scan_job()
