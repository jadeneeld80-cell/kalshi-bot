"""Offline KXBTC15M XGBoost research and paper-trading model.

No order placement. Uses recorded snapshots, closed Coinbase candles, and
settled Kalshi market outcomes. Preserve FEATURE_COLUMNS order for inference.
"""
import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from backend.config import settings
from backend.models.database import SessionLocal, KxBtcSnapshot
from backend.btcmarket import candles as candles_mod
from backend.scanner.kalshi_api import KalshiScannerAPI

logger = logging.getLogger('trading_bot')
ARTIFACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'btcmarket', 'artifacts'))
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'kxbtc15m_xgb.json')
REPORT_PATH = os.path.join(ARTIFACT_DIR, 'training_report.json')
CALIBRATION_PATH = os.path.join(ARTIFACT_DIR, 'calibration.json')
LIVE_CALIBRATION_PATH = os.path.join(ARTIFACT_DIR, 'live_calibration.json')
VOL_WINDOW_MINUTES = 180
MIN_VOL_SAMPLES = 20
SETTLEMENT_BIAS_WINDOW = 20
PATH_EFFICIENCY_WINDOW = 15
LIVE_RECALIBRATION_MIN_SAMPLES = 30
SHRINKAGE_PRIOR_STRENGTH = 50
FEATURE_COLUMNS = [
    'seconds_remaining', 'elapsed_fraction', 'distance_log', 'yes_mid',
    'spread', 'gbm_prob', 'realized_vol_per_second', 'm1_ret_1m',
    'm1_ret_5m', 'm1_vol_sum_5m', 'm1_vol_sum_15m', 'h1_log_return',
    'h1_range_pct', 'quote_age_seconds', 'btc_move_since_quote',
    'path_efficiency_15m', 'hour_sin', 'hour_cos', 'rsi_14',
    'macd_histogram_pct', 'bollinger_pct_b', 'vwap_deviation_60m',
    'settlement_bias',
]
_api = KalshiScannerAPI()


async def _fetch_settlement_labels(min_close_ts: int, max_close_ts: int) -> Dict[str, str]:
    labels = {}
    cursor = None
    for _ in range(50):
        data = await _api.get_markets(
            series_ticker=settings.KXBTC15M_SERIES_TICKER,
            status='settled', min_close_ts=min_close_ts,
            max_close_ts=max_close_ts, limit=200, cursor=cursor,
        )
        for market in data.get('markets', []):
            if market.get('result') in ('yes', 'no') and market.get('ticker'):
                labels[market['ticker']] = market['result']
        cursor = data.get('cursor')
        if not cursor or not data.get('markets'):
            break
    return labels


async def get_live_settlement_bias(window: int = SETTLEMENT_BIAS_WINDOW) -> Optional[float]:
    now = datetime.utcnow()
    markets, cursor = [], None
    for _ in range(10):
        data = await _api.get_markets(
            series_ticker=settings.KXBTC15M_SERIES_TICKER,
            status='settled', min_close_ts=int((now - timedelta(hours=12)).timestamp()),
            max_close_ts=int(now.timestamp()), limit=200, cursor=cursor,
        )
        markets.extend(data.get('markets', []))
        cursor = data.get('cursor')
        if not cursor or not data.get('markets'):
            break
    settled = sorted(
        [(m.get('close_time', ''), m['result']) for m in markets
         if m.get('result') in ('yes', 'no')], reverse=True,
    )[:window]
    if len(settled) < 5:
        return None
    return sum(result == 'yes' for _, result in settled) / len(settled)


async def _ensure_historical_1m_candles(start: datetime, end: datetime) -> int:
    db = SessionLocal()
    try:
        fetched = await candles_mod.fetch_coinbase_candles(start, end, 60)
        return candles_mod._upsert_candles(db, fetched, 60)
    finally:
        db.close()


def rolling_realized_vol_per_second(closes: List[float]) -> Optional[float]:
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    window = returns[-VOL_WINDOW_MINUTES:]
    if len(window) < MIN_VOL_SAMPLES:
        return None
    return float(np.std(window, ddof=1) / math.sqrt(60))


def _quote_staleness_for_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('timestamp')
    ages, moves, change_idx = [], [], 0
    timestamps = g['timestamp'].tolist()
    bids, asks, prices = g['yes_bid'].tolist(), g['yes_ask'].tolist(), g['btc_price'].tolist()
    for i in range(len(g)):
        if i and (bids[i] != bids[i - 1] or asks[i] != asks[i - 1]):
            change_idx = i
        ages.append((timestamps[i] - timestamps[change_idx]).total_seconds())
        start_price = prices[change_idx]
        moves.append(math.log(prices[i] / start_price) if start_price > 0 and prices[i] > 0 else 0.0)
    return pd.DataFrame({'quote_age_seconds': ages, 'btc_move_since_quote': moves}, index=g.index)


def add_quote_staleness_features(snap_df: pd.DataFrame) -> pd.DataFrame:
    if snap_df.empty:
        return snap_df.copy()
    parts = [_quote_staleness_for_group(g) for _, g in snap_df.groupby('ticker', sort=False)]
    return snap_df.join(pd.concat(parts))


def quote_staleness_now(rows: List[dict]) -> 'tuple[float, float]':
    if not rows:
        return 0.0, 0.0
    last = rows[-1]
    i = len(rows) - 2
    change_idx = len(rows) - 1
    while i >= 0 and rows[i]['yes_bid'] == last['yes_bid'] and rows[i]['yes_ask'] == last['yes_ask']:
        change_idx = i
        i -= 1
    first = rows[change_idx]
    age = max((last['timestamp'] - first['timestamp']).total_seconds(), 0.0)
    a, b = first['btc_price'], last['btc_price']
    return age, math.log(b / a) if a > 0 and b > 0 else 0.0


def technical_indicators(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    delta = close.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    rsi = rsi.where(avg_gain.notna())
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    histogram = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / close
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    bollinger = (close - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    vwap = (close * volume).rolling(60, min_periods=20).sum() / volume.rolling(60, min_periods=20).sum().replace(0, np.nan)
    return pd.DataFrame({
        'rsi_14': rsi, 'macd_histogram_pct': histogram,
        'bollinger_pct_b': bollinger,
        'vwap_deviation_60m': (close - vwap) / vwap,
    }, index=close.index)


def _candle_features_1m(lookback: timedelta) -> pd.DataFrame:
    df = pd.DataFrame(candles_mod.get_cached_candles(60, lookback))
    if df.empty:
        return pd.DataFrame(columns=['candle_1m_available_at', 'realized_vol_per_second',
            'm1_ret_1m', 'm1_ret_5m', 'm1_vol_sum_5m', 'm1_vol_sum_15m',
            'path_efficiency_15m', 'rsi_14', 'macd_histogram_pct',
            'bollinger_pct_b', 'vwap_deviation_60m'])
    df['open_time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('open_time').reset_index(drop=True)
    log_return = np.log(df['close'] / df['close'].shift(1))
    df['m1_ret_1m'] = log_return
    df['m1_ret_5m'] = np.log(df['close'] / df['close'].shift(5))
    df['m1_vol_sum_5m'] = df['volume'].rolling(5, min_periods=1).sum()
    df['m1_vol_sum_15m'] = df['volume'].rolling(15, min_periods=1).sum()
    df['realized_vol_per_second'] = log_return.rolling(VOL_WINDOW_MINUTES, min_periods=MIN_VOL_SAMPLES).std() / math.sqrt(60)
    displacement = (np.log(df['close']) - np.log(df['close'].shift(PATH_EFFICIENCY_WINDOW))).abs()
    movement = log_return.abs().rolling(PATH_EFFICIENCY_WINDOW, min_periods=PATH_EFFICIENCY_WINDOW).sum()
    df['path_efficiency_15m'] = (displacement / movement.replace(0, np.nan)).clip(0, 1)
    df = pd.concat([df, technical_indicators(df['close'], df['volume'])], axis=1)
    # A completed candle cannot be used before its END time.
    df['candle_1m_available_at'] = df['open_time'] + pd.Timedelta(minutes=1)
    return df[['candle_1m_available_at', 'realized_vol_per_second', 'm1_ret_1m',
        'm1_ret_5m', 'm1_vol_sum_5m', 'm1_vol_sum_15m', 'path_efficiency_15m',
        'rsi_14', 'macd_histogram_pct', 'bollinger_pct_b', 'vwap_deviation_60m']]


def _candle_features_1h(lookback: timedelta) -> pd.DataFrame:
    df = pd.DataFrame(candles_mod.get_cached_candles(3600, lookback))
    if df.empty:
        return pd.DataFrame(columns=['candle_1h_available_at', 'h1_log_return', 'h1_range_pct'])
    df['open_time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('open_time').reset_index(drop=True)
    df['h1_log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['h1_range_pct'] = (df['high'] - df['low']) / df['close']
    df['candle_1h_available_at'] = df['open_time'] + pd.Timedelta(hours=1)
    return df[['candle_1h_available_at', 'h1_log_return', 'h1_range_pct']]


def _load_snapshots(since: datetime) -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = (db.query(KxBtcSnapshot)
            .filter(KxBtcSnapshot.timestamp >= since)
            .order_by(KxBtcSnapshot.timestamp.asc()).all())
        return pd.DataFrame([{
            'ticker': r.ticker, 'timestamp': r.timestamp,
            'floor_strike': r.floor_strike, 'open_time': r.open_time,
            'close_time': r.close_time, 'yes_bid': r.yes_bid,
            'yes_ask': r.yes_ask, 'btc_price': r.btc_price,
        } for r in rows])
    finally:
        db.close()


async def build_feature_frame(lookback_days: int = 60) -> pd.DataFrame:
    since = datetime.utcnow() - timedelta(days=lookback_days)
    snap = _load_snapshots(since)
    if snap.empty:
        return snap
    earliest = snap['timestamp'].min().to_pydatetime()
    await _ensure_historical_1m_candles(earliest - timedelta(hours=4), datetime.utcnow())
    df1m = _candle_features_1m(timedelta(days=lookback_days + 1))
    df1h = _candle_features_1h(timedelta(days=lookback_days + 1))
    if df1m.empty or df1h.empty:
        return pd.DataFrame()
    snap = snap.dropna(subset=['floor_strike', 'btc_price', 'yes_bid', 'yes_ask', 'close_time', 'open_time'])
    snap = snap[(snap['yes_bid'] > 0) & (snap['yes_ask'] > 0) & (snap['btc_price'] > 0) & (snap['floor_strike'] > 0)].copy()
    if snap.empty:
        return snap
    snap = add_quote_staleness_features(snap)
    merged = pd.merge_asof(snap.sort_values('timestamp'),
        df1m.sort_values('candle_1m_available_at'),
        left_on='timestamp', right_on='candle_1m_available_at',
        direction='backward', tolerance=pd.Timedelta(minutes=10))
    merged = pd.merge_asof(merged.sort_values('timestamp'),
        df1h.sort_values('candle_1h_available_at'),
        left_on='timestamp', right_on='candle_1h_available_at',
        direction='backward', tolerance=pd.Timedelta(hours=2))
    labels = await _fetch_settlement_labels(
        int((since - timedelta(hours=1)).timestamp()), int(datetime.utcnow().timestamp()))
    merged['label'] = merged['ticker'].map(labels)
    merged = merged.dropna(subset=['label', 'realized_vol_per_second']).copy()
    if merged.empty:
        return merged
    ticker_order = merged.groupby('ticker')['close_time'].first().sort_values()
    ticker_labels = merged.groupby('ticker')['label'].first().reindex(ticker_order.index)
    trailing = (ticker_labels == 'yes').astype(float).rolling(
        SETTLEMENT_BIAS_WINDOW, min_periods=5).mean().shift(1)
    merged['settlement_bias'] = merged['ticker'].map(trailing.to_dict())
    seconds = (merged['close_time'] - merged['timestamp']).dt.total_seconds().clip(lower=1)
    total = (merged['close_time'] - merged['open_time']).dt.total_seconds().clip(lower=1)
    sigma = merged['realized_vol_per_second'] * np.sqrt(seconds)

    def rolling_twap(group):
        group = group.sort_values('timestamp')
        # Duplicate timestamps are legal; rolling needs a monotonic index.
        rolled = group.set_index('timestamp')['btc_price'].rolling(
            f'{candles_mod.TWAP_WINDOW_SECONDS}s', min_periods=1).mean()
        return pd.Series(rolled.to_numpy(), index=group.index)

    twap = merged.groupby('ticker', group_keys=False)[['timestamp', 'btc_price']].apply(rolling_twap)
    distance = np.log(twap / merged['floor_strike'])
    z = (distance - 0.5 * sigma ** 2) / sigma.replace(0, np.nan)
    gbm = pd.Series(norm.cdf(z), index=merged.index).clip(0.01, 0.99)
    hour = merged['timestamp'].dt.hour + merged['timestamp'].dt.minute / 60
    out = pd.DataFrame({
        'ticker': merged['ticker'], 'timestamp': merged['timestamp'],
        'open_time': merged['open_time'], 'close_time': merged['close_time'],
        'seconds_remaining': seconds, 'elapsed_fraction': 1 - seconds / total,
        'distance_log': distance, 'yes_mid': (merged['yes_bid'] + merged['yes_ask']) / 2,
        'spread': merged['yes_ask'] - merged['yes_bid'], 'gbm_prob': gbm,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'label': (merged['label'] == 'yes').astype(int),
    })
    for col in FEATURE_COLUMNS:
        if col not in out:
            out[col] = merged[col]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)


def _fit_platt_scaling(prob: np.ndarray, label: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(label)) < 2:
        raise ValueError('Calibration requires both YES and NO outcomes')
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression().fit(logits, label)
    return {'a': float(lr.coef_[0][0]), 'b': float(lr.intercept_[0])}


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_calibration() -> Optional[dict]:
    return _load_json(CALIBRATION_PATH)


def apply_calibration(raw_prob: float, calib: Optional[dict] = None) -> float:
    if calib is None:
        calib = load_calibration()
    if calib is None:
        return raw_prob
    p = min(max(float(raw_prob), 1e-6), 1 - 1e-6)
    logit = calib['a'] * math.log(p / (1 - p)) + calib['b']
    return float(1 / (1 + math.exp(-max(min(logit, 700), -700))))


def load_live_calibration() -> Optional[dict]:
    return _load_json(LIVE_CALIBRATION_PATH)


def apply_live_recalibration(calibrated_prob: float, calib: Optional[dict] = None) -> float:
    if calib is None:
        calib = load_live_calibration()
    return apply_calibration(calibrated_prob, calib) if calib is not None else calibrated_prob


def fit_live_recalibration(since: Optional[datetime] = None,
                           min_samples: int = LIVE_RECALIBRATION_MIN_SAMPLES) -> dict:
    from backend.models.database import BtcPaperTrade
    since = since or datetime(2026, 9, 11, 9, 39, 28)
    db = SessionLocal()
    try:
        trades = (db.query(BtcPaperTrade)
            .filter(BtcPaperTrade.entry_time >= since, BtcPaperTrade.settled == True).all())
        n = len(trades)
        if n < min_samples:
            return {'status': 'insufficient_data', 'n': n, 'min_samples': min_samples}
        prob = np.asarray([t.model_probability for t in trades], dtype=float)
        label = np.asarray([t.settlement_value for t in trades], dtype=int)
        if len(np.unique(label)) < 2:
            return {'status': 'insufficient_class_diversity', 'n': n}
        raw = _fit_platt_scaling(prob, label)
        weight = n / (n + SHRINKAGE_PRIOR_STRENGTH)
        calib = {'a': weight * raw['a'] + (1 - weight), 'b': weight * raw['b']}
        recalibrated = np.array([apply_calibration(p, calib) for p in prob])
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(LIVE_CALIBRATION_PATH, 'w', encoding='utf-8') as f:
            json.dump(calib, f, indent=2)
        return {'status': 'ok', 'n': n, 'since': since.isoformat(),
            'calibration': calib, 'unshrunk_calibration': raw,
            'shrinkage_weight': round(weight, 4),
            'pre_recalibration_brier': round(_brier(prob, label), 5),
            'post_recalibration_brier_on_same_data_DO_NOT_TRUST_FOR_COMPARISON':
                round(_brier(recalibrated, label), 5)}
    finally:
        db.close()


def _time_split_by_ticker(df: pd.DataFrame, train_frac: float = 0.8):
    ordered = df.groupby('ticker')['close_time'].first().sort_values().index.tolist()
    cutoff = int(len(ordered) * train_frac)
    train = set(ordered[:cutoff])
    return df[df['ticker'].isin(train)], df[~df['ticker'].isin(train)]


def _brier(prob: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((prob - label) ** 2))


def _log_loss(prob: np.ndarray, label: np.ndarray) -> float:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return float(-np.mean(label * np.log(p) + (1 - label) * np.log(1 - p)))


def _metrics_of(prob: np.ndarray, label: np.ndarray) -> dict:
    return {'brier': round(_brier(prob, label), 5),
        'log_loss': round(_log_loss(prob, label), 5),
        'accuracy': round(float(np.mean((prob >= 0.5) == label)), 4)}


def _calibration_buckets(prob: np.ndarray, label: np.ndarray, n_bins: int = 10) -> List[dict]:
    bins = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    result = []
    for i in range(n_bins):
        mask = bins == i
        if mask.any():
            result.append({'bucket': f'{i * 100 // n_bins}-{(i + 1) * 100 // n_bins}%',
                'n': int(mask.sum()), 'predicted_avg': round(float(prob[mask].mean()), 4),
                'actual_rate': round(float(label[mask].mean()), 4)})
    return result


def _mismatch_buckets(model_prob: np.ndarray, market_prob: np.ndarray,
                      label: np.ndarray) -> List[dict]:
    result = []
    for lo, hi in zip([0, .02, .05, .10, .20], [.02, .05, .10, .20, 1.01]):
        mask = (np.abs(model_prob - market_prob) >= lo) & (np.abs(model_prob - market_prob) < hi)
        if mask.any():
            result.append({'abs_edge_range': f'{lo:.0%}-{hi:.0%}' if hi <= 1 else f'>= {lo:.0%}',
                'n': int(mask.sum()),
                'model_accuracy': round(float(np.mean((model_prob[mask] >= .5) == label[mask])), 4),
                'market_accuracy': round(float(np.mean((market_prob[mask] >= .5) == label[mask])), 4)})
    return result


def _fit_xgb(X_train, y_train, X_val, y_val):
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=150, max_depth=2, learning_rate=.05,
        subsample=.7, colsample_bytree=.7, reg_lambda=5.0,
        objective='binary:logistic', eval_metric='logloss', early_stopping_rounds=20)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


async def train_btc_model(lookback_days: int = 60, n_folds: int = 5,
                          min_train_windows: int = 150) -> dict:
    df = await build_feature_frame(lookback_days)
    cutoff = datetime.fromisoformat(settings.BTC_HOLDOUT_START)
    if not df.empty:
        df = df[df['close_time'] < cutoff]
    order = df.groupby('ticker')['close_time'].first().sort_values().index.tolist() if not df.empty else []
    n = len(order)
    if n < min_train_windows * 2:
        return {'status': 'insufficient_data', 'rows': len(df), 'windows': n,
            'note': f'Need at least {min_train_windows * 2} settled pre-holdout windows.'}
    cuts = np.linspace(min_train_windows, n, n_folds + 1).astype(int)
    reports, probabilities, markets, baselines, labels = [], [], [], [], []
    for i in range(n_folds):
        train_set = set(order[:cuts[i]])
        test_set = set(order[cuts[i]:cuts[i + 1]])
        full_train = df[df['ticker'].isin(train_set)]
        test = df[df['ticker'].isin(test_set)]
        train, val = _time_split_by_ticker(full_train, .85)
        if train.empty or val.empty or test.empty:
            continue
        if train['label'].nunique() < 2 or val['label'].nunique() < 2:
            continue
        model = _fit_xgb(train[FEATURE_COLUMNS].values, train['label'].values,
            val[FEATURE_COLUMNS].values, val['label'].values)
        prob = model.predict_proba(test[FEATURE_COLUMNS].values)[:, 1]
        label = test['label'].to_numpy()
        market = test['yes_mid'].to_numpy()
        gbm = test['gbm_prob'].to_numpy()
        reports.append({'fold': i + 1, 'train_windows': full_train['ticker'].nunique(),
            'test_windows': test['ticker'].nunique(), 'test_rows': len(test),
            'xgb_brier': round(_brier(prob, label), 5),
            'gbm_brier': round(_brier(gbm, label), 5),
            'market_brier': round(_brier(market, label), 5)})
        probabilities.append(prob)
        markets.append(market)
        baselines.append(gbm)
        labels.append(label)
    if not reports:
        return {'status': 'insufficient_data', 'rows': len(df), 'windows': n,
            'note': 'No usable walk-forward folds.'}
    prob, market, gbm, label = map(np.concatenate, (probabilities, markets, baselines, labels))
    metrics = {'xgb': _metrics_of(prob, label),
        'gbm_baseline': _metrics_of(gbm, label), 'market': _metrics_of(market, label)}
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    calibration = _fit_platt_scaling(prob, label)
    calibrated = np.array([apply_calibration(p, calibration) for p in prob])
    metrics['xgb_calibrated_on_same_data_DO_NOT_TRUST_FOR_COMPARISON'] = _metrics_of(calibrated, label)
    prod_train, prod_val = _time_split_by_ticker(df, .9)
    if prod_train['label'].nunique() < 2 or prod_val['label'].nunique() < 2:
        return {'status': 'insufficient_class_diversity', 'windows': n}
    final = _fit_xgb(prod_train[FEATURE_COLUMNS].values, prod_train['label'].values,
        prod_val[FEATURE_COLUMNS].values, prod_val['label'].values)
    importance = sorted(zip(FEATURE_COLUMNS, final.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True)
    report = {'status': 'ok', 'method': 'walk_forward', 'trained_at': datetime.utcnow().isoformat(),
        'lookback_days': lookback_days, 'total_windows': n,
        'production_windows': int(prod_train['ticker'].nunique()),
        'fold_reports': reports, 'metrics': metrics,
        'feature_importance': [{'feature': f, 'importance': round(v, 4)} for f, v in importance],
        'calibration_xgb': _calibration_buckets(prob, label),
        'calibration_market': _calibration_buckets(market, label),
        'mismatch_vs_market': _mismatch_buckets(prob, market, label),
        'model_path': MODEL_PATH}
    final.save_model(MODEL_PATH)
    with open(CALIBRATION_PATH, 'w', encoding='utf-8') as f:
        json.dump(calibration, f, indent=2)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return report


async def evaluate_holdout(lookback_days: int = 90) -> dict:
    import xgboost as xgb
    if not os.path.exists(MODEL_PATH):
        return {'status': 'no_model'}
    cutoff = datetime.fromisoformat(settings.BTC_HOLDOUT_START)
    df = await build_feature_frame(lookback_days)
    if df.empty:
        return {'status': 'no_data'}
    df = df[df['close_time'] >= cutoff]
    if df.empty:
        return {'status': 'no_holdout_data_yet', 'holdout_start': cutoff.isoformat()}
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    prob = model.predict_proba(df[FEATURE_COLUMNS].values)[:, 1]
    label = df['label'].to_numpy()
    market = df['yes_mid'].to_numpy()
    gbm = df['gbm_prob'].to_numpy()
    return {'status': 'ok', 'holdout_start': cutoff.isoformat(),
        'evaluated_at': datetime.utcnow().isoformat(),
        'windows': int(df['ticker'].nunique()), 'rows': len(df),
        'metrics': {'xgb': _metrics_of(prob, label),
            'gbm_baseline': _metrics_of(gbm, label), 'market': _metrics_of(market, label)},
        'calibration_xgb': _calibration_buckets(prob, label),
        'mismatch_vs_market': _mismatch_buckets(prob, market, label)}
