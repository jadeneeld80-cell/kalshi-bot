import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { fetchDashboard, runScan, startBot, stopBot } from './api'
import { EquityChart } from './components/EquityChart'
import { CalibrationPanel } from './components/CalibrationPanel'
import { Terminal } from './components/Terminal'
import { TradesTable } from './components/TradesTable'
import { BacktestPanel } from './components/BacktestPanel'
import { WhaleScannerPanel } from './components/WhaleScannerPanel'
import { BtcMarketView } from './components/BtcMarketView'
import type { WeatherSignal, WeatherForecast } from './types'

type View = 'dashboard' | 'backtest' | 'whales' | 'btc'

// ── helpers ──────────────────────────────────────────────────────────────────

function LiveClock() {
  const [time, setTime] = useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return (
    <span className="text-xs tabular-nums text-neutral-400 font-mono">
      {time.toUTCString().slice(17, 25)} UTC
    </span>
  )
}

function RefreshBar({ interval }: { interval: number }) {
  const [progress, setProgress] = useState(100)
  useEffect(() => {
    setProgress(100)
    const step = 100 / (interval / 1000)
    const id = setInterval(() => setProgress(p => Math.max(0, p - step)), 1000)
    return () => clearInterval(id)
  }, [interval])
  return (
    <div className="h-0.5 w-16 bg-neutral-800 rounded overflow-hidden">
      <div className="h-full bg-green-600 transition-all" style={{ width: `${progress}%` }} />
    </div>
  )
}

function EdgeBar({ edge, actionable }: { edge: number; actionable: boolean }) {
  const pct = Math.min(100, Math.abs(edge) * 500)
  return (
    <div className="flex items-center gap-1.5">
      <div className="h-1 w-16 bg-neutral-800 rounded overflow-hidden">
        <div
          className={`h-full rounded ${actionable ? 'bg-green-500' : 'bg-neutral-600'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-[10px] tabular-nums font-mono ${actionable ? 'text-green-400' : 'text-neutral-500'}`}>
        {edge >= 0 ? '+' : ''}{(edge * 100).toFixed(1)}%
      </span>
    </div>
  )
}

// ── signals table ─────────────────────────────────────────────────────────────

function SignalsPanel({ signals }: { signals: WeatherSignal[] }) {
  if (signals.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-[11px] text-neutral-600 uppercase tracking-wider">
        No signals — waiting for next scan
      </div>
    )
  }
  return (
    <div className="overflow-auto h-full">
      <table className="w-full text-[11px] font-mono">
        <thead className="sticky top-0 bg-neutral-950 border-b border-neutral-800">
          <tr className="text-neutral-500 uppercase tracking-wider">
            <th className="text-left px-3 py-1.5">City</th>
            <th className="text-left px-2 py-1.5">Date</th>
            <th className="text-left px-2 py-1.5">Question</th>
            <th className="text-right px-2 py-1.5">Model</th>
            <th className="text-right px-2 py-1.5">Market</th>
            <th className="text-left px-2 py-1.5">Edge</th>
            <th className="text-right px-2 py-1.5">Size</th>
            <th className="text-center px-2 py-1.5">Trade</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => (
            <tr
              key={s.market_id}
              className={`border-b border-neutral-900 ${
                s.actionable ? 'bg-green-950/20 hover:bg-green-950/30' : 'hover:bg-neutral-900/30'
              } transition-colors`}
            >
              <td className="px-3 py-1.5 text-neutral-200 font-semibold whitespace-nowrap">
                {s.city_name}
              </td>
              <td className="px-2 py-1.5 text-neutral-400">
                {s.target_date.slice(5)}
              </td>
              <td className="px-2 py-1.5 text-neutral-300 whitespace-nowrap">
                High {s.direction} {s.threshold_f.toFixed(0)}°F
              </td>
              <td className="px-2 py-1.5 text-right text-cyan-400 tabular-nums">
                {(s.model_probability * 100).toFixed(0)}%
              </td>
              <td className="px-2 py-1.5 text-right text-neutral-400 tabular-nums">
                {(s.market_probability * 100).toFixed(0)}%
              </td>
              <td className="px-2 py-1.5">
                <EdgeBar edge={s.edge} actionable={s.actionable} />
              </td>
              <td className="px-2 py-1.5 text-right tabular-nums text-neutral-300">
                ${s.suggested_size.toFixed(0)}
              </td>
              <td className="px-2 py-1.5 text-center">
                {s.actionable ? (
                  <span className="px-1.5 py-0.5 text-[9px] font-bold uppercase bg-green-500/15 text-green-400 border border-green-500/30 rounded">
                    {s.trade_direction.toUpperCase()}
                  </span>
                ) : (
                  <span className="text-[9px] text-neutral-700">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── forecast cards ────────────────────────────────────────────────────────────

function ForecastCard({ f }: { f: WeatherForecast }) {
  const agreement = Math.round(f.ensemble_agreement * 100)
  const agreementColor = agreement >= 80 ? 'text-green-400' : agreement >= 65 ? 'text-amber-400' : 'text-neutral-500'
  return (
    <div className="border border-neutral-800 bg-neutral-900/40 p-3 rounded-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] font-semibold text-neutral-200">{f.city_name}</span>
        <span className="text-[9px] text-neutral-600">{f.target_date.slice(5)}</span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-[10px]">
        <div>
          <div className="text-neutral-500 uppercase tracking-wider mb-0.5">High</div>
          <div className="text-white tabular-nums font-mono font-semibold">
            {f.mean_high.toFixed(1)}°F
          </div>
          <div className="text-neutral-600 tabular-nums">±{f.std_high.toFixed(1)}°</div>
        </div>
        <div>
          <div className="text-neutral-500 uppercase tracking-wider mb-0.5">Agreement</div>
          <div className={`tabular-nums font-mono font-semibold ${agreementColor}`}>
            {agreement}%
          </div>
          <div className="text-neutral-600">{f.num_members} members</div>
        </div>
      </div>
    </div>
  )
}

// ── main app ──────────────────────────────────────────────────────────────────

function App() {
  const [view, setView] = useState<View>('dashboard')
  const queryClient = useQueryClient()

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['dashboard'],
    queryFn: fetchDashboard,
    refetchInterval: 15000,
  })

  const scanMutation = useMutation({
    mutationFn: runScan,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['dashboard'] }),
  })

  const startMutation = useMutation({
    mutationFn: startBot,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['dashboard'] }),
  })

  const stopMutation = useMutation({
    mutationFn: stopBot,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['dashboard'] }),
  })

  if (isLoading) {
    return (
      <div className="h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-8 h-8 mx-auto mb-4">
            <div className="absolute inset-0 border border-neutral-800 rounded-full" />
            <div className="absolute inset-0 border border-transparent border-t-green-500 rounded-full animate-spin" />
          </div>
          <div className="text-[10px] text-neutral-600 uppercase tracking-widest font-mono">Connecting</div>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="h-screen bg-black flex items-center justify-center">
        <div className="text-center space-y-3">
          <div className="text-red-500 text-xs uppercase tracking-wider">Backend Offline</div>
          <button
            onClick={() => refetch()}
            className="px-3 py-1.5 bg-neutral-900 border border-neutral-700 text-neutral-300 text-xs uppercase tracking-wider"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  const stats = data.stats
  const signals = data.weather_signals ?? []
  const forecasts = data.weather_forecasts ?? []
  const actionableCount = signals.filter(s => s.actionable).length
  const isLive = !stats.simulation_mode
  const kalshiOk = data.kalshi_status?.connected

  return (
    <div className="h-screen bg-black text-neutral-200 flex flex-col overflow-hidden font-mono">

      {/* ── HEADER ── */}
      <motion.header
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        className="shrink-0 border-b border-neutral-800 px-4 py-2 flex items-center gap-4"
      >
        <div className="flex items-center gap-2 shrink-0">
          <h1 className="text-xs font-bold uppercase tracking-widest text-neutral-100">
            Kalshi Weather
          </h1>
          {/* Run state */}
          <span className={`px-1.5 py-0.5 text-[9px] font-bold uppercase border ${
            stats.is_running
              ? 'bg-green-500/10 text-green-500 border-green-500/20'
              : 'bg-neutral-800 text-neutral-500 border-neutral-700'
          }`}>
            {stats.is_running ? 'Running' : 'Idle'}
          </span>
          {/* Live vs Sim */}
          <span className={`px-1.5 py-0.5 text-[9px] font-bold uppercase border ${
            isLive
              ? 'bg-red-500/10 text-red-400 border-red-500/20'
              : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
          }`}>
            {isLive ? 'Live' : 'Sim'}
          </span>
          {/* Kalshi connection */}
          <span className={`px-1.5 py-0.5 text-[9px] font-bold uppercase border ${
            kalshiOk
              ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
              : 'bg-neutral-800 text-neutral-600 border-neutral-700'
          }`}>
            {kalshiOk ? 'Kalshi ✓' : 'Kalshi —'}
          </span>
        </div>

        {/* Stats strip */}
        <div className="flex items-center gap-5 text-[11px]">
          <div>
            <span className="text-neutral-600 mr-1">Bankroll</span>
            <span className="text-neutral-100 tabular-nums">${stats.bankroll.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
          </div>
          <div>
            <span className="text-neutral-600 mr-1">P&L</span>
            <span className={`tabular-nums ${stats.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {stats.total_pnl >= 0 ? '+' : ''}${stats.total_pnl.toFixed(0)}
            </span>
          </div>
          <div>
            <span className="text-neutral-600 mr-1">Win rate</span>
            <span className="text-neutral-200 tabular-nums">{(stats.win_rate * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span className="text-neutral-600 mr-1">Trades</span>
            <span className="text-neutral-200 tabular-nums">{stats.total_trades}</span>
          </div>
          <div>
            <span className="text-amber-400 tabular-nums">{actionableCount} actionable</span>
          </div>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-3 shrink-0">
          {/* view tabs */}
          <div className="flex items-center border border-neutral-800 rounded-sm overflow-hidden">
            <button
              onClick={() => setView('dashboard')}
              className={`px-2.5 py-1 text-[9px] uppercase tracking-wider transition-colors ${
                view === 'dashboard'
                  ? 'bg-neutral-800 text-neutral-200'
                  : 'text-neutral-600 hover:text-neutral-400'
              }`}
            >
              Dashboard
            </button>
            <button
              onClick={() => setView('backtest')}
              className={`px-2.5 py-1 text-[9px] uppercase tracking-wider transition-colors border-l border-neutral-800 ${
                view === 'backtest'
                  ? 'bg-neutral-800 text-cyan-400'
                  : 'text-neutral-600 hover:text-neutral-400'
              }`}
            >
              Backtest
            </button>
            <button
              onClick={() => setView('whales')}
              className={`px-2.5 py-1 text-[9px] uppercase tracking-wider transition-colors border-l border-neutral-800 ${
                view === 'whales'
                  ? 'bg-neutral-800 text-amber-400'
                  : 'text-neutral-600 hover:text-neutral-400'
              }`}
            >
              Whales
            </button>
            <button
              onClick={() => setView('btc')}
              className={`px-2.5 py-1 text-[9px] uppercase tracking-wider transition-colors border-l border-neutral-800 ${
                view === 'btc'
                  ? 'bg-neutral-800 text-orange-400'
                  : 'text-neutral-600 hover:text-neutral-400'
              }`}
            >
              BTC 15M
            </button>
          </div>

          {view === 'dashboard' && (
            <>
              <button
                onClick={() => stats.is_running ? stopMutation.mutate() : startMutation.mutate()}
                className={`px-2.5 py-1 border text-[10px] uppercase tracking-wider transition-colors ${
                  stats.is_running
                    ? 'border-red-800 text-red-400 hover:border-red-600'
                    : 'border-green-800 text-green-400 hover:border-green-600'
                }`}
              >
                {stats.is_running ? 'Stop' : 'Start'}
              </button>
              <button
                onClick={() => scanMutation.mutate()}
                disabled={scanMutation.isPending}
                className="px-2.5 py-1 bg-neutral-900 border border-neutral-700 hover:border-neutral-500 text-neutral-300 text-[10px] uppercase tracking-wider transition-colors disabled:opacity-40"
              >
                {scanMutation.isPending ? 'Scanning…' : 'Scan'}
              </button>
              <RefreshBar interval={15000} />
            </>
          )}
          <LiveClock />
        </div>
      </motion.header>

      {/* ── BACKTEST VIEW ── */}
      {view === 'backtest' && (
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          <BacktestPanel />
        </div>
      )}

      {/* ── WHALE SCANNER VIEW ── */}
      {view === 'whales' && (
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          <WhaleScannerPanel />
        </div>
      )}

      {/* ── BTC 15M VIEW ── */}
      {view === 'btc' && (
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          <BtcMarketView />
        </div>
      )}

      {/* ── MAIN GRID ── */}
      {/* Left 280px | Center flex-1 | Right 300px */}
      {view === 'dashboard' && <div className="flex-1 min-h-0 grid grid-cols-[280px_1fr_300px]">

        {/* ── LEFT: equity + calibration + terminal ── */}
        <div className="flex flex-col border-r border-neutral-800 min-h-0">
          {/* Equity */}
          <div className="border-b border-neutral-800" style={{ height: '30%', minHeight: 120 }}>
            <div className="px-3 py-1 border-b border-neutral-800 flex items-center justify-between shrink-0">
              <span className="text-[10px] text-neutral-500 uppercase tracking-wider">Equity</span>
              <span className={`text-[10px] tabular-nums ${stats.total_pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {stats.total_pnl >= 0 ? '+' : ''}${stats.total_pnl.toFixed(0)}
              </span>
            </div>
            <div className="h-[calc(100%-28px)] p-1">
              <EquityChart data={data.equity_curve} initialBankroll={stats.bankroll - stats.total_pnl} />
            </div>
          </div>

          {/* Calibration */}
          {data.calibration && data.calibration.total_with_outcome > 0 && (
            <div className="shrink-0 border-b border-neutral-800 px-3 py-2">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] text-neutral-500 uppercase tracking-wider">Calibration</span>
                <span className="text-[9px] text-neutral-600">{data.calibration.total_with_outcome} settled</span>
              </div>
              <div className="grid grid-cols-2 gap-x-4 text-[10px] mb-1.5">
                <div>
                  <span className="text-neutral-600">Accuracy </span>
                  <span className="text-neutral-200 tabular-nums">{(data.calibration.accuracy * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-neutral-600">Brier </span>
                  <span className="text-neutral-200 tabular-nums">{data.calibration.brier_score.toFixed(3)}</span>
                </div>
              </div>
              <CalibrationPanel calibration={data.calibration} />
            </div>
          )}

          {/* Terminal fills rest */}
          <div className="flex-1 min-h-0">
            <Terminal
              isRunning={stats.is_running}
              lastRun={stats.last_run}
              stats={{ total_trades: stats.total_trades, total_pnl: stats.total_pnl }}
              onStart={() => startMutation.mutate()}
              onStop={() => stopMutation.mutate()}
              onScan={() => scanMutation.mutate()}
            />
          </div>
        </div>

        {/* ── CENTER: signals table ── */}
        <div className="flex flex-col min-h-0 border-r border-neutral-800">
          <div className="shrink-0 px-3 py-1.5 border-b border-neutral-800 flex items-center justify-between">
            <span className="text-[10px] text-neutral-500 uppercase tracking-wider">
              Kalshi Signals
            </span>
            <div className="flex items-center gap-3 text-[10px]">
              <span className="text-neutral-600">{signals.length} markets</span>
              <span className="text-green-400">{actionableCount} actionable</span>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <SignalsPanel signals={signals} />
          </div>
        </div>

        {/* ── RIGHT: forecasts + trades ── */}
        <div className="flex flex-col min-h-0 overflow-hidden">
          {/* Forecast cards */}
          <div className="shrink-0 border-b border-neutral-800" style={{ maxHeight: '50%' }}>
            <div className="px-3 py-1.5 border-b border-neutral-800 flex items-center justify-between">
              <span className="text-[10px] text-neutral-500 uppercase tracking-wider">Ensemble Forecasts</span>
              <span className="text-[9px] text-neutral-600">Open-Meteo GFS</span>
            </div>
            <div className="overflow-y-auto" style={{ maxHeight: 'calc(50vh - 60px)' }}>
              {forecasts.length === 0 ? (
                <div className="px-3 py-4 text-[10px] text-neutral-700">No forecast data</div>
              ) : (
                <div className="p-2 grid grid-cols-1 gap-2">
                  {forecasts.map(f => <ForecastCard key={f.city_key} f={f} />)}
                </div>
              )}
            </div>
          </div>

          {/* Trades */}
          <div className="flex-1 min-h-0 flex flex-col">
            <div className="shrink-0 px-3 py-1.5 border-b border-neutral-800 flex items-center justify-between">
              <span className="text-[10px] text-neutral-500 uppercase tracking-wider">Trades</span>
              <span className="text-[10px] text-neutral-600 tabular-nums">{data.recent_trades.length}</span>
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto">
              <TradesTable trades={data.recent_trades} />
            </div>
          </div>
        </div>
      </div>}

      {/* ── FOOTER ── */}
      <footer className="shrink-0 border-t border-neutral-800 px-4 py-1 flex items-center justify-between">
        <span className="text-[10px] text-neutral-700">
          Kalshi KXHIGH | Open-Meteo Ensemble | {stats.simulation_mode ? 'Simulation mode' : '⚡ Live trading'}
        </span>
        <div className="flex items-center gap-3 text-[10px] text-neutral-700">
          <span>{new Date().toLocaleDateString()}</span>
          <div className="flex items-center gap-1">
            <div className={`w-1.5 h-1.5 rounded-full ${kalshiOk ? 'bg-green-500' : 'bg-neutral-600'}`} />
            <span>{kalshiOk ? 'Connected' : 'No credentials'}</span>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
