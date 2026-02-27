#!/usr/bin/env python3
"""
================================================
  Crypto Signal Bot - Chart Generator API
  Runs on Railway.app as a web service
  Called by n8n via HTTP Request node
================================================
Endpoints:
  GET  /health          → health check
  POST /generate-charts → generate 3 charts
  
Request body:
  { "symbol": "XRPUSDT" }

Response:
  {
    "symbol": "XRPUSDT",
    "chart_15m": "base64...",
    "chart_1h":  "base64...",
    "chart_4h":  "base64...",
    "summary": {
      "15m_trend": "BULLISH",
      "15m_rsi": 58.2,
      ...
    }
  }
"""

import os
import io
import json
import base64
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

BINANCE_URL = "https://api.binance.com/api/v3/klines"


# ─────────────────────────────────────────────
#  INDICATOR CALCULATIONS
# ─────────────────────────────────────────────

def calc_sma(closes, period):
    result = []
    for i in range(len(closes)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(closes[i - period + 1:i + 1]) / period)
    return result


def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return [None] * len(closes)
    
    rsi = [None] * period
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0)) / period
        if avg_loss == 0:
            rsi.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(round(100 - (100 / (1 + rs)), 2))
    return rsi


def calc_bollinger(closes, period=20, std_dev=2):
    upper, lower, mid = [], [], []
    for i in range(len(closes)):
        if i < period - 1:
            upper.append(None); lower.append(None); mid.append(None)
        else:
            window = closes[i - period + 1:i + 1]
            m = sum(window) / period
            sd = (sum((x - m) ** 2 for x in window) / period) ** 0.5
            mid.append(m)
            upper.append(m + std_dev * sd)
            lower.append(m - std_dev * sd)
    return upper, mid, lower


# ─────────────────────────────────────────────
#  FETCH CANDLES FROM BINANCE
# ─────────────────────────────────────────────

def fetch_candles(symbol, interval, limit=80):
    url = f"{BINANCE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = []
    for c in r.json():
        data.append({
            "time":   datetime.fromtimestamp(c[0] / 1000),
            "open":   float(c[1]),
            "high":   float(c[2]),
            "low":    float(c[3]),
            "close":  float(c[4]),
            "volume": float(c[5])
        })
    return data


# ─────────────────────────────────────────────
#  GENERATE ONE CHART → returns base64 string
# ─────────────────────────────────────────────

def generate_chart_base64(symbol, interval, candles):
    closes  = [c["close"]  for c in candles]
    opens   = [c["open"]   for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    volumes = [c["volume"] for c in candles]
    times   = list(range(len(candles)))

    sma20        = calc_sma(closes, 20)
    sma50        = calc_sma(closes, 50)
    rsi_vals     = calc_rsi(closes, 14)
    bb_up, bb_mid, bb_lo = calc_bollinger(closes, 20)

    # ── Colors ──────────────────────────────────────
    BG      = '#0d1117'
    GRID    = '#1c2333'
    UP      = '#26a69a'
    DOWN    = '#ef5350'
    TEXT    = '#e0e0e0'
    SMA20_C = '#f5a623'
    SMA50_C = '#4a90e2'
    BB_C    = '#9c59b6'

    # ── Layout ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(4, 1, height_ratios=[3.5, 0.8, 1, 0.6], hspace=0.0, figure=fig)

    ax_candle = fig.add_subplot(gs[0])
    ax_vol    = fig.add_subplot(gs[1], sharex=ax_candle)
    ax_rsi    = fig.add_subplot(gs[2], sharex=ax_candle)
    ax_info   = fig.add_subplot(gs[3])

    for ax in [ax_candle, ax_vol, ax_rsi]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.yaxis.tick_right()
        ax.grid(color=GRID, linewidth=0.4, linestyle='--')
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # ── Candlesticks ─────────────────────────────────
    for i in times:
        color = UP if closes[i] >= opens[i] else DOWN
        ax_candle.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8)
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i]) or (closes[i] * 0.0001)
        rect = plt.Rectangle(
            (i - 0.3, body_bottom), 0.6, body_height,
            facecolor=color, edgecolor=color, linewidth=0.3
        )
        ax_candle.add_patch(rect)

    # ── Bollinger Bands ──────────────────────────────
    v_t = [t for t in times if bb_up[t] is not None]
    ax_candle.plot(v_t, [bb_up[t] for t in v_t], color=BB_C, lw=0.8, ls='--', alpha=0.7)
    ax_candle.plot(v_t, [bb_lo[t] for t in v_t], color=BB_C, lw=0.8, ls='--', alpha=0.7)
    ax_candle.fill_between(v_t, [bb_up[t] for t in v_t], [bb_lo[t] for t in v_t],
                           alpha=0.04, color=BB_C)

    # ── SMA Lines ────────────────────────────────────
    v_s20 = [t for t in times if sma20[t] is not None]
    v_s50 = [t for t in times if sma50[t] is not None]
    ax_candle.plot(v_s20, [sma20[t] for t in v_s20], color=SMA20_C, lw=1.1, label='SMA20')
    ax_candle.plot(v_s50, [sma50[t] for t in v_s50], color=SMA50_C, lw=1.1, label='SMA50')

    # ── Price Label ──────────────────────────────────
    last_close = closes[-1]
    ax_candle.annotate(
        f'${last_close:,.5f}',
        xy=(times[-1], last_close),
        color=TEXT, fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1c2333', edgecolor=TEXT, lw=0.5)
    )

    # ── Title ────────────────────────────────────────
    tf_label = {"15m": "15 Minute", "1h": "1 Hour", "4h": "4 Hour"}
    ax_candle.set_title(
        f'  {symbol}  ·  {tf_label.get(interval, interval)}  ·  {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC',
        color=TEXT, fontsize=11, fontweight='bold', loc='left', pad=8, fontfamily='monospace'
    )

    # ── Legend ───────────────────────────────────────
    handles = [
        mpatches.Patch(color=SMA20_C, label='SMA 20'),
        mpatches.Patch(color=SMA50_C, label='SMA 50'),
        mpatches.Patch(color=BB_C,    label='Bollinger Bands'),
    ]
    ax_candle.legend(handles=handles, loc='upper left',
                     facecolor='#1c2333', edgecolor=GRID, labelcolor=TEXT, fontsize=7.5)

    # ── Volume ───────────────────────────────────────
    avg_vol = sum(volumes) / len(volumes)
    for i in times:
        col = UP if closes[i] >= opens[i] else DOWN
        alpha = 0.9 if volumes[i] > avg_vol * 1.5 else 0.5
        ax_vol.bar(i, volumes[i], color=col, alpha=alpha, width=0.7)
    ax_vol.axhline(avg_vol, color='#ffffff', lw=0.6, ls=':', alpha=0.4)
    ax_vol.set_ylabel('VOL', color=TEXT, fontsize=7)

    # ── RSI ──────────────────────────────────────────
    v_rsi = [t for t in times if rsi_vals[t] is not None]
    rsi_y  = [rsi_vals[t] for t in v_rsi]
    ax_rsi.plot(v_rsi, rsi_y, color='#e91e63', lw=1.1)
    ax_rsi.axhline(70, color=DOWN, lw=0.7, ls='--', alpha=0.7)
    ax_rsi.axhline(30, color=UP,   lw=0.7, ls='--', alpha=0.7)
    ax_rsi.axhline(50, color=TEXT, lw=0.4, ls=':', alpha=0.3)
    ax_rsi.fill_between(v_rsi, rsi_y, 70, where=[v > 70 for v in rsi_y],
                         alpha=0.15, color=DOWN, interpolate=True)
    ax_rsi.fill_between(v_rsi, rsi_y, 30, where=[v < 30 for v in rsi_y],
                         alpha=0.15, color=UP, interpolate=True)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_yticks([30, 50, 70])
    ax_rsi.set_ylabel('RSI', color=TEXT, fontsize=7)

    current_rsi = rsi_y[-1] if rsi_y else 50
    rsi_color = DOWN if current_rsi > 70 else UP if current_rsi < 30 else '#f5a623'
    ax_rsi.annotate(f'RSI: {current_rsi:.1f}', xy=(0.01, 0.75),
                    xycoords='axes fraction', color=rsi_color, fontsize=8, fontweight='bold')

    # ── Info Bar ─────────────────────────────────────
    ax_info.set_facecolor('#111827')
    ax_info.axis('off')

    last_sma20 = next((sma20[i] for i in range(len(sma20)-1, -1, -1) if sma20[i] is not None), last_close)
    last_sma50 = next((sma50[i] for i in range(len(sma50)-1, -1, -1) if sma50[i] is not None), last_close)
    trend = "BULLISH ▲" if last_close > last_sma50 else "BEARISH ▼"
    rsi_status = "OVERBOUGHT ⚠" if current_rsi > 70 else "OVERSOLD ✓" if current_rsi < 30 else "NEUTRAL"
    bb_pos = "ABOVE UPPER" if last_close > (bb_up[-1] or last_close) \
             else "BELOW LOWER" if last_close < (bb_lo[-1] or last_close) else "INSIDE BANDS ✓"

    info = (f"  Trend: {trend}   │   RSI: {current_rsi:.1f} → {rsi_status}   │"
            f"   SMA20: ${last_sma20:,.5f}   │   SMA50: ${last_sma50:,.5f}   │   BB: {bb_pos}")
    ax_info.text(0.5, 0.5, info, transform=ax_info.transAxes,
                 color=TEXT, fontsize=7.5, ha='center', va='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#1c2333', edgecolor='#30363d', lw=0.8))

    # ── X-axis time labels ───────────────────────────
    plt.setp(ax_candle.get_xticklabels(), visible=False)
    plt.setp(ax_vol.get_xticklabels(), visible=False)
    step = max(1, len(times) // 8)
    ax_rsi.set_xticks(times[::step])
    ax_rsi.set_xticklabels(
        [candles[t]["time"].strftime("%H:%M") for t in times[::step]],
        color=TEXT, fontsize=6.5
    )

    # ── Save to bytes → Base64 ───────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8'), {
        "trend":      "BULLISH" if last_close > last_sma50 else "BEARISH",
        "rsi":        round(current_rsi, 2),
        "sma20":      round(last_sma20, 6),
        "sma50":      round(last_sma50, 6),
        "last_close": round(last_close, 6)
    }


# ─────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "crypto-signal-chart-generator"})


@app.route('/generate-charts', methods=['POST'])
def generate_charts():
    try:
        body   = request.get_json(force=True)
        symbol = body.get('symbol', 'BTCUSDT').upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'

        results = {}
        for interval, limit in [('15m', 80), ('1h', 80), ('4h', 80)]:
            candles          = fetch_candles(symbol, interval, limit)
            b64, info        = generate_chart_base64(symbol, interval, candles)
            results[interval] = {"base64": b64, "info": info}

        return jsonify({
            "symbol":    symbol,
            "generated": datetime.utcnow().isoformat(),
            "chart_15m": results["15m"]["base64"],
            "chart_1h":  results["1h"]["base64"],
            "chart_4h":  results["4h"]["base64"],
            "summary": {
                "15m_trend": results["15m"]["info"]["trend"],
                "15m_rsi":   results["15m"]["info"]["rsi"],
                "1h_trend":  results["1h"]["info"]["trend"],
                "1h_rsi":    results["1h"]["info"]["rsi"],
                "4h_trend":  results["4h"]["info"]["trend"],
                "4h_rsi":    results["4h"]["info"]["rsi"],
                "price":     results["15m"]["info"]["last_close"]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
