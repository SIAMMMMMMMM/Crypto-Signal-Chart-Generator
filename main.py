#!/usr/bin/env python3
"""
=================================================
  AI Crypto Signal Bot - Chart Generator
  Generates 3 Timeframe Charts with Indicators
  Called by n8n Execute Command Node
=================================================
Usage: python3 chart_generator.py XRPUSDT
Output: 3 PNG images in /tmp/charts/
"""

import sys
import os
import json
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
SYMBOL        = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
OUTPUT_DIR    = "/tmp/charts"
BINANCE_URL   = "https://api.binance.com/api/v3/klines"
TIMEFRAMES    = [("15m", 80), ("1h", 80), ("4h", 80)]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  FETCH CANDLE DATA FROM BINANCE
# ─────────────────────────────────────────────
def fetch_candles(symbol, interval, limit=80):
    url = f"{BINANCE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    raw = r.json()
    data = []
    for c in raw:
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
#  INDICATOR CALCULATIONS
# ─────────────────────────────────────────────
def calc_sma(closes, period):
    sma = []
    for i in range(len(closes)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(sum(closes[i - period + 1:i + 1]) / period)
    return sma

def calc_rsi(closes, period=14):
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
            rsi.append(100)
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
#  GENERATE CHART IMAGE
# ─────────────────────────────────────────────
def generate_chart(symbol, interval, candles, output_path):
    closes  = [c["close"]  for c in candles]
    opens   = [c["open"]   for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    volumes = [c["volume"] for c in candles]
    times   = list(range(len(candles)))

    # Calculate indicators
    sma20   = calc_sma(closes, 20)
    sma50   = calc_sma(closes, 50)
    rsi     = calc_rsi(closes, 14)
    bb_up, bb_mid, bb_lo = calc_bollinger(closes, 20)

    # ── Layout ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor='#0d1117')
    gs  = gridspec.GridSpec(
        4, 1,
        height_ratios=[3.5, 0.8, 1, 1],
        hspace=0.0,
        figure=fig
    )

    ax_candle = fig.add_subplot(gs[0])  # Candlestick + BB + SMA
    ax_vol    = fig.add_subplot(gs[1], sharex=ax_candle)  # Volume
    ax_rsi    = fig.add_subplot(gs[2], sharex=ax_candle)  # RSI
    ax_info   = fig.add_subplot(gs[3])                    # Signal info box

    BG   = '#0d1117'
    GRID = '#1c2333'
    UP   = '#26a69a'
    DOWN = '#ef5350'
    TEXT = '#e0e0e0'
    SMA20_C = '#f5a623'
    SMA50_C = '#4a90e2'
    BB_C    = '#9c59b6'

    for ax in [ax_candle, ax_vol, ax_rsi]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.yaxis.tick_right()
        ax.grid(color=GRID, linewidth=0.4, linestyle='--')
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # ── Candlestick ──────────────────────────────────
    for i in times:
        color = UP if closes[i] >= opens[i] else DOWN
        # Wick
        ax_candle.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8)
        # Body
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        rect = plt.Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                              facecolor=color, edgecolor=color, linewidth=0.5)
        ax_candle.add_patch(rect)

    # ── Bollinger Bands ──────────────────────────────
    valid_t = [t for t in times if bb_up[t] is not None]
    ax_candle.plot(valid_t, [bb_up[t] for t in valid_t],
                   color=BB_C, linewidth=0.8, linestyle='--', alpha=0.7, label='BB Upper')
    ax_candle.plot(valid_t, [bb_lo[t] for t in valid_t],
                   color=BB_C, linewidth=0.8, linestyle='--', alpha=0.7, label='BB Lower')
    ax_candle.fill_between(valid_t,
                           [bb_up[t] for t in valid_t],
                           [bb_lo[t] for t in valid_t],
                           alpha=0.04, color=BB_C)

    # ── SMA Lines ────────────────────────────────────
    valid_s20 = [t for t in times if sma20[t] is not None]
    valid_s50 = [t for t in times if sma50[t] is not None]
    ax_candle.plot(valid_s20, [sma20[t] for t in valid_s20],
                   color=SMA20_C, linewidth=1.1, label='SMA 20')
    ax_candle.plot(valid_s50, [sma50[t] for t in valid_s50],
                   color=SMA50_C, linewidth=1.1, label='SMA 50')

    # Price label on right
    last_close = closes[-1]
    ax_candle.annotate(
        f'${last_close:,.4f}',
        xy=(times[-1], last_close),
        xytext=(times[-1] + 1, last_close),
        color=TEXT, fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1c2333', edgecolor=TEXT, linewidth=0.5)
    )

    # Title
    tf_label = {"15m": "15 Minute", "1h": "1 Hour", "4h": "4 Hour"}
    ax_candle.set_title(
        f'  {symbol}  ·  {tf_label.get(interval, interval)} Chart  ·  {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC',
        color=TEXT, fontsize=11, fontweight='bold',
        loc='left', pad=8,
        fontfamily='monospace'
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=SMA20_C, label='SMA 20'),
        mpatches.Patch(color=SMA50_C, label='SMA 50'),
        mpatches.Patch(color=BB_C,    label='Bollinger Bands'),
    ]
    ax_candle.legend(
        handles=legend_handles, loc='upper left',
        facecolor='#1c2333', edgecolor=GRID,
        labelcolor=TEXT, fontsize=7.5
    )

    # ── Volume Bars ───────────────────────────────────
    avg_vol = sum(volumes) / len(volumes)
    for i in times:
        col = UP if closes[i] >= opens[i] else DOWN
        alpha = 0.9 if volumes[i] > avg_vol * 1.5 else 0.5
        ax_vol.bar(i, volumes[i], color=col, alpha=alpha, width=0.7)
    ax_vol.axhline(avg_vol, color='#ffffff', linewidth=0.6,
                   linestyle=':', alpha=0.4, label='Avg Vol')
    ax_vol.set_ylabel('VOL', color=TEXT, fontsize=7)

    # ── RSI Panel ─────────────────────────────────────
    valid_rsi = [t for t in times if rsi[t] is not None]
    rsi_vals  = [rsi[t] for t in valid_rsi]
    ax_rsi.plot(valid_rsi, rsi_vals, color='#e91e63', linewidth=1.1)
    ax_rsi.axhline(70, color=DOWN,  linewidth=0.7, linestyle='--', alpha=0.7)
    ax_rsi.axhline(30, color=UP,    linewidth=0.7, linestyle='--', alpha=0.7)
    ax_rsi.axhline(50, color=TEXT,  linewidth=0.4, linestyle=':', alpha=0.3)
    ax_rsi.fill_between(valid_rsi, rsi_vals, 70,
                        where=[v > 70 for v in rsi_vals],
                        alpha=0.15, color=DOWN, interpolate=True)
    ax_rsi.fill_between(valid_rsi, rsi_vals, 30,
                        where=[v < 30 for v in rsi_vals],
                        alpha=0.15, color=UP, interpolate=True)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_yticks([30, 50, 70])
    ax_rsi.set_ylabel('RSI', color=TEXT, fontsize=7)

    # Current RSI label
    if rsi_vals:
        current_rsi = rsi_vals[-1]
        rsi_color = DOWN if current_rsi > 70 else UP if current_rsi < 30 else '#f5a623'
        ax_rsi.annotate(
            f'RSI: {current_rsi:.1f}',
            xy=(0.01, 0.75), xycoords='axes fraction',
            color=rsi_color, fontsize=8, fontweight='bold'
        )

    # ── Info Box (Bottom) ─────────────────────────────
    ax_info.set_facecolor('#111827')
    ax_info.axis('off')

    # Compute quick signal for this TF
    last_rsi    = rsi[-1] if rsi[-1] else 50
    last_sma20  = sma20[-1] if sma20[-1] else last_close
    last_sma50  = sma50[-1] if sma50[-1] else last_close
    trend       = "BULLISH" if last_close > last_sma50 else "BEARISH"
    rsi_status  = "OVERBOUGHT ⚠️" if last_rsi > 70 else "OVERSOLD 🟢" if last_rsi < 30 else "NEUTRAL ✅"
    bb_pos      = "ABOVE UPPER 🔴" if last_close > (bb_up[-1] or last_close) \
                  else "BELOW LOWER 🟢" if last_close < (bb_lo[-1] or last_close) \
                  else "INSIDE BANDS ✅"

    info_text = (
        f"  Trend: {trend}   │   RSI({14}): {last_rsi:.1f} → {rsi_status}   │"
        f"   SMA20: ${last_sma20:,.4f}   │   SMA50: ${last_sma50:,.4f}   │   BB: {bb_pos}"
    )
    ax_info.text(
        0.5, 0.55, info_text,
        transform=ax_info.transAxes,
        color=TEXT, fontsize=7.8,
        ha='center', va='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1c2333',
                  edgecolor='#30363d', linewidth=0.8)
    )

    plt.setp(ax_candle.get_xticklabels(), visible=False)
    plt.setp(ax_vol.get_xticklabels(), visible=False)

    # Time labels on RSI x-axis
    tick_step = max(1, len(times) // 8)
    tick_positions = times[::tick_step]
    ax_rsi.set_xticks(tick_positions)
    ax_rsi.set_xticklabels(
        [candles[t]["time"].strftime("%H:%M") for t in tick_positions],
        color=TEXT, fontsize=6.5, rotation=0
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"✅ Chart saved: {output_path}")

    return {
        "timeframe": interval,
        "last_close": last_close,
        "rsi": round(last_rsi, 2),
        "sma20": round(last_sma20, 4),
        "sma50": round(last_sma50, 4),
        "trend": trend,
        "rsi_status": rsi_status
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print(f"\n🚀 Generating charts for {SYMBOL}...")
    results = {}

    for interval, limit in TIMEFRAMES:
        print(f"  📊 Fetching {interval} candles...")
        candles = fetch_candles(SYMBOL, interval, limit)
        path    = os.path.join(OUTPUT_DIR, f"{SYMBOL}_{interval}.png")
        info    = generate_chart(SYMBOL, interval, candles, path)
        results[interval] = {**info, "path": path}

    # Output JSON for n8n to read
    output = {
        "symbol":     SYMBOL,
        "generated":  datetime.utcnow().isoformat(),
        "charts":     results,
        "chart_15m":  results.get("15m", {}).get("path", ""),
        "chart_1h":   results.get("1h",  {}).get("path", ""),
        "chart_4h":   results.get("4h",  {}).get("path", ""),
        "summary": {
            "15m_trend": results.get("15m", {}).get("trend", "N/A"),
            "1h_trend":  results.get("1h",  {}).get("trend", "N/A"),
            "4h_trend":  results.get("4h",  {}).get("trend", "N/A"),
            "15m_rsi":   results.get("15m", {}).get("rsi", "N/A"),
            "1h_rsi":    results.get("1h",  {}).get("rsi", "N/A"),
            "4h_rsi":    results.get("4h",  {}).get("rsi", "N/A"),
        }
    }

    print(json.dumps(output, indent=2))
    return output

if __name__ == "__main__":
    main()
