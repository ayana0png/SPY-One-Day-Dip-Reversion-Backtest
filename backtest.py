# Quant Project with 200-Day MA Trend Filter
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# SECTION 1: Download Data
# Pull daily SPY price data from Yahoo Finance.
# 'auto_adjust=True' gives us clean prices adjusted for splits/dividends.
#
# IMPORTANT: We start the download on 2020-04-01 a full year before our
# strategy start date of 2021-04-01. We need this extra history so that the
# 200-day moving average has enough data points to be valid by the time we
# start trading. Without it, the first 200 rows would have NaN for the MA
# and all those signals would be skipped silently.

print("Downloading SPY data...")
df = yf.download("SPY", start="2020-04-01", auto_adjust=True, progress=False)

# Flatten multi-level columns if yfinance returns them that way
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Keep only the columns we need
df = df[["Close"]].copy()

# SECTION 2: Compute Daily Returns 
# pct_change() calculates: (today's close - yesterday's close) / yesterday's close
# This gives us a decimal return, e.g. -0.025 means a -2.5% day.

df["Daily_Return"] = df["Close"].pct_change()

# SECTION 3: Compute the 200-Day Moving Average 
# A moving average smooths out daily noise by averaging the last N closing prices.
# The 200-day MA is one of the most widely watched trend indicators on Wall Street.
#
# How it works:
#   On any given day, df["MA200"] = average of the past 200 closing prices.
#   - If Close > MA200 → SPY is in an uptrend = good time to look for bounces
#   - If Close < MA200 → SPY is in a downtrend = drops may keep dropping
#
# rolling(200) creates a sliding 200-row window; .mean() averages each window.
# The first 199 rows will be NaN because there isn't enough history yet
# that's expected and fine; those rows will simply never trigger a signal.

df["MA200"] = df["Close"].rolling(200).mean()

# Trim the DataFrame to our actual strategy period (2021-04-01 onward).
# earlier rows were only downloaded to warm up the 200-day MA calculation.
df = df[df.index >= "2021-04-01"].copy()

# SECTION 4: Generate Buy Signals 
# Mean reversion logic: after a big drop (< -2%) the price bounces back (often).
# We then can add a trend filter: only trade bounces that happen in an uptrend.
# Both conditions must be TRUE on the same day to fire a signal:
#   Condition A — Daily_Return < -0.02  → today was a big down day (the dip)
#   Condition B — Close > MA200         → price is still above the long-term trend
#
# Why add the MA filter?
#   Without it the strategy also buys dips during prolonged bear markets where
#   prices can keep falling for weeks. The MA filter ensures that the script knows to
#   only buy the dip if the overall trend is still healthy.
#
# Signal = 1 means "buy at today's close, sell at tomorrow's close".

df["Signal"] = (
    (df["Daily_Return"] < -0.02) &   # big down day
    (df["Close"] > df["MA200"])       # price is above the 200-day MA (uptrend)
).astype(int)

# SECTION 5: Build the Trades Table 
# For every signal day, record the buy date, sell date, prices, and return.

trades = []

for i in range(len(df) - 1):  # stop one row early so we always have a "next day"
    if df["Signal"].iloc[i] == 1:
        buy_date  = df.index[i]
        sell_date = df.index[i + 1]
        buy_price  = df["Close"].iloc[i]
        sell_price = df["Close"].iloc[i + 1]
        trade_return = (sell_price - buy_price) / buy_price

        trades.append({
            "Buy Date":      buy_date.date(),
            "Sell Date":     sell_date.date(),
            "Buy Price":     round(float(buy_price), 2),
            "Sell Price":    round(float(sell_price), 2),
            "MA200":         round(float(df["MA200"].iloc[i]), 2),  # shows how far above the trend we were
            "Trade Return":  round(float(trade_return), 4),
        })

trades_df = pd.DataFrame(trades)

print("\n── All Trades ──────────────────────────────────────────")
print(trades_df.to_string(index=False))

# ── SECTION 6: Summary Statistics ────────────────────────────
# Quick performance snapshot so you know if the strategy is any good.

n_trades       = len(trades_df)
win_rate       = (trades_df["Trade Return"] > 0).mean()
avg_return     = trades_df["Trade Return"].mean()
best_trade     = trades_df["Trade Return"].max()
worst_trade    = trades_df["Trade Return"].min()

# Cumulative return: multiply all (1 + trade_return) factors together
cumulative_return = (1 + trades_df["Trade Return"]).prod() - 1

print("\n── Summary Statistics ──────────────────────────────────")
print(f"  Number of Trades     : {n_trades}")
print(f"  Win Rate             : {win_rate:.1%}")
print(f"  Avg Trade Return     : {avg_return:.2%}")
print(f"  Total Cumul. Return  : {cumulative_return:.2%}")
print(f"  Best Trade           : {best_trade:.2%}")
print(f"  Worst Trade          : {worst_trade:.2%}")

#  SECTION 7: Plot Cumulative Returns Over Time 
# Show how $1 invested across all trades would have grown (or shrunk).

trades_df["Cumulative_Return"] = (1 + trades_df["Trade Return"]).cumprod() - 1

plt.figure(figsize=(10, 5))
plt.plot(
    pd.to_datetime(trades_df["Sell Date"]),
    trades_df["Cumulative_Return"] * 100,
    marker="o",
    linewidth=2,
    color="steelblue",
    label="Strategy Cumulative Return (%)"
)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.title("SPY Mean Reversion Strategy (200-MA Filter) — Cumulative Return", fontsize=14)
plt.xlabel("Sell Date")
plt.ylabel("Cumulative Return (%)")
plt.legend()
plt.tight_layout()
plt.savefig("spy_strategy_returns_ma200.png", dpi=150)
plt.show()
print("\nChart saved as spy_strategy_returns_ma200.png")
