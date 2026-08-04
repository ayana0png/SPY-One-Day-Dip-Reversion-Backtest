# SPY One-Day Dip-Reversion Backtest
#
# Research question:
# After SPY falls more than 2% in one session while remaining above its
# 200-day moving average, does it tend to rebound during the next session?
#
# Trade timing:
# 1. Detect the signal after today's close.
# 2. Enter at the next trading day's open.
# 3. Exit at that same trading day's close.
#
# This is an educational historical study, not financial advice.
# Results exclude commissions, slippage, taxes, and execution delays.

import sys

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


TICKER = "SPY"

# Extra history is needed to calculate the 200-day moving average.
DATA_START = "2020-04-01"

# Signals are evaluated beginning on this date.
STRATEGY_START = "2021-04-01"

# yfinance treats the end date as exclusive, so 2026-07-01 includes
# available trading data through 2026-06-30.
DATA_END = "2026-07-01"

DROP_THRESHOLD = -0.02
MA_WINDOW = 200


# SECTION 1: Download historical data

print("Downloading SPY data...")

df = yf.download(
    TICKER,
    start=DATA_START,
    end=DATA_END,
    auto_adjust=True,
    progress=False,
)

if df.empty:
    raise RuntimeError("No SPY data was downloaded. Check your connection and try again.")

# Current yfinance versions may return multi-level columns.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

required_columns = {"Open", "Close"}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise RuntimeError(
        f"Downloaded data is missing required columns: {sorted(missing_columns)}"
    )

df = df[["Open", "Close"]].dropna().copy()


# SECTION 2: Calculate daily returns and long-term trend

df["Daily_Return"] = df["Close"].pct_change(fill_method=None)

df["MA200"] = (
    df["Close"]
    .rolling(window=MA_WINDOW, min_periods=MA_WINDOW)
    .mean()
)

# Remove the earlier warm-up period after calculating the moving average.
df = df[df.index >= STRATEGY_START].copy()


# SECTION 3: Generate signals

# A signal occurs after the market closes when:
# 1. SPY fell more than 2% that day.
# 2. SPY still closed above its 200-day moving average.

df["Signal"] = (
    (df["Daily_Return"] < DROP_THRESHOLD)
    & (df["Close"] > df["MA200"])
    & df["MA200"].notna()
).astype(int)


# SECTION 4: Build the trades table

trades = []

for i in range(len(df) - 1):
    if df["Signal"].iloc[i] != 1:
        continue

    signal_date = df.index[i]
    trade_date = df.index[i + 1]

    signal_close = float(df["Close"].iloc[i])
    ma200 = float(df["MA200"].iloc[i])

    # The signal is known only after the signal day's close.
    # Therefore, the trade begins at the next session's open.
    entry_price = float(df["Open"].iloc[i + 1])
    exit_price = float(df["Close"].iloc[i + 1])

    trade_return = (exit_price - entry_price) / entry_price

    trades.append(
        {
            "Signal Date": signal_date.date(),
            "Trade Date": trade_date.date(),
            "Signal Close": round(signal_close, 2),
            "MA200": round(ma200, 2),
            "Entry Open": round(entry_price, 2),
            "Exit Close": round(exit_price, 2),
            "Trade Return": round(trade_return, 4),
        }
    )

trades_df = pd.DataFrame(trades)

if trades_df.empty:
    print(
        "\nNo trades met the selected conditions during the fixed study period."
    )
    sys.exit(0)


# SECTION 5: Display and save the trades

print("\n── All Trades ──────────────────────────────────────────")
print(trades_df.to_string(index=False))

trades_df.to_csv(
    "spy_dip_reversion_trades.csv",
    index=False,
)


# SECTION 6: Calculate summary statistics

number_of_trades = len(trades_df)
win_rate = (trades_df["Trade Return"] > 0).mean()
average_return = trades_df["Trade Return"].mean()
median_return = trades_df["Trade Return"].median()
best_trade = trades_df["Trade Return"].max()
worst_trade = trades_df["Trade Return"].min()

# This compounds only the recorded trade returns.
# Capital is assumed to remain unchanged while no trade is active.
cumulative_return = (
    (1 + trades_df["Trade Return"]).prod() - 1
)

print("\n── Summary Statistics ──────────────────────────────────")
print(f"  Fixed Study Period    : {STRATEGY_START} to 2026-06-30")
print(f"  Signal Threshold      : Below {DROP_THRESHOLD:.0%}")
print(f"  Trend Filter          : Close above {MA_WINDOW}-day MA")
print(f"  Entry / Exit          : Next open / next close")
print(f"  Number of Trades      : {number_of_trades}")
print(f"  Win Rate              : {win_rate:.1%}")
print(f"  Average Trade Return  : {average_return:.2%}")
print(f"  Median Trade Return   : {median_return:.2%}")
print(f"  Compounded Trade Ret. : {cumulative_return:.2%}")
print(f"  Best Trade            : {best_trade:.2%}")
print(f"  Worst Trade           : {worst_trade:.2%}")
print("  Costs and Slippage    : Not included")


# SECTION 7: Plot compounded trade returns

trades_df["Cumulative_Return"] = (
    (1 + trades_df["Trade Return"]).cumprod() - 1
)

plt.figure(figsize=(10, 5))

plt.plot(
    pd.to_datetime(trades_df["Trade Date"]),
    trades_df["Cumulative_Return"] * 100,
    marker="o",
    linewidth=2,
    label="Compounded Trade Return (%)",
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.title(
    "SPY One-Day Dip-Reversion Backtest\n"
    "Next-Session Open to Close | 200-Day Trend Filter",
    fontsize=14,
)

plt.xlabel("Trade Date")
plt.ylabel("Compounded Trade Return (%)")
plt.legend()
plt.tight_layout()

chart_filename = "spy_dip_reversion_results.png"

plt.savefig(
    chart_filename,
    dpi=150,
    bbox_inches="tight",
)

print(f"\nTrades saved as spy_dip_reversion_trades.csv")
print(f"Chart saved as {chart_filename}")

plt.show()