# 🚀 Quick Start Guide

Get MTC Trading v1.0 running in under 5 minutes.

---

## ⚡ Prerequisites

Before you begin, make sure you have:
- **Python 3.8 or higher** installed ([Download here](https://www.python.org/downloads/))
- **pip** (Python package manager - comes with Python)
- **Internet connection** (required for downloading market data)

**Check your Python version:**
```bash
python --version
# or
python3 --version
```

Expected output: `Python 3.8.x` or higher

---

## 📦 Installation

### Step 1: Get the Code

**Option A: Download ZIP**
1. Click the green **"Code"** button at the top of this page
2. Select **"Download ZIP"**
3. Extract the ZIP file to your desired location

**Option B: Clone with Git**
```bash
git clone https://github.com/yourusername/Trading-Research-Framework.git
cd Trading-Research-Framework
```

### Step 2: Install Dependencies

Open your terminal/command prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

**If you get a "pip not found" error, try:**
```bash
python -m pip install -r requirements.txt
# or
python3 -m pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed pandas-2.1.0 numpy-1.24.0 yfinance-0.2.28 matplotlib-3.7.0
```

---

## 🎮 Running Your First Backtest

### Quick Test (30 seconds)

Run the program:
```bash
python "Best Version.py"
```

**On Mac/Linux, use:**
```bash
python3 "Best Version.py"
```

### Interactive Menu

You'll see the main menu:
```
==================================================
🚀 MTC Trading Bot v1.0 - Ultimate Edition
==================================================
⚠️  Disclaimer: Past performance does not guarantee future results
💎 'If you can't hold, you won't be rich.' - CZ
==================================================

MAIN MENU:
--------------------------------------------------
 1. 📊 Single Asset Trading (Advanced Risk Management)
 2. 💼 Multi-Asset Portfolio Management
 3. 🔬 Strategy Comparison (All Strategies)
 4. ⚡ Quick Backtest (SPY with defaults)
 5. 📚 Strategy Performance Report
 6. ❌ Exit
--------------------------------------------------

👉 Select option (1-6):
```

### Try the Quick Backtest

Type **`4`** and press Enter:
```
👉 Select option (1-6): 4
```

**That's it!** The system will automatically:
1. ✅ Download SPY historical data (2022-2024)
2. ✅ Run a Moving Average Crossover strategy (50/200)
3. ✅ Calculate comprehensive performance metrics
4. ✅ Generate a performance chart

**Expected output:**
```
📥 Fetching data for SPY...
   Period: 2022-01-01 to 2024-12-31
------------------------------------------------------------
✅ Successfully loaded 756 trading days
   Actual range: 2022-01-03 to 2024-12-31
   Price range: $348.12 - $589.45

🔄 Running backtest...
   Strategy: MA Crossover (50/200)
   Mode: REALISTIC
   Risk Management: ON
------------------------------------------------------------
✅ Backtest completed

======================================================================
PERFORMANCE METRICS - SPY
Strategy: MA Crossover (50/200)
======================================================================
Total Return (%)............................................... +42.30
Buy & Hold Return (%)..........................................  +35.10
Outperformance (%)............................................. +7.20
Sharpe Ratio................................................... 1.18
Sortino Ratio.................................................. 1.45
Calmar Ratio................................................... 3.41
Max Drawdown (%).............................................. -12.40
Volatility (%)................................................ 18.50
Win Rate (%).................................................. 54.20
Max Win Streak................................................ 8
Max Loss Streak............................................... 5
Final Portfolio Value ($)..................................... $142,300.00
Total Trades.................................................. 12
Stop Losses Triggered......................................... 2
Take Profits Triggered........................................ 3
Trailing Stops Triggered...................................... 1
======================================================================
```

A chart window will pop up showing your strategy's performance! 📊

---

## 📊 Running a Custom Backtest

### Example 1: Test a Different Stock

Select option **`1`** from the main menu:
```
👉 Select option (1-6): 1

Enter ticker (e.g., SPY, AAPL, TSLA): AAPL
Start date (YYYY-MM-DD) [default: 2022-01-01]: 2020-01-01
End date (YYYY-MM-DD) [default: 2025-01-19]: 2024-12-31
Initial capital [default: $100,000]: 50000

📋 Select Strategy:
1. Moving Average Crossover
2. Momentum Strategy
3. Mean Reversion (Bollinger Bands)
4. RSI Strategy

Strategy (1-4) [default: 1]: 1

Short MA window [default: 50]: 50
Long MA window [default: 200]: 200

Enable risk management? (Y/n) [default: Y]: Y
```

Press Enter through defaults or customize as needed!

### Example 2: Compare All Strategies

Select option **`3`** to see how all strategies perform on the same asset:
```
👉 Select option (1-6): 3

Enter ticker [default: SPY]: TSLA
```

This will run all 4 strategies and show a comparison table! 📈

### Example 3: Multi-Asset Portfolio

Select option **`2`** to build a diversified portfolio:
```
👉 Select option (1-6): 2

Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL): AAPL,MSFT,GOOGL,NVDA
Start date [default: 2022-01-01]: 
End date [default: 2025-10-15]: 
Initial capital [default: $100,000]: 

📊 Allocation Methods:
1. Equal Weight
2. Risk Parity
3. Momentum Weighted
4. Market Cap Weighted

Select method (1-4): 2
```

---

## 🎯 Menu Options Explained

| Option | What It Does | Best For |
|--------|--------------|----------|
| **1. Single Asset Trading** | Backtest one stock with full customization | Testing specific strategies and parameters |
| **2. Multi-Asset Portfolio** | Build and analyze diversified portfolios | Portfolio optimization and diversification |
| **3. Strategy Comparison** | Compare all 4 strategies side-by-side | Finding the best strategy for an asset |
| **4. Quick Backtest** | Run SPY with defaults (fastest option) | Quick demonstration and testing |
| **5. Performance Report** | Detailed analytics and trade breakdown | Deep analysis of strategy performance |
| **6. Exit** | Close the program | When you're done! |

---

## 🐛 Troubleshooting

### Problem: "No module named 'pandas'" (or other package)
**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "No data returned for [ticker]"
**Solutions:**
- ✅ Check the ticker symbol is correct (use Yahoo Finance format: `AAPL`, not `Apple`)
- ✅ Verify you have an internet connection
- ✅ Try a more recent date range (some tickers have limited history)
- ✅ Check if the ticker exists on [Yahoo Finance](https://finance.yahoo.com)

### Problem: "Insufficient data: only X valid days"
**Solution:** The strategy needs at least 20 trading days. Extend your date range:
```
Start date: 2020-01-01  (instead of 2024-01-01)
End date: 2024-12-31
```

### Problem: Charts not displaying
**Solutions:**
```bash
# Upgrade matplotlib
pip install matplotlib --upgrade

# On Mac, you may need:
pip install matplotlib --upgrade --force-reinstall
```

### Problem: "python: command not found"
**Solution:**
```bash
# Try python3 instead
python3 "Best Version.py"

# Or add Python to your PATH (Windows)
# Google: "Add Python to PATH Windows"
```

### Problem: Program crashes or freezes
**Solutions:**
- ✅ Make sure you have a stable internet connection (yfinance needs it)
- ✅ Try a different ticker or date range
- ✅ Restart the program
- ✅ Check you have at least Python 3.8: `python --version`

---

## 💡 Pro Tips

### Tip 1: Start Simple
Use the Quick Backtest (option 4) first to verify everything works before trying custom parameters.

### Tip 2: Realistic vs Optimistic Mode
- **Realistic** (default): Executes trades the next day (no look-ahead bias) ✅ Recommended
- **Optimistic**: Executes trades same day (for research comparison only)

### Tip 3: Risk Management
Keep it **ON** unless you're specifically testing theoretical performance. It prevents catastrophic losses.

### Tip 4: Date Ranges
- Use at least 2+ years for meaningful backtest results
- Include different market conditions (bull, bear, sideways)
- End date can be "today" - the system will use the most recent data

### Tip 5: Save Your Charts
When asked "Generate chart?", say Yes! Charts are automatically saved as PNG files in your project folder.

---

## 📚 What's Next?

Now that you're up and running, explore:

1. **[Usage Examples](EXAMPLES.md)** - Advanced code examples and patterns
2. **[Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md)** - Understand the formulas and calculations
3. **[Main README](README.md)** - Complete project documentation
4. **[Changelog](CHANGELOG.md)** - Version history and updates

---

## 🎓 Understanding Your Results

### Key Metrics Explained

**Total Return**: Your strategy's profit/loss percentage
- Good: >20% over 2-3 years
- Great: >40% over 2-3 years

**Sharpe Ratio**: Risk-adjusted return (higher is better)
- <1.0 = Poor
- 1.0-2.0 = Good ✅
- >2.0 = Excellent

**Max Drawdown**: Largest peak-to-trough decline (lower is better)
- <10% = Excellent
- 10-20% = Good ✅
- >30% = High risk

**Win Rate**: Percentage of winning trades
- >50% = Positive edge
- >60% = Strong strategy

**Outperformance**: How much you beat Buy & Hold
- Positive = You beat the market! 🎉
- Negative = Buy & Hold was better

---

## 💬 Need Help?

**Found a bug?** [Open an issue](https://github.com/yourusername/Trading-Research-Framework/issues)

**Have a question?** Check the [Examples](EXAMPLES.md) or [Main Documentation](README.md)

**Want to contribute?** See [Contributing Guidelines](CONTRIBUTING.md) *(coming soon)*

---

## ⚡ Quick Reference Card
```bash
# Install dependencies
pip install -r requirements.txt

# Run program
python "Best Version.py"

# Quick test (select option 4)
# Custom backtest (select option 1)
# Compare strategies (select option 3)
# Multi-asset portfolio (select option 2)
```

**Menu Options:**
- `1` → Single Asset (customizable)
- `2` → Multi-Asset Portfolio  
- `3` → Strategy Comparison
- `4` → Quick Demo (fastest)
- `5` → Performance Report
- `6` → Exit

---

## 🎉 You're Ready!

You now have a professional-grade backtesting framework at your fingertips. Happy testing! 🚀

**Remember:** Past performance does not guarantee future results. Use this for research and education, not as financial advice. 💎
