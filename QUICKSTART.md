# 🚀 Quick Start Guide

Get MTC Trading v1.0 running in under 5 minutes.

---

## ⚡ Prerequisites

Before you begin, make sure you have:
- **Python 3.8 or higher** installed ([Download here](https://www.python.org/downloads/))
- **pip** (Python package manager - comes with Python)
- **Git** (optional, for cloning)

**Check your Python version:**
```bash
python --version
# or
python3 --version
```

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

**If you get a "pip not found" error:**
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

When the menu appears, select option **4** (Quick Backtest):
```
👉 Select option (1-6): 4
```

**That's it!** The system will:
1. Download SPY historical data (2022-2024)
2. Run a Moving Average strategy
3. Display performance metrics
4. Generate a performance chart

**Expected output:**
```
✅ Successfully loaded 756 trading days
📊 Running backtest...
✅ Backtest completed

==================================================
PERFORMANCE METRICS - SPY
==================================================
Total Return (%)............................ +42.30
Sharpe Ratio................................   1.18
Max Drawdown (%)............................ -12.40
...
```

---

## 📊 Running a Custom Backtest

### Example 1: Different Stock
```python
# In the menu, select option 1
👉 Select option (1-6): 1

# Then enter:
Enter ticker: AAPL
Start date: 2020-01-01
End date: 2024-12-31
Initial capital: 50000
Strategy (1-4): 1  # Moving Average
```

### Example 2: Different Strategy
```python
# Select RSI strategy instead
Strategy (1-4): 4
RSI period: 14
Oversold level: 30
Overbought level: 70
```

---

## 🐛 Troubleshooting

### "No module named 'pandas'" (or other package)
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### "No data returned for [ticker]"
**Solution:** 
- Check that the ticker symbol is correct (use Yahoo Finance format)
- Verify you have an internet connection
- Try a different date range

### Charts not displaying
**Solution:** Make sure matplotlib is installed correctly
```bash
pip install matplotlib --upgrade
```

### "Permission denied" error
**Solution:** Run with proper permissions
```bash
# Mac/Linux
python3 "Best Version.py"

# Windows - run terminal as Administrator
python "Best Version.py"
```

---

## 📚 What's Next?

Now that you're up and running:

1. **[View Examples](EXAMPLES.md)** - Learn advanced usage patterns
2. **[Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md)** - Understand the math
3. **[Main README](README.md)** - Full project documentation

---

## 💬 Need Help?

- **Issues:** [Open an issue](https://github.com/yourusername/Trading-Research-Framework/issues)
- **Questions:** Check [EXAMPLES.md](EXAMPLES.md) for common use cases

---

## ⚡ Quick Reference

| Command | Purpose |
|---------|---------|
| `python "Best Version.py"` | Launch the program |
| Option 1 | Single asset trading (custom parameters) |
| Option 2 | Multi-asset portfolio |
| Option 3 | Compare all strategies |
| Option 4 | Quick demo backtest (SPY) |
| Option 5 | Detailed performance report |
| Option 6 | Exit |
