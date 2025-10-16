# 📊 MTC Trading v1.0 — Trading Research Framwork
 
**MTC Trading v1.0** is a full-featured backtesting engine designed to model, test, and analyze trading strategies under realistic market conditions.

---

## 🧠 Overview

**MTC Trading v1.0** is not a simple trading bot — it is a **comprehensive quantitative research system** built in Python that integrates strategy simulation, multi-asset portfolio management, and institutional-style risk control.

The framework allows for:
- Single-asset and multi-asset backtesting
- Multiple trading strategies (momentum, mean reversion, RSI, MA crossovers)
- Advanced risk management (position sizing, stop-loss, take-profit, trailing stop)
- Dynamic portfolio optimization and asset allocation
- Performance analytics and visualization

The goal of this project was to design a **transparent, configurable, and research-oriented environment** for testing systematic strategies with academic rigor and real-world practicality.

---

## ⚙️ Key Features

### 🔹 Strategy Engine
Implements multiple quantitative trading strategies:
- **Moving Average Crossover** – classical trend-following setup using short and long MAs.
- **Momentum Strategy** – directional trading based on recent return persistence.
- **Mean Reversion (Bollinger Bands)** – volatility-based contrarian system.
- **RSI Strategy** – oscillator-driven approach capturing overbought/oversold conditions.

Each strategy supports configurable parameters and realistic execution timing (next-bar vs same-bar).

---

### 🔹 Risk Management System
A dedicated **`RiskManager`** module models position sizing and trade protection using institutional principles:
- **Kelly-inspired sizing** based on volatility and portfolio risk limits  
- **Stop Loss**, **Take Profit**, and **Trailing Stop** execution logic  
- **Maximum portfolio exposure** and per-trade risk budgeting  
- **Detailed event logging** for triggered risk events  

This ensures all strategies operate under controlled, defensible risk conditions.

---

### 🔹 Portfolio Management & Optimization
The **`PortfolioManager`** provides advanced multi-asset functionality:
- Equal-weight, Risk-Parity, Momentum-Weighted, and Market-Cap allocations  
- Mean-Variance Optimization (Markowitz framework)  
- Correlation matrix and inter-asset risk diagnostics  

Enables both diversified portfolio construction and performance attribution studies.

---

### 🔹 Backtesting Engine
The **`TradingBot`** executes historical simulations with:
- Configurable **commission and slippage models**  
- **Realistic execution modes** to prevent look-ahead bias  
- Complete portfolio accounting (cash, holdings, total value)  
- **Buy & Hold benchmarks** for direct performance comparison  
- Full trade logging with entry/exit dates, reason codes, and duration tracking  

---

### 🔹 Performance Analytics
Comprehensive post-simulation metrics include:
- Total, annualized, and benchmark returns  
- **Sharpe, Sortino, and Calmar ratios**  
- **Maximum drawdown** and volatility statistics  
- **Win rate**, streaks, and trade-level summaries  
- Optional graphical analysis:
  - Equity curve vs Buy & Hold  
  - Signal overlay on price chart  
  - Drawdown visualization  
  - Rolling Sharpe ratio dynamics  

---

## 🧩 Technical Logic (Detailed)

### 1️⃣ Execution Architecture
- **BacktestConfig** governs execution behavior (commission, slippage, and look-ahead mode).  
- Each strategy produces a **signal DataFrame**.  
- Depending on mode:
  - *Realistic*: signals execute on the next bar (avoiding look-ahead bias).  
  - *Optimistic*: executes on the same bar for theoretical comparison.

### 2️⃣ Position Management
- Position size is determined using portfolio value, volatility, and max-risk constraints.  
- When an entry signal occurs:
  - Shares are sized using the `calculate_position_size()` method.  
  - The bot verifies capital sufficiency and deducts commission/slippage.  
- When exit criteria (signal reversal or stop event) occur, al
