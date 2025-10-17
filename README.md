# 📊 MTC Trading v1.0 — Trading Research Framwork


## 🧠 Overview

**MTC Trading v1.0** is a full-featured backtesting engine built to model, test, and analyze systematic trading strategies under realistic market conditions. Using real, up-to-date market data on any listed equity, allowing users to evaluate performance across custom historical periods or the most recent market environment. 

Supporting Stocks, Etfs, Indices, Commodities, Currencies throught live market data from *Yahoo Finance*,  with future integration plans of Crypto.
Users can freely select **custom start and end dates**, or backtest strategies using data **up to the current date**.

The goal of this project was to challenge myself to design and build a **transparent, configurable sandbox** for testing systematic strategies and real-world practicality.

---

## ⚙️ Key Features

### 🔹 Strategies Implemented
Includes several classical and research-backed trading models:
- **Moving Average Crossover** – trend-following logic with configurable short/long windows.  
- **Momentum Strategy** – trades based on directional price persistence.  
- **Mean Reversion (Bollinger Bands)** – volatility-based contrarian model.
- **RSI Strategy** – oscillator-driven entry and exit signals.

Each strategy supports configurable parameters and realistic execution timing (next-bar vs same-bar).

### 🔹 Built-In Risk Parameters
One of the Main goals of MTC Trading v1.0 was to design a realistic framwork were it simulates how a disciplined, risk-aware trader would actually manage capital in real markets. MTC Trading v1.0 comes with realistic, industry-aligned risk parameters built in by default. These act as the foundation for position sizing, trade exits, and portfolio exposure, helping keep results meaningful and practical.
- **`max_position_size` (20%)** - Limits any single trade to ensure no single idea dominates risk.
- **`stop_loss_pct ` (5%)** - Automatically exits a position (built-in capital preservation rule).
- **`take_profit_pct ` (15%)** - Locks in gains, maintaining a balanced 3:1 reward-to-risk ratio.
- **`max_portfolio_risk ` (2%)** - Caps total risk exposure per trade, enforcing institutional-style discipline.
- **`trailing_stop_pct ` (10%)** - Adjusts dynamically as price moves in your favor, protecting profits.

---

### 🔹 Risk Management System
The framework includes a dedicated **`RiskManager`** module designed to replicate institutional-style capital management. It controls trade sizing, exit logic, and portfolio exposure, ensuring that every strategy operates under realistic, disciplined risk conditions.

Core Functions:
- **Kelly-inspired sizing** - trade sizes adjust dynamically based on portfolio risk and market volatility. This draws on the Kelly Criterion principle: allocating more when conditions are favorable and less when risk increases, balancing long-term growth and drawdown control.
- **Stop Loss**, **Take Profit**, and **Trailing Stop** - execution logic  
- **Maximum Portfolio Exposure** and **Per-Trade Risk** - caping total open risk to prevent over-leverage and preserve capital during volatile markets. 
- **Detailed event logging** - tracks all risk triggers and exits, providing full transparency for analysis and debugging.

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
