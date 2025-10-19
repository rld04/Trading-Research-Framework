# 📊 MTC Trading v1.0 — Trading Research Framwork

MTC Trading v1.0 is a production-grade backtesting engine that lets you test systematic trading strategies against real market data with institutional-level risk management built in.

## 🧠 Overview

**MTC Trading v1.0** is a full-featured backtesting engine built to model, test, and analyze systematic trading strategies under realistic market conditions. Using real, up-to-date market data on any listed equity, allowing users to evaluate performance across custom historical periods or the most recent market environment. 

Supporting Stocks, Etfs, Indices, Commodities, Currencies throught live market data from *Yahoo Finance*,  with future integration plans of Crypto.
Users can freely select **custom start and end dates**, or backtest strategies using data **up to the current date**.

The goal of this project was to challenge myself to design and build a **transparent, configurable sandbox** for testing systematic strategies and real-world practicality.

---

## ⚙️ Key Features

### 🔹 Strategies Implemented
Includes several classical and research-backed trading models:
- **Moving Average Crossover** – Trend-following logic with configurable short/long windows.  
- **Momentum Strategy** – Trades based on directional price persistence.  
- **Mean Reversion (Bollinger Bands)** – Volatility-based contrarian model.
- **RSI Strategy** – Oscillator-driven entry and exit signals.

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
- **Kelly-inspired sizing** - Dynamically adjusts trade sizes based on portfolio risk and market volatility, balancing potential growth with drawdown control.
- **Stop Loss**, **Take Profit**, and **Trailing Stop** - Execution logic ensures positions are automatically managed to minimize losses and lock in gains.  
- **Maximum Portfolio Exposure** and **Per-Trade Risk** - Caps total open risk to prevent over-leverage and safeguard capital during volatile markets.
- **Detailed event logging** - Tracks all risk triggers and exits, providing full transparency for analysis, debugging, and strategy optimization.

This ensures all strategies operate under controlled, defensible risk conditions.

---

### 🔹 Portfolio Management & Optimization
The **`PortfolioManager`** The PortfolioManager module expands the framework beyond single-asset strategies, enabling multi-asset portfolio simulation and optimization. It integrates several allocation methodologies and diagnostic tools that mirror techniques used in institutional portfolio research.

Core Capabilities:
- **Allocation Models**: Supports multiple weighting schemes, including Equal-Weight, Risk-Parity, Momentum-Weighted, and Market-Cap Weighted portfolios.
- **Optimization Framework**: Implements Mean–Variance Optimization (Markowitz model) to explore the trade-off between expected return and portfolio volatility.
- **Correlation & Risk Analysis**: Computes inter-asset correlations and risk contribution diagnostics, helping evaluate diversification effectiveness and asset dependencies. 

By combining these features, MTC Trading v1.0 allows users to construct, test, and analyze portfolios the same way quantitative research teams evaluate strategy combinations enabling the user to balance risk, return, and diversification in a controlled, data-driven manner.

---

### 🔹 Backtesting Engine
The **`TradingBot`** module serves as the core of the framework, executing historical simulations that replicate real-world trading conditions as closely as possible.
It handles trade execution, accounting, and performance tracking across any chosen date range using live market data.

Core Capabilities:
- **Configurable Transaction Costs**: Supports realistic commission and slippage models to simulate trading friction.
- **Execution Modes**: Offers Realistic and Optimistic settings to control for look-ahead bias, ensuring fair backtest results.  
- **Portfolio Accounting**: Tracks cash balances, holdings, and total portfolio value throughout the simulation.
- **Benchmark Comparison**: Automatically computes a Buy & Hold reference for direct performance evaluation.  
- **Comprehensive Trade Logging**: Records every trade with entry and exit dates, reason codes, and holding duration for full transparency.

By combining flexibility with data integrity, MTC Trading v1.0 delivers backtests that are both statistically robust and operationally realistic, bridging the gap between academic research and real-world trading execution.

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
