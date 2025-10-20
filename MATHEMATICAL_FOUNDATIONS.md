# 📊 MTC Trading v1.0 — Mathematical Foundations

# 📐 Mathematical Foundations

This section summarizes the key mathematical principles and formulas that power **MTC Trading v1.0**.  
The framework integrates concepts from quantitative finance, portfolio theory, and risk management to simulate realistic trading behavior and performance analytics.

---

## 📚 Table of Contents

1. [Core Concepts](#i-core-concepts)
2. [Risk Metrics](#ii-risk-metrics)
3. [Position Sizing & Risk Management](#iii-position-sizing--risk-management)
4. [Technical Indicators & Strategy Logic](#iv-technical-indicators--strategy-logic)
5. [Portfolio Theory](#v-portfolio-theory)
6. [Execution Model](#vi-execution-model)
7. [Assumptions & Limitations](#vii-assumptions--limitations)
8. [References](#viii-references)

---
 
## I. Core Concepts

### 🔹 Returns Calculation
**Daily Return:**

For each time period $t$, the daily return is calculated as:

$$r_t = \frac{V_t - V_{t-1}}{V_{t-1}}$$

Where:
- $V_t$ = Portfolio value at time $t$
- $r_t$ = Return on day $t$

The sequence $[r_1, r_2, \dots, r_T]$ forms the **time series of returns**, which is the foundation for all performance analytics.

**Implementation:**
```python
# See TradingBot.backtest() method
portfolio.loc[date, 'returns'] = (portfolio.loc[date, 'total'] / prev_total) - 1
```

### 🔹 Cumulative and Annualized Returns
**Total Return:**

$$R_{\text{total}} = \frac{V_{\text{final}} - V_{\text{initial}}}{V_{\text{initial}}} = \frac{V_{\text{final}}}{V_{\text{initial}}} - 1$$

**Cumulative Compounded Return:**

$$R_{\text{cumulative}} = \prod_{t=1}^{T}(1 + r_t) - 1$$

**Annualized Return:**

$$R_{\text{annualized}} = \left(1 + R_{\text{total}}\right)^{\frac{252}{N}} - 1$$

Where:
- $N$ = Number of trading days in the backtest
- 252 = Average number of trading days per year in U.S. markets

**Why 252 days?** U.S. stock markets are typically open ~252 days per year (365 days - weekends - holidays).

---


### 🔹 Portfolio Accounting
At any time $t$, the portfolio value is:

$$V_t = C_t + \sum_{i=1}^{n} P_{i,t} \times Q_{i,t}$$

Where:
- $C_t$ = Cash balance at time $t$
- $P_{i,t}$ = Price of asset $i$ at time $t$
- $Q_{i,t}$ = Quantity (shares) of asset $i$ held at time $t$
- $n$ = Number of assets in portfolio

**Implementation:**
```python
# See TradingBot.backtest() method
portfolio.loc[date, 'total'] = cash + (position * price)
```
---

## II. Risk Metrics

### 🔹 Volatility (Standard Deviation)

**Sample Standard Deviation:**

$$\sigma = \sqrt{\frac{1}{N - 1}\sum_{t=1}^{N}(r_t - \bar{r})^2}$$

Where:
- $\bar{r}$ = Mean return = $\frac{1}{N}\sum_{t=1}^{N}r_t$
- $N$ = Number of observations

**Annualized Volatility:**

$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$

**Why $\sqrt{252}$?** Variance scales linearly with time, so standard deviation scales with the square root of time.

**Implementation:**
```python
# See TradingBot.calculate_metrics()
volatility = returns.std() * np.sqrt(252) * 100
```
---
### 🔹 Sharpe Ratio
**Definition:** Risk-adjusted return metric that measures excess return per unit of risk.

$$\text{Sharpe Ratio} = \frac{\bar{r} - r_f}{\sigma} \times \sqrt{252}$$

Where:
- $\bar{r}$ = Average daily return
- $r_f$ = Risk-free rate (typically assumed to be 0 for simplicity)
- $\sigma$ = Standard deviation of daily returns
- $\sqrt{252}$ = Annualization factor

**Interpretation:**
- **< 1.0** = Poor risk-adjusted performance
- **1.0 - 2.0** = Good performance
- **> 2.0** = Excellent performance
- **> 3.0** = Exceptional (rare without overfitting)

**Implementation:**
```python
# See TradingBot.calculate_metrics()
sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
```
---

### 🔹 Sortino Ratio
\[
\text{Sortino} = \frac{\bar{r} - r_f}{\sigma_{\text{downside}}}
\]
where \(\sigma_{\text{downside}}\) is the standard deviation of returns below zero.

### 🔹 Calmar Ratio
\[
\text{Calmar} = \frac{R_{\text{annualized}}}{|\text{Max Drawdown}|}
\]

### 🔹 Maximum Drawdown
\[
\text{MDD} = \max\left(\frac{P_{\text{peak}} - P_{\text{trough}}}{P_{\text{peak}}}\right)
\]

---

## III. Position Sizing

### 🔹 Risk-per-Trade
\[
\text{Risk per trade} = \text{entry price} \times \text{stop\_loss\_pct}
\]
\[
\text{Position size} = \frac{\text{Portfolio value} \times \text{max\_portfolio\_risk}}{\text{Risk per trade}}
\]

### 🔹 Kelly Criterion (Conceptual)
\[
f^* = \frac{p(b + 1) - 1}{b}
\]
Used conceptually to guide proportional exposure and position scaling.

### 🔹 Volatility-Based Sizing
Positions are inversely scaled to volatility to maintain consistent portfolio risk:
\[
w_i = \frac{1 / \sigma_i}{\sum_{j=1}^{n} (1 / \sigma_j)}
\]

---

## IV. Strategy Logic

### 🔹 Moving Average Crossover
\[
\text{Signal} =
\begin{cases}
1, & \text{if } \text{MA}_{\text{short}} > \text{MA}_{\text{long}} \\
-1, & \text{if } \text{MA}_{\text{short}} < \text{MA}_{\text{long}}
\end{cases}
\]

### 🔹 RSI (Relative Strength Index)
\[
\text{RSI} = 100 - \frac{100}{1 + \frac{\text{Avg Gain}}{\text{Avg Loss}}}
\]
Typical thresholds: **Buy below 30**, **Sell above 70**.

### 🔹 Bollinger Bands
\[
\text{Upper Band} = \text{MA}_n + k\sigma
\]
\[
\text{Lower Band} = \text{MA}_n - k\sigma
\]

### 🔹 Momentum Indicator
\[
\text{Signal} =
\begin{cases}
1, & P_t > P_{t-k} \\
-1, & P_t < P_{t-k}
\end{cases}
\]

---

## V. Portfolio Theory

### 🔹 Mean–Variance Optimization
\[
E[R_p] = \sum_{i=1}^{n} w_i E[R_i]
\]
\[
\sigma_p^2 = w^T \Sigma w
\]
where \( w \) is the weight vector and \( \Sigma \) is the covariance matrix of asset returns.

### 🔹 Correlation Matrix
\[
\rho_{ij} = \frac{\text{Cov}(R_i, R_j)}{\sigma_i \sigma_j}
\]

### 🔹 Risk Parity Allocation
Balances risk contribution across assets:
\[
RC_i = w_i \times (\Sigma w)_i
\]
Weights are adjusted so that each \( RC_i \) contributes equally to portfolio variance.

---

## VI. Practical Considerations

### 🔹 Transaction Costs
Applied as:
\[
\text{Cost} = \text{Trade Value} \times \text{Commission Rate}
\]

### 🔹 Slippage Modeling
\[
\text{Executed Price} = \text{Market Price} \times (1 \pm \text{Slippage Rate})
\]

### 🔹 Look-Ahead Bias Prevention
Ensured by executing signals on the **next bar** when in `REALISTIC` mode:
\[
\text{Trade}_{t+1} = \text{Signal at } t
\]

---

📘 *These formulas and concepts collectively form the quantitative foundation of MTC Trading v1.0, ensuring that every backtest reflects both mathematical rigor and real-market behavior.*
