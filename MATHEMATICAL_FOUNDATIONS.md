# 📐 Mathematical Foundations of MTC Trading v1.0

> **Comprehensive documentation of the quantitative methods, formulas, and statistical foundations underlying the MTC Trading backtesting framework.**

This document explains the mathematical principles that power MTC Trading v1.0, integrating concepts from quantitative finance, portfolio theory, and risk management to simulate realistic trading behavior and performance analytics.

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

---

### 🔹 Cumulative Returns

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

**Definition:** Similar to Sharpe, but only penalizes downside volatility.

$$\text{Sortino Ratio} = \frac{\bar{r} - r_f}{\sigma_{\text{downside}}} \times \sqrt{252}$$

Where:

$$\sigma_{\text{downside}} = \sqrt{\frac{1}{N_{\text{down}}}\sum_{r_t < 0}(r_t)^2}$$

**Why use Sortino?** Upside volatility is desirable, so Sortino only penalizes negative returns. This gives a more accurate picture of risk for asymmetric return distributions.

**Implementation:**
```python
# See TradingBot.calculate_metrics()
downside_returns = returns[returns < 0]
downside_std = downside_returns.std()
sortino_ratio = np.sqrt(252) * returns.mean() / downside_std
```

---

### 🔹 Calmar Ratio

**Definition:** Ratio of annualized return to maximum drawdown.

$$\text{Calmar Ratio} = \frac{R_{\text{annualized}}}{|\text{Max Drawdown}|}$$

**Interpretation:**
- **> 3.0** = Excellent
- **1.0 - 3.0** = Good
- **< 1.0** = Poor (returns don't justify the risk)

Higher Calmar ratios indicate better risk-adjusted returns relative to worst-case losses.

---

### 🔹 Maximum Drawdown (MDD)

**Definition:** Largest peak-to-trough decline in portfolio value.

$$\text{MDD} = \max_{t \in [0,T]} \left(\frac{P_{\text{peak}}(t) - P_{\text{trough}}(t)}{P_{\text{peak}}(t)}\right)$$

**Calculation Process:**
1. Track running maximum (peak): $P_{\text{peak}}(t) = \max_{s \leq t} V_s$
2. Calculate drawdown at each point: $DD_t = \frac{V_t - P_{\text{peak}}(t)}{P_{\text{peak}}(t)}$
3. Maximum drawdown: $\text{MDD} = \min_t DD_t$

**Implementation:**
```python
# See TradingBot.calculate_metrics()
cumulative = (1 + returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min() * 100
```

**Interpretation:**
- **< 10%** = Excellent capital preservation
- **10-20%** = Acceptable for most strategies
- **20-30%** = High risk
- **> 30%** = Very high risk; may indicate poor risk management

---

## III. Position Sizing

### 🔹 Risk-per-Trade
**Risk Amount Per Trade:**

$$\text{Risk}_{\text{trade}} = \text{Entry Price} \times \text{Stop Loss %}$$

**Position Size (Shares):**

$$Q = \left\lfloor \frac{V_{\text{portfolio}} \times \text{Max Portfolio Risk %}}{\text{Risk}_{\text{trade}}} \right\rfloor$$

Where:
- $Q$ = Number of shares to buy (integer)
- $V_{\text{portfolio}}$ = Current portfolio value
- Max Portfolio Risk % = Maximum percentage of portfolio to risk (default: 2%)
- $\lfloor \cdot \rfloor$ = Floor function (round down to integer)

**Example:**
- Portfolio value: $100,000
- Max portfolio risk: 2% = $2,000
- Entry price: $100
- Stop loss: 5% = $5 per share
- Position size: $2,000 / $5 = **400 shares**

**Implementation:**
```python
# See RiskManager.calculate_position_size()
risk_dollars = portfolio_value * self.max_portfolio_risk
stop_amount = self.stop_loss_pct * price
shares_by_risk = int(risk_dollars / stop_amount)
```

---
### 🔹 Kelly Criterion (Conceptual)
**Formula:**

$$f^* = \frac{p(b + 1) - 1}{b} = \frac{p \cdot b - q}{b}$$

Where:
- $f^*$ = Optimal fraction of capital to risk
- $p$ = Win probability
- $q$ = Loss probability = $1 - p$
- $b$ = Win/loss ratio (average win ÷ average loss)

**Practical Application:**

The framework uses a **modified Kelly approach** by:
1. Capping position sizes at 20% of portfolio (prevents over-leverage)
2. Using volatility as a proxy for uncertainty
3. Applying both dollar-based and risk-based constraints

**Why not pure Kelly?** Pure Kelly can be too aggressive and lead to excessive drawdowns. Professional traders typically use **fractional Kelly** (e.g., 25-50% of Kelly optimal) for more conservative sizing.

---
### 🔹 Position Size Constraints

The system applies **dual constraints** to ensure safe position sizing:

**Constraint 1: Maximum Position Size**

$$Q_{\text{max,dollars}} = \left\lfloor \frac{V_{\text{portfolio}} \times \text{Max Position %}}{P_{\text{entry}}} \right\rfloor$$

Default: 20% of portfolio per position

**Constraint 2: Risk-Based Sizing**

$$Q_{\text{max,risk}} = \left\lfloor \frac{V_{\text{portfolio}} \times \text{Max Risk %}}{P_{\text{entry}} \times \text{Stop Loss %}} \right\rfloor$$

Default: 2% portfolio risk per trade

**Final Position Size:**

$$Q_{\text{final}} = \min(Q_{\text{max,dollars}}, Q_{\text{max,risk}})$$

This ensures positions are limited by BOTH absolute size AND potential loss.

---
### 🔹 Stop Loss Logic

**Trigger Condition:**

$$\text{Stop Loss Triggered} = \begin{cases} 
\text{True}, & \text{if } \frac{P_{\text{current}} - P_{\text{entry}}}{P_{\text{entry}}} \leq -\text{Stop Loss %} \\
\text{False}, & \text{otherwise}
\end{cases}$$

Default: 5% stop loss

**Implementation:**
```python
# See RiskManager.check_stop_loss()
loss_pct = (current_price - entry_price) / entry_price
triggered = loss_pct <= -self.stop_loss_pct
```

---
### 🔹 Take Profit Logic

**Trigger Condition:**

$$\text{Take Profit Triggered} = \begin{cases} 
\text{True}, & \text{if } \frac{P_{\text{current}} - P_{\text{entry}}}{P_{\text{entry}}} \geq \text{Take Profit %} \\
\text{False}, & \text{otherwise}
\end{cases}$$

Default: 15% take profit (3:1 reward-to-risk ratio)

---
### 🔹 Trailing Stop Logic

**Peak Tracking:**

$$P_{\text{peak}}(t) = \max(P_{\text{peak}}(t-1), P_{\text{current}}(t))$$

**Trailing Stop Price:**

$$P_{\text{stop}}(t) = P_{\text{peak}}(t) \times (1 - \text{Trailing Stop %})$$

**Trigger Condition:**

$$\text{Trailing Stop Triggered} = P_{\text{current}} \leq P_{\text{stop}}$$

Default: 10% trailing stop

**Example:**
- Entry: $100
- Peak: $120 (position up 20%)
- Trailing stop at 10%: $120 × 0.90 = **$108**
- Triggered if price falls to $108, locking in ~8% gain

---
## IV. Strategy Logic

### 🔹 Moving Average (MA)

**Simple Moving Average (SMA):**

$$\text{MA}_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}$$

Where:
- $n$ = Window size (e.g., 50 days, 200 days)
- $P_t$ = Price at time $t$

**Crossover Signal:**

$$\text{Signal}(t) = \begin{cases}
+1 \text{ (Buy)}, & \text{if } \text{MA}_{\text{short}}(t) > \text{MA}_{\text{long}}(t) \\
-1 \text{ (Sell)}, & \text{if } \text{MA}_{\text{short}}(t) < \text{MA}_{\text{long}}(t) \\
0 \text{ (Hold)}, & \text{otherwise}
\end{cases}$$

**Common Configurations:**
- **Golden Cross**: MA(50) crosses above MA(200) → Bullish signal
- **Death Cross**: MA(50) crosses below MA(200) → Bearish signal

**Implementation:**
```python
# See TradingBot.moving_average_crossover()
signals['short_ma'] = data['Close'].rolling(window=short_window).mean()
signals['long_ma'] = data['Close'].rolling(window=long_window).mean()
signals['signal'] = np.where(signals['short_ma'] > signals['long_ma'], 1.0, 0.0)
```

### 🔹 Relative Strength Index (RSI)

**Step 1: Calculate Price Changes**

$$\Delta P_t = P_t - P_{t-1}$$

**Step 2: Separate Gains and Losses**

$$\text{Gain}_t = \begin{cases} \Delta P_t, & \text{if } \Delta P_t > 0 \\ 0, & \text{otherwise} \end{cases}$$

$$\text{Loss}_t = \begin{cases} |\Delta P_t|, & \text{if } \Delta P_t < 0 \\ 0, & \text{otherwise} \end{cases}$$

**Step 3: Calculate Average Gain and Loss**

$$\text{Avg Gain} = \frac{1}{n}\sum_{i=1}^{n}\text{Gain}_i$$

$$\text{Avg Loss} = \frac{1}{n}\sum_{i=1}^{n}\text{Loss}_i$$

**Step 4: Calculate RS and RSI**

$$\text{RS} = \frac{\text{Avg Gain}}{\text{Avg Loss}}$$

$$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$

**Trading Signals:**

$$\text{Signal} = \begin{cases}
+1 \text{ (Buy)}, & \text{if RSI} < 30 \text{ (Oversold)} \\
-1 \text{ (Sell)}, & \text{if RSI} > 70 \text{ (Overbought)} \\
0 \text{ (Hold)}, & \text{otherwise}
\end{cases}$$

**Interpretation:**
- **RSI < 30**: Asset is oversold (potential buy opportunity)
- **RSI > 70**: Asset is overbought (potential sell opportunity)
- **RSI = 50**: Neutral (no directional bias)

**Implementation:**
```python
# See TradingBot.rsi_strategy()
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```
---
### 🔹 Bollinger Bands

**Middle Band (SMA):**

$$\text{BB}_{\text{middle}} = \text{MA}_n$$

**Upper Band:**

$$\text{BB}_{\text{upper}} = \text{MA}_n + k \cdot \sigma_n$$

**Lower Band:**

$$\text{BB}_{\text{lower}} = \text{MA}_n - k \cdot \sigma_n$$

Where:
- $\text{MA}_n$ = n-period moving average
- $\sigma_n$ = n-period standard deviation of price
- $k$ = Number of standard deviations (typically 2)

**Standard Deviation Calculation:**

$$\sigma_n = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1}(P_{t-i} - \text{MA}_n)^2}$$

**Mean Reversion Signals:**

$$\text{Signal} = \begin{cases}
+1 \text{ (Buy)}, & \text{if } P_t < \text{BB}_{\text{lower}} \\
-1 \text{ (Sell)}, & \text{if } P_t > \text{BB}_{\text{upper}} \\
0 \text{ (Hold)}, & \text{otherwise}
\end{cases}$$

**Interpretation:**
- Price touching lower band → Potential reversal up
- Price touching upper band → Potential reversal down
- Bands widening → Increased volatility
- Bands narrowing → Decreased volatility (potential breakout coming)

**Implementation:**
```python
# See TradingBot.mean_reversion_strategy()
signals['ma'] = data['Close'].rolling(window=window).mean()
signals['std'] = data['Close'].rolling(window=window).std()
signals['upper_band'] = signals['ma'] + (num_std * signals['std'])
signals['lower_band'] = signals['ma'] - (num_std * signals['std'])
```

---
### 🔹 Momentum Indicator

**Price Rate of Change:**

$$\text{Momentum}(t) = \frac{P_t - P_{t-k}}{P_{t-k}} = \frac{P_t}{P_{t-k}} - 1$$

Where:
- $k$ = Lookback period (e.g., 20 days)

**Trading Signals:**

$$\text{Signal} = \begin{cases}
+1 \text{ (Buy)}, & \text{if Momentum} > \theta_{\text{buy}} \\
-1 \text{ (Sell)}, & \text{if Momentum} < \theta_{\text{sell}} \\
0 \text{ (Hold)}, & \text{otherwise}
\end{cases}$$

Where:
- $\theta_{\text{buy}}$ = Positive threshold (e.g., +5%)
- $\theta_{\text{sell}}$ = Negative threshold (e.g., -5%)

**Implementation:**
```python
# See TradingBot.momentum_strategy()
signals['momentum'] = data['Close'].pct_change(periods=lookback_period)
signals['signal'] = np.where(signals['momentum'] > threshold, 1.0, 0.0)
```

---
## V. Portfolio Theory
### 🔹 Expected Portfolio Return

**Weighted Average of Asset Returns:**

$$E[R_p] = \sum_{i=1}^{n} w_i \cdot E[R_i]$$

Where:
- $w_i$ = Weight of asset $i$ in portfolio ($\sum w_i = 1$)
- $E[R_i]$ = Expected return of asset $i$
- $n$ = Number of assets

---
### 🔹 Portfolio Variance

**Variance Formula:**

$$\sigma_p^2 = \sum_{i=1}^{n}\sum_{j=1}^{n} w_i w_j \sigma_i \sigma_j \rho_{ij}$$

**Matrix Notation:**

$$\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$$

Where:
- $\mathbf{w}$ = Column vector of weights $[w_1, w_2, \dots, w_n]^T$
- $\Sigma$ = Covariance matrix
- $\rho_{ij}$ = Correlation between assets $i$ and $j$

**Portfolio Standard Deviation:**

$$\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$$

---
### 🔹 Covariance Matrix

**Covariance Calculation:**

$$\text{Cov}(R_i, R_j) = \frac{1}{T-1}\sum_{t=1}^{T}(R_{i,t} - \bar{R}_i)(R_{j,t} - \bar{R}_j)$$

**Matrix Form:**

$$\Sigma = \begin{bmatrix}
\sigma_1^2 & \text{Cov}(R_1,R_2) & \cdots & \text{Cov}(R_1,R_n) \\
\text{Cov}(R_2,R_1) & \sigma_2^2 & \cdots & \text{Cov}(R_2,R_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(R_n,R_1) & \text{Cov}(R_n,R_2) & \cdots & \sigma_n^2
\end{bmatrix}$$

**Implementation:**
```python
# See PortfolioManager.calculate_correlation_matrix()
returns_df = pd.DataFrame(returns_dict).dropna()
correlation_matrix = returns_df.corr()
covariance_matrix = returns_df.cov()
```

---
### 🔹 Correlation Coefficient

**Definition:**

$$\rho_{ij} = \frac{\text{Cov}(R_i, R_j)}{\sigma_i \cdot \sigma_j}$$

**Properties:**
- Range: $-1 \leq \rho_{ij} \leq 1$
- $\rho_{ij} = 1$: Perfect positive correlation
- $\rho_{ij} = 0$: No linear correlation
- $\rho_{ij} = -1$: Perfect negative correlation

**Portfolio Diversification:**
- Lower correlations → Better diversification
- Target: $\rho < 0.7$ between major holdings

---
## 🔹 Mean–Variance Optimization (Markowitz)

**Objective:** Maximize expected return for a given level of risk.

**Optimization Problem:**

$$\max_{\mathbf{w}} \left[ \mathbf{w}^T \mathbf{\mu} - \frac{\lambda}{2} \mathbf{w}^T \Sigma \mathbf{w} \right]$$

Subject to:
$$\sum_{i=1}^{n} w_i = 1 \quad \text{(weights sum to 1)}$$
$$w_i \geq 0 \quad \text{(no short selling)}$$

Where:
- $\mathbf{\mu}$ = Vector of expected returns
- $\lambda$ = Risk aversion parameter
- Higher $\lambda$ → More risk-averse (prefer lower variance)

**Simplified Implementation (Risk-Adjusted Scoring):**

$$\text{Score}_i = \frac{E[R_i]}{\sigma_i}$$

$$w_i = \frac{\text{Score}_i}{\sum_{j=1}^{n}\text{Score}_j}$$

**Implementation:**
```python
# See PortfolioManager.optimize_portfolio_weights()
mean_returns = returns_df.mean() * 252  # Annualized
volatilities = np.sqrt(np.diag(cov_matrix))
scores = mean_returns.values / volatilities  # Sharpe-like ratio
weights = scores / scores.sum()
```

---
### 🔹 Risk Parity Allocation

**Concept:** Allocate capital so each asset contributes equally to portfolio risk.

**Risk Contribution:**

$$RC_i = w_i \cdot \frac{\partial \sigma_p}{\partial w_i} = w_i \cdot \frac{(\Sigma \mathbf{w})_i}{\sigma_p}$$

**Equal Risk Contribution:**

$$RC_1 = RC_2 = \cdots = RC_n = \frac{\sigma_p^2}{n}$$

**Simplified Inverse Volatility Weighting:**

$$w_i = \frac{1/\sigma_i}{\sum_{j=1}^{n}(1/\sigma_j)}$$

This approximation works well when correlations are similar across assets.

**Implementation:**
```python
# See PortfolioManager.calculate_allocations() - risk_parity method
volatilities = {ticker: returns.std() for ticker, returns in returns_dict.items()}
inv_vols = {t: 1/v for t, v in volatilities.items()}
total = sum(inv_vols.values())
allocations = {t: iv/total for t, iv in inv_vols.items()}
```

---
## VI. Execution Model
### 🔹 Transaction Costs

**Total Execution Cost:**

$$\text{Cost}_{\text{total}} = \text{Trade Value} \times (\text{Commission} + \text{Slippage})$$

**Buy Transaction:**

$$\text{Cash Out} = Q \times P_{\text{entry}} \times (1 + c + s)$$

**Sell Transaction:**

$$\text{Cash In} = Q \times P_{\text{exit}} \times (1 - c - s)$$

Where:
- $Q$ = Quantity (shares)
- $P$ = Price per share
- $c$ = Commission rate (default: 0.1% = 0.001)
- $s$ = Slippage rate (default: 0.1% = 0.001)

**Implementation:**
```python
# See TradingBot.backtest()
# Buy
cost = shares * price * (1 + commission + slippage)
cash -= cost

# Sell
proceeds = shares * price * (1 - commission - slippage)
cash += proceeds
```

---
### 🔹 Slippage Modeling
**Definition:** Difference between expected execution price and actual execution price.

**Linear Slippage Model:**

$$P_{\text{executed}} = P_{\text{market}} \times (1 + \epsilon \cdot \text{direction})$$

Where:
- $\epsilon$ = Slippage rate (default: 0.1%)
- direction = +1 for buys (pay more), -1 for sells (receive less)

**Why model slippage?**
- Market orders don't execute at exact quoted prices
- Bid-ask spread causes execution costs
- Large orders can move the market against you

---
### 🔹 Look-Ahead Bias Prevention
**Realistic Mode (Default):**

$$\text{Trade Execution Time} = t + 1$$
$$\text{Signal Generation Time} = t$$

**Signal Shift:**
```python
# Shift signals by 1 period
signals['positions'] = signals['signal'].diff().shift(1).fillna(0)
```

**Why this matters:**
- You can't trade on today's close USING today's close
- Real trading requires time to process signals and place orders
- Prevents artificially inflated backtest results

**Optimistic Mode (Research Only):**

$$\text{Trade Execution Time} = t$$
$$\text{Signal Generation Time} = t$$

Used for theoretical maximum performance comparison only.

---
## VII. Assumptions & Limitations
### 🔹 Market Assumptions

**What We Assume:**
1. **Liquidity:** Infinite liquidity at quoted prices (unrealistic for large orders)
2. **Market Impact:** No market impact from our trades (realistic for small accounts)
3. **Continuous Trading:** Can enter/exit at any daily close (ignores intraday dynamics)
4. **No Gaps:** No overnight gap risk modeling
5. **Perfect Execution:** Orders always fill (ignores partial fills, rejected orders)

**What We DON'T Assume:**
- ❌ Future knowledge (look-ahead bias prevented)
- ❌ Perfect market timing
- ❌ Zero transaction costs (commission and slippage modeled)
- ❌ Normally distributed returns (use robust metrics like Sortino)

---
### 🔹 Statistical Assumptions

**Return Distributions:**
- Many metrics (Sharpe, volatility) assume **normal distribution**
- Real returns exhibit:
  - **Fat tails** (extreme events more common than normal distribution predicts)
  - **Skewness** (asymmetric upside/downside)
  - **Time-varying volatility** (volatility clusters)

**Stationarity:**
- Assumes statistical properties remain constant over time
- Reality: Market regimes change (bull, bear, crisis)

**Independence:**
- Assumes returns are independent (no serial correlation)
- Reality: Some momentum and mean-reversion effects exist

---
### 🔹 Risk Management Limitations

**What's Covered:**
- ✅ Position sizing
- ✅ Stop losses
- ✅ Take profits
- ✅ Portfolio-level risk limits

**What's NOT Covered:**
- ❌ Liquidity risk (assumes you can always exit)
- ❌ Counterparty risk
- ❌ Regulatory changes
- ❌ Black swan events (tail risk beyond normal volatility)
- ❌ Correlation breakdown in crises (correlations → 1 during crashes)

---
### 🔹 Model Risk

**Overfitting Risk:**
- Strategy parameters optimized on historical data may not work going forward
- **Mitigation:** Use walk-forward testing, out-of-sample validation

**Data Quality:**
- Yahoo Finance data may have errors, missing values, or survivorship bias
- **Mitigation:** Data validation checks in code

**Regime Changes:**
- Market structure changes over time (algorithms, regulations, technology)
- **Mitigation:** Regular revalidation, adaptive parameters

---
## VIII. References
### 📚 Academic Papers

1. **Sharpe, W. F. (1994)**  
   *"The Sharpe Ratio"*  
   Journal of Portfolio Management, Fall 1994  
   → Foundational paper on risk-adjusted return measurement

2. **Sortino, F. A. & Price, L. N. (1994)**  
   *"Performance Measurement in a Downside Risk Framework"*  
   Journal of Investing, 3(3), 59-64  
   → Introduced downside deviation as alternative to standard deviation

3. **Markowitz, H. (1952)**  
   *"Portfolio Selection"*  
   The Journal of Finance, 7(1), 77-91  
   → Modern Portfolio Theory foundation

4. **Kelly, J. L. (1956)**  
   *"A New Interpretation of Information Rate"*  
   Bell System Technical Journal, 35(4), 917-926  
   → Kelly Criterion for optimal position sizing

5. **Maillard, S., Roncalli, T., & Teïletche, J. (2010)**  
   *"The Properties of Equally Weighted Risk Contribution Portfolios"*  
   Journal of Portfolio Management, 36(4), 60-70  
   → Risk parity allocation methodology
📘 *These formulas and concepts collectively form the quantitative foundation of MTC Trading v1.0, ensuring that every backtest reflects both mathematical rigor and real-market behavior.*
