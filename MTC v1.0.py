"""
MTC Trading v1.0 
Advanced algorithmic trading backtesting system

Features:
- Multiple trading strategies with proper signal timing
- Comprehensive risk management (stop loss, take profit, trailing stops)
- Portfolio optimization and multi-asset allocation
- Performance metrics and visualizations
- Robust error handling and data validation
- Production-ready code architecture
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================

class StrategyMode(Enum):
    """Strategy execution modes"""
    REALISTIC = "realistic"  # Execute next bar (no look-ahead bias)
    OPTIMISTIC = "optimistic"  # Execute same bar (for comparison only)


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    commission: float = 0.001
    slippage: float = 0.001
    execution_mode: StrategyMode = StrategyMode.REALISTIC
    enable_shorting: bool = False  # Future feature
    
    def __post_init__(self):
        self.commission = max(0.0, float(self.commission))
        self.slippage = max(0.0, float(self.slippage))


# ==================== RISK MANAGER ====================

class RiskManager:
    """
    Professional risk management system with defensive programming.
    Handles position sizing, stop losses, and portfolio protection.
    """

    def __init__(self, 
                 max_position_size: float = 0.20,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15,
                 max_portfolio_risk: float = 0.02,
                 trailing_stop_pct: float = 0.10):
        """
        Initialize risk manager with validated parameters.
        
        Args:
            max_position_size: Max position as fraction of portfolio (0.20 = 20%)
            stop_loss_pct: Stop loss trigger percentage (0.05 = 5%)
            take_profit_pct: Take profit trigger percentage (0.15 = 15%)
            max_portfolio_risk: Max portfolio risk per trade (0.02 = 2%)
            trailing_stop_pct: Trailing stop distance (0.10 = 10%)
        """
        self.max_position_size = self._validate_pct(max_position_size, 0.01, 1.0)
        self.stop_loss_pct = self._validate_pct(stop_loss_pct, 0.001, 0.5)
        self.take_profit_pct = self._validate_pct(take_profit_pct, 0.001, 2.0)
        self.max_portfolio_risk = self._validate_pct(max_portfolio_risk, 0.001, 0.1)
        self.trailing_stop_pct = self._validate_pct(trailing_stop_pct, 0.01, 0.5)
        
        self.active_stops: Dict[str, Dict] = {}
        self.risk_events: List[Dict] = []  # Track all risk events

    @staticmethod
    def _validate_pct(value: float, min_val: float, max_val: float) -> float:
        """Validate percentage parameters"""
        try:
            val = float(value)
            return max(min_val, min(val, max_val))
        except (ValueError, TypeError):
            return min_val

    def calculate_position_size(self, 
                               portfolio_value: float, 
                               price: float,
                               volatility: float = 0.02) -> int:
        """
        Calculate optimal position size using Kelly-inspired sizing.
        
        Returns:
            Number of shares (int). Returns 0 if invalid inputs.
        """
        try:
            if portfolio_value <= 0 or price <= 0:
                return 0

            # Dollar-based position limit
            max_dollars = portfolio_value * self.max_position_size

            # Risk-based sizing (protect against stop loss)
            risk_dollars = portfolio_value * self.max_portfolio_risk
            stop_amount = self.stop_loss_pct * price
            
            if stop_amount <= 0:
                return 0

            shares_by_risk = int(risk_dollars / stop_amount)

            # Apply both constraints
            max_shares = int(max_dollars / price)
            shares = min(shares_by_risk, max_shares)

            return max(shares, 0)
            
        except Exception as e:
            print(f"⚠️  Position sizing error: {e}")
            return 0

    def check_stop_loss(self, ticker: str, entry_price: float, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        if entry_price <= 0 or current_price <= 0:
            return False
        loss_pct = (current_price - entry_price) / entry_price
        triggered = loss_pct <= -self.stop_loss_pct
        
        if triggered:
            self.risk_events.append({
                'ticker': ticker,
                'type': 'stop_loss',
                'entry': entry_price,
                'exit': current_price,
                'pct': loss_pct * 100
            })
        return triggered

    def check_take_profit(self, ticker: str, entry_price: float, current_price: float) -> bool:
        """Check if take profit is triggered"""
        if entry_price <= 0 or current_price <= 0:
            return False
        gain_pct = (current_price - entry_price) / entry_price
        triggered = gain_pct >= self.take_profit_pct
        
        if triggered:
            self.risk_events.append({
                'ticker': ticker,
                'type': 'take_profit',
                'entry': entry_price,
                'exit': current_price,
                'pct': gain_pct * 100
            })
        return triggered

    def update_trailing_stop(self, ticker: str, current_price: float) -> None:
        """Update trailing stop for active position"""
        if current_price <= 0:
            return

        if ticker not in self.active_stops:
            self.active_stops[ticker] = {
                'peak': current_price,
                'stop_price': current_price * (1 - self.trailing_stop_pct)
            }
        elif current_price > self.active_stops[ticker]['peak']:
            self.active_stops[ticker]['peak'] = current_price
            self.active_stops[ticker]['stop_price'] = current_price * (1 - self.trailing_stop_pct)

    def check_trailing_stop(self, ticker: str, current_price: float) -> bool:
        """Check if trailing stop is triggered"""
        if ticker not in self.active_stops or current_price <= 0:
            return False
        
        triggered = current_price <= self.active_stops[ticker]['stop_price']
        if triggered:
            self.risk_events.append({
                'ticker': ticker,
                'type': 'trailing_stop',
                'stop_price': self.active_stops[ticker]['stop_price'],
                'exit': current_price
            })
        return triggered

    def clear_stops(self, ticker: str) -> None:
        """Clear active stops for ticker"""
        if ticker in self.active_stops:
            del self.active_stops[ticker]

    def get_risk_summary(self) -> pd.DataFrame:
        """Get summary of all risk events"""
        if not self.risk_events:
            return pd.DataFrame()
        return pd.DataFrame(self.risk_events)


# ==================== PORTFOLIO MANAGER ====================

class PortfolioManager:
    """
    Multi-asset portfolio management with advanced allocation strategies.
    """

    def __init__(self, 
                 tickers: List[str],
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 100000,
                 allocation_method: str = 'equal_weight'):
        """
        Initialize portfolio manager.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            allocation_method: 'equal_weight', 'risk_parity', 'momentum_weighted', 'market_cap'
        """
        self.tickers = [t.upper().strip() for t in tickers if t and t.strip()]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = max(1000, float(initial_capital))
        self.allocation_method = allocation_method
        self.data: Dict[str, pd.DataFrame] = {}
        self.failed_tickers: List[str] = []
        self.risk_manager = RiskManager()

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data with comprehensive error reporting"""
        if not self.tickers:
            print("❌ No tickers provided")
            return {}

        print(f"\n📥 Fetching data for {len(self.tickers)} asset(s)...")
        print(f"   Period: {self.start_date} to {self.end_date}")
        print("-" * 60)

        try:
            # Batch download for efficiency
            raw = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                threads=True
            )

            if raw is None or raw.empty:
                print("❌ No data downloaded")
                return {}

            # Handle multi-ticker vs single-ticker response
            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in self.tickers:
                    try:
                        df = raw.xs(ticker, axis=1, level=1).copy()
                        if self._validate_data(df, ticker):
                            self.data[ticker] = df
                        else:
                            self.failed_tickers.append(ticker)
                    except KeyError:
                        print(f"❌ {ticker}: Not found")
                        self.failed_tickers.append(ticker)
            else:
                ticker = self.tickers[0]
                if self._validate_data(raw, ticker):
                    self.data[ticker] = raw
                else:
                    self.failed_tickers.append(ticker)

            # Report results
            print("-" * 60)
            print(f"✅ Successfully loaded: {len(self.data)}/{len(self.tickers)} assets")
            if self.failed_tickers:
                print(f"⚠️  Failed: {', '.join(self.failed_tickers)}")
            print()

        except Exception as e:
            print(f"❌ Data fetch error: {e}")

        return self.data

    def _validate_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """Validate downloaded data quality"""
        if df is None or df.empty:
            print(f"❌ {ticker}: No data")
            return False

        if 'Close' not in df.columns:
            print(f"❌ {ticker}: Missing 'Close' column")
            return False

        valid_prices = df['Close'].dropna()
        if len(valid_prices) < 20:
            print(f"❌ {ticker}: Insufficient data ({len(valid_prices)} days)")
            return False

        print(f"✅ {ticker}: {len(valid_prices)} days loaded")
        return True

    def _get_valid_tickers(self) -> List[str]:
        """Get tickers with valid data"""
        return [t for t in self.tickers if t in self.data]

    def calculate_allocations(self) -> Dict[str, float]:
        """Calculate portfolio allocations using selected method"""
        valid = self._get_valid_tickers()
        if not valid:
            return {}

        allocations = {}

        try:
            if self.allocation_method == 'equal_weight':
                weight = 1.0 / len(valid)
                allocations = {ticker: weight for ticker in valid}

            elif self.allocation_method == 'risk_parity':
                volatilities = {}
                for ticker in valid:
                    returns = self.data[ticker]['Close'].pct_change().dropna()
                    vol = returns.std()
                    volatilities[ticker] = max(vol, 1e-8)

                inv_vols = {t: 1 / v for t, v in volatilities.items()}
                total = sum(inv_vols.values())
                allocations = {t: iv / total for t, iv in inv_vols.items()}

            elif self.allocation_method == 'momentum_weighted':
                momentums = {}
                for ticker in valid:
                    close = self.data[ticker]['Close']
                    lookback = min(60, len(close) - 1)
                    if lookback > 0:
                        ret = close.pct_change(periods=lookback).iloc[-1]
                        momentums[ticker] = max(ret, 0.0)
                    else:
                        momentums[ticker] = 0.0

                total_momentum = sum(momentums.values())
                if total_momentum > 0:
                    allocations = {t: m / total_momentum for t, m in momentums.items()}
                else:
                    weight = 1.0 / len(valid)
                    allocations = {ticker: weight for ticker in valid}

            elif self.allocation_method == 'market_cap':
                prices = {ticker: self.data[ticker]['Close'].iloc[-1] for ticker in valid}
                total = sum(prices.values()) or 1.0
                allocations = {t: p / total for t, p in prices.items()}

        except Exception as e:
            print(f"⚠️  Allocation calculation error: {e}")
            weight = 1.0 / len(valid) if valid else 0
            allocations = {ticker: weight for ticker in valid}

        return allocations

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate return correlations between assets"""
        valid = self._get_valid_tickers()
        if len(valid) < 2:
            return pd.DataFrame()

        try:
            returns_dict = {}
            for ticker in valid:
                returns_dict[ticker] = self.data[ticker]['Close'].pct_change().dropna()

            returns_df = pd.DataFrame(returns_dict).dropna()
            return returns_df.corr() if not returns_df.empty else pd.DataFrame()

        except Exception as e:
            print(f"⚠️  Correlation calculation error: {e}")
            return pd.DataFrame()

    def optimize_portfolio_weights(self) -> Dict[str, float]:
        """
        Optimize weights using mean-variance approach.
        Simplified efficient frontier calculation.
        """
        valid = self._get_valid_tickers()
        if not valid:
            return {}

        try:
            returns_dict = {t: self.data[t]['Close'].pct_change().dropna() for t in valid}
            returns_df = pd.DataFrame(returns_dict).dropna()

            if returns_df.empty:
                return {}

            # Annualized metrics
            mean_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252
            volatilities = np.sqrt(np.diag(cov_matrix))
            volatilities = np.maximum(volatilities, 1e-8)

            # Risk-adjusted scores (Sharpe-like)
            inv_vols = 1 / volatilities
            scores = mean_returns.values * inv_vols
            scores = np.maximum(scores, 0)  # No negative weights

            total = scores.sum()
            if total > 0:
                weights = scores / total
            else:
                weights = np.ones(len(valid)) / len(valid)

            return dict(zip(valid, weights))

        except Exception as e:
            print(f"⚠️  Optimization error: {e}")
            weight = 1.0 / len(valid) if valid else 0
            return {ticker: weight for ticker in valid}


# ==================== TRADING BOT ====================

class TradingBot:
    """
    Professional trading bot with multiple strategies and risk management.
    """

    def __init__(self,
                 ticker: str,
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 100000,
                 config: Optional[BacktestConfig] = None,
                 use_risk_management: bool = True):
        """
        Initialize trading bot.
        
        Args:
            ticker: Stock symbol (e.g., 'SPY', 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            config: Backtest configuration
            use_risk_management: Enable risk management features
        """
        self.ticker = ticker.upper().strip()
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = max(1000, float(initial_capital))
        self.config = config or BacktestConfig()
        self.use_risk_management = use_risk_management
        
        self.data: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.portfolio: Optional[pd.DataFrame] = None
        self.risk_manager = RiskManager() if use_risk_management else None
        self.strategy_name = "Unknown"

    def fetch_data(self) -> pd.DataFrame:
        """Fetch and validate historical data"""
        print(f"\n📥 Fetching data for {self.ticker}...")
        print(f"   Period: {self.start_date} to {self.end_date}")
        print("-" * 60)

        try:
            df = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True  # This helps with column naming issues
            )

            if df is None or df.empty:
                raise ValueError(f"No data returned for {self.ticker}")

            # Handle both formats (with and without multi-index)
            if isinstance(df.columns, pd.MultiIndex):
                # Multi-ticker format - extract single ticker
                df = df.xs(self.ticker, axis=1, level=1)
            
            if 'Close' not in df.columns:
                raise ValueError(f"Missing 'Close' price data")

            valid_data = df['Close'].dropna()
            if len(valid_data) < 20:
                raise ValueError(f"Insufficient data: only {len(valid_data)} valid days")

            self.data = df
            
            # Show actual date range received
            actual_start = df.index[0].strftime('%Y-%m-%d')
            actual_end = df.index[-1].strftime('%Y-%m-%d')
            
            print(f"✅ Successfully loaded {len(valid_data)} trading days")
            print(f"   Actual range: {actual_start} to {actual_end}")
            print(f"   Price range: ${valid_data.min():.2f} - ${valid_data.max():.2f}")
            print()

            return self.data

        except Exception as e:
            raise RuntimeError(f"❌ Failed to fetch data for {self.ticker}: {e}")

    def _ensure_data(self):
        """Ensure data is loaded"""
        if self.data is None:
            raise ValueError("❌ Data not loaded. Call fetch_data() first.")

    def _apply_execution_timing(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply execution timing based on configuration.
        
        REALISTIC mode: Execute next bar (no look-ahead bias)
        OPTIMISTIC mode: Execute same bar (for comparison only)
        """
        if self.config.execution_mode == StrategyMode.REALISTIC:
            # Shift signals by 1 to execute on next bar
            signals['positions'] = signals['signal'].diff().shift(1).fillna(0)
        else:
            # Execute on same bar (optimistic, for research only)
            signals['positions'] = signals['signal'].diff()

        return signals

    # ==================== STRATEGIES ====================

    def moving_average_crossover(self, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
        """
        Moving Average Crossover Strategy
        
        BUY: When short MA crosses above long MA
        SELL: When short MA crosses below long MA
        """
        self._ensure_data()
        self.strategy_name = f"MA Crossover ({short_window}/{long_window})"

        short_window = max(2, int(short_window))
        long_window = max(short_window + 1, int(long_window))

        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['Close']
        signals['short_ma'] = self.data['Close'].rolling(window=short_window, min_periods=1).mean()
        signals['long_ma'] = self.data['Close'].rolling(window=long_window, min_periods=1).mean()

        # Generate signals
        signals['signal'] = 0.0
        # Use iloc for position-based indexing
        mask = signals.index >= signals.index[short_window]
        signals.loc[mask, 'signal'] = np.where(
            signals.loc[mask, 'short_ma'] > signals.loc[mask, 'long_ma'],
            1.0, 0.0
        )

        self.signals = self._apply_execution_timing(signals)
        return self.signals

    def momentum_strategy(self, lookback_period: int = 20, threshold: float = 0.05) -> pd.DataFrame:
        """
        Momentum Strategy
        
        BUY: When momentum > threshold
        SELL: When momentum < -threshold
        """
        self._ensure_data()
        self.strategy_name = f"Momentum (lookback={lookback_period})"

        lookback_period = max(1, int(lookback_period))

        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['Close']

        actual_lookback = min(lookback_period, max(1, len(self.data) - 1))
        signals['momentum'] = self.data['Close'].pct_change(periods=actual_lookback)

        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['momentum'] > threshold, 1.0, 0.0)
        signals['signal'] = np.where(signals['momentum'] < -threshold, -1.0, signals['signal'])
        signals['signal'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)

        self.signals = self._apply_execution_timing(signals)
        return self.signals

    def mean_reversion_strategy(self, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Mean Reversion Strategy (Bollinger Bands)
        
        BUY: When price < lower band
        SELL: When price > upper band
        """
        self._ensure_data()
        self.strategy_name = f"Mean Reversion (BB {window}/{num_std}σ)"

        window = max(2, int(window))

        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['Close']
        signals['ma'] = self.data['Close'].rolling(window=window, min_periods=1).mean()
        signals['std'] = self.data['Close'].rolling(window=window, min_periods=1).std().fillna(0)
        signals['upper_band'] = signals['ma'] + (num_std * signals['std'])
        signals['lower_band'] = signals['ma'] - (num_std * signals['std'])

        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['price'] < signals['lower_band'], 1.0, signals['signal'])
        signals['signal'] = np.where(signals['price'] > signals['upper_band'], -1.0, signals['signal'])
        signals['signal'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)

        self.signals = self._apply_execution_timing(signals)
        return self.signals

    def rsi_strategy(self, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
        """
        RSI Strategy
        
        BUY: When RSI < oversold
        SELL: When RSI > overbought
        """
        self._ensure_data()
        self.strategy_name = f"RSI ({period}/{oversold}/{overbought})"

        period = max(2, int(period))

        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['Close']

        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.nan)
        signals['rsi'] = 100 - (100 / (1 + rs.fillna(0)))

        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['rsi'] < oversold, 1.0, signals['signal'])
        signals['signal'] = np.where(signals['rsi'] > overbought, -1.0, signals['signal'])
        signals['signal'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)

        self.signals = self._apply_execution_timing(signals)
        return self.signals

    # ==================== BACKTESTING ====================

    def backtest(self) -> pd.DataFrame:
        """
        Execute backtest with comprehensive risk management.
        """
        if self.signals is None:
            raise ValueError("❌ No signals generated. Run a strategy first.")

        print(f"\n🔄 Running backtest...")
        print(f"   Strategy: {self.strategy_name}")
        print(f"   Mode: {self.config.execution_mode.value.upper()}")
        print(f"   Risk Management: {'ON' if self.use_risk_management else 'OFF'}")
        print("-" * 60)

        portfolio = pd.DataFrame(index=self.signals.index)
        portfolio['price'] = self.signals['price']
        portfolio['holdings'] = 0.0
        portfolio['cash'] = 0.0
        portfolio['total'] = 0.0
        portfolio['returns'] = 0.0
        portfolio['position_shares'] = 0
        
        if self.use_risk_management:
            portfolio['stop_loss_triggered'] = 0
            portfolio['take_profit_triggered'] = 0
            portfolio['trailing_stop_triggered'] = 0

        # Initialize state
        position = 0
        cash = float(self.initial_capital)
        entry_price = 0.0
        
        # Calculate volatility for position sizing
        returns = self.signals['price'].pct_change().dropna()
        volatility = max(returns.std(), 1e-8)

        # Initial state
        portfolio.iloc[0, portfolio.columns.get_loc('cash')] = cash
        portfolio.iloc[0, portfolio.columns.get_loc('total')] = cash

        # Backtest loop
        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            price = portfolio['price'].iloc[i]

            # Skip invalid prices
            if price <= 0 or np.isnan(price):
                portfolio.loc[date, 'holdings'] = position * price
                portfolio.loc[date, 'cash'] = cash
                portfolio.loc[date, 'total'] = cash + (position * price)
                portfolio.loc[date, 'position_shares'] = position
                continue

            portfolio_value = cash + (position * price)

            # Check risk management conditions
            stop_triggered = False
            profit_triggered = False
            trailing_triggered = False

            if self.use_risk_management and position > 0 and entry_price > 0:
                # Stop loss check
                if self.risk_manager.check_stop_loss(self.ticker, entry_price, price):
                    stop_triggered = True
                    portfolio.loc[date, 'stop_loss_triggered'] = 1

                # Take profit check
                if self.risk_manager.check_take_profit(self.ticker, entry_price, price):
                    profit_triggered = True
                    portfolio.loc[date, 'take_profit_triggered'] = 1

                # Trailing stop
                self.risk_manager.update_trailing_stop(self.ticker, price)
                if self.risk_manager.check_trailing_stop(self.ticker, price):
                    trailing_triggered = True
                    portfolio.loc[date, 'trailing_stop_triggered'] = 1

            # ENTRY LOGIC: Buy signal
            if self.signals['positions'].iloc[i] > 0 and position == 0:
                if self.use_risk_management:
                    shares_to_buy = self.risk_manager.calculate_position_size(
                        portfolio_value, price, volatility
                    )
                else:
                    # Use 95% of available cash
                    shares_to_buy = int((cash * 0.95) / (price * (1 + self.config.commission + self.config.slippage)))

                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.config.commission + self.config.slippage)
                    if cost <= cash:
                        cash -= cost
                        position = shares_to_buy
                        entry_price = price

            # EXIT LOGIC: Sell signal or risk management
            elif (self.signals['positions'].iloc[i] < 0 or stop_triggered or 
                  profit_triggered or trailing_triggered) and position > 0:
                
                proceeds = position * price * (1 - self.config.commission - self.config.slippage)
                cash += proceeds
                position = 0
                entry_price = 0.0

                # Clear trailing stops
                if self.use_risk_management:
                    self.risk_manager.clear_stops(self.ticker)

            # Update portfolio state
            portfolio.loc[date, 'holdings'] = position * price
            portfolio.loc[date, 'cash'] = cash
            portfolio.loc[date, 'total'] = cash + (position * price)
            portfolio.loc[date, 'position_shares'] = position

            # Calculate returns
            prev_total = portfolio['total'].iloc[i - 1]
            if prev_total > 0:
                portfolio.loc[date, 'returns'] = (portfolio.loc[date, 'total'] / prev_total) - 1

        # Buy & Hold benchmark
        portfolio['buy_hold'] = self.initial_capital * (
            portfolio['price'] / portfolio['price'].iloc[0]
        )

        self.portfolio = portfolio
        print("✅ Backtest completed\n")
        return portfolio

    # ==================== ANALYTICS ====================

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if self.portfolio is None:
            raise ValueError("❌ No portfolio data. Run backtest first.")

        # Returns
        total_return = (self.portfolio['total'].iloc[-1] / self.initial_capital - 1) * 100
        buy_hold_return = (self.portfolio['buy_hold'].iloc[-1] / self.initial_capital - 1) * 100
        
        returns = self.portfolio['returns'].dropna()
        if returns.empty or len(returns) < 2:
            # Not enough data for meaningful metrics
            return {
                'Total Return (%)': round(total_return, 2),
                'Buy & Hold Return (%)': round(buy_hold_return, 2),
                'Outperformance (%)': round(total_return - buy_hold_return, 2),
                'Final Portfolio Value ($)': round(self.portfolio['total'].iloc[-1], 2),
            }

        # Risk-adjusted metrics
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

        # Win rate
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = (winning_days / total_days * 100) if total_days > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * returns.mean() / downside_std if downside_std != 0 else 0

        # Calmar ratio
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        # Max consecutive wins/losses
        win_streak = self._calculate_streak(returns > 0)
        loss_streak = self._calculate_streak(returns < 0)

        # Risk management stats
        metrics = {
            'Total Return (%)': round(total_return, 2),
            'Buy & Hold Return (%)': round(buy_hold_return, 2),
            'Outperformance (%)': round(total_return - buy_hold_return, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Sortino Ratio': round(sortino_ratio, 2),
            'Calmar Ratio': round(calmar_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Volatility (%)': round(volatility, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Max Win Streak': win_streak,
            'Max Loss Streak': loss_streak,
            'Final Portfolio Value ($)': round(self.portfolio['total'].iloc[-1], 2),
            'Total Trades': self._count_trades(),
        }

        # Add risk management metrics
        if self.use_risk_management:
            metrics.update({
                'Stop Losses Triggered': int(self.portfolio['stop_loss_triggered'].sum()),
                'Take Profits Triggered': int(self.portfolio['take_profit_triggered'].sum()),
                'Trailing Stops Triggered': int(self.portfolio['trailing_stop_triggered'].sum()),
            })

        return metrics

    def _calculate_streak(self, condition: pd.Series) -> int:
        """Calculate maximum consecutive streak"""
        streaks = condition.astype(int).groupby((condition != condition.shift()).cumsum()).sum()
        return int(streaks.max()) if not streaks.empty else 0

    def _count_trades(self) -> int:
        """Count total number of trades executed"""
        if self.signals is None:
            return 0
        return int((self.signals['positions'].abs() > 0).sum())

    def print_metrics(self):
        """Print formatted performance metrics"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*70)
        print(f"PERFORMANCE METRICS - {self.ticker}")
        print(f"Strategy: {self.strategy_name}")
        print("="*70)
        
        for key, value in metrics.items():
            print(f"{key:.<50} {value:>15}")
        
        print("="*70 + "\n")

    def get_trade_log(self) -> pd.DataFrame:
        """Generate detailed trade log with entry/exit information"""
        if self.signals is None or self.portfolio is None:
            return pd.DataFrame()

        trades = []
        position = 0
        entry_price = 0.0
        entry_date = None
        entry_shares = 0

        for i in range(len(self.signals)):
            date = self.signals.index[i]
            price = self.signals['price'].iloc[i]

            # Entry
            if self.signals['positions'].iloc[i] > 0 and position == 0:
                position = 1
                entry_price = price
                entry_date = date
                entry_shares = int(self.portfolio['position_shares'].iloc[i])
                
                trades.append({
                    'Entry Date': date,
                    'Exit Date': None,
                    'Entry Price': price,
                    'Exit Price': None,
                    'Shares': entry_shares,
                    'Return (%)': 0,
                    'Exit Reason': None,
                    'Duration (days)': 0
                })

            # Exit
            elif position == 1 and len(trades) > 0:
                exit_reason = None
                should_exit = False

                if self.signals['positions'].iloc[i] < 0:
                    should_exit = True
                    exit_reason = 'Strategy Signal'
                elif self.use_risk_management:
                    if self.portfolio['stop_loss_triggered'].iloc[i] > 0:
                        should_exit = True
                        exit_reason = 'Stop Loss'
                    elif self.portfolio['take_profit_triggered'].iloc[i] > 0:
                        should_exit = True
                        exit_reason = 'Take Profit'
                    elif self.portfolio['trailing_stop_triggered'].iloc[i] > 0:
                        should_exit = True
                        exit_reason = 'Trailing Stop'

                if should_exit:
                    position = 0
                    trade_return = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                    duration = (date - entry_date).days if entry_date else 0
                    
                    trades[-1].update({
                        'Exit Date': date,
                        'Exit Price': price,
                        'Return (%)': round(trade_return, 2),
                        'Exit Reason': exit_reason,
                        'Duration (days)': duration
                    })

        return pd.DataFrame(trades)

    # ==================== VISUALIZATION ====================

    def plot_results(self, figsize: Tuple[int, int] = (16, 12), save_path: Optional[str] = None):
        """Generate comprehensive performance visualization"""
        if self.portfolio is None:
            raise ValueError("❌ No portfolio data. Run backtest first.")

        # Smart downsampling - always include the last point
        max_points = 2000
        if len(self.portfolio) > max_points:
            step = len(self.portfolio) // max_points
            indices = list(range(0, len(self.portfolio), step))
            # Always include the last index
            if indices[-1] != len(self.portfolio) - 1:
                indices.append(len(self.portfolio) - 1)
            plot_data = self.portfolio.iloc[indices].copy()
            plot_signals = self.signals.iloc[indices].copy()
        else:
            plot_data = self.portfolio
            plot_signals = self.signals

        fig, axes = plt.subplots(4, 1, figsize=figsize)
        fig.suptitle(f'{self.ticker} - {self.strategy_name}', fontsize=16, fontweight='bold')

        # Plot 1: Portfolio Value
        axes[0].plot(plot_data.index, plot_data['total'], label='Strategy', linewidth=2, color='#10b981')
        axes[0].plot(plot_data.index, plot_data['buy_hold'], label='Buy & Hold', linewidth=2, 
                    color='#ef4444', linestyle='--', alpha=0.7)
        axes[0].set_ylabel('Portfolio Value ($)', fontsize=11)
        axes[0].set_title('Portfolio Value Over Time', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Plot 2: Price with Signals
        axes[1].plot(plot_signals.index, plot_signals['price'], label='Price', 
                    linewidth=1.5, color='#3b82f6', alpha=0.8)

        # Moving averages if available
        if 'short_ma' in plot_signals.columns:
            axes[1].plot(plot_signals.index, plot_signals['short_ma'], label='Short MA', 
                        linewidth=1, color='#f59e0b', alpha=0.6)
            axes[1].plot(plot_signals.index, plot_signals['long_ma'], label='Long MA', 
                        linewidth=1, color='#8b5cf6', alpha=0.6)

        # Bollinger bands if available
        if 'upper_band' in plot_signals.columns:
            axes[1].plot(plot_signals.index, plot_signals['upper_band'], 
                        linewidth=1, color='gray', linestyle='--', alpha=0.5)
            axes[1].plot(plot_signals.index, plot_signals['lower_band'], 
                        linewidth=1, color='gray', linestyle='--', alpha=0.5)
            axes[1].fill_between(plot_signals.index, plot_signals['upper_band'], 
                                plot_signals['lower_band'], alpha=0.1, color='gray')

        # Buy/Sell signals - use full data, not downsampled
        buy_signals = self.signals[self.signals['positions'] > 0]
        if not buy_signals.empty:
            axes[1].scatter(buy_signals.index, buy_signals['price'], 
                           marker='^', color='green', s=100, label='Buy', zorder=5, alpha=0.8)

        sell_signals = self.signals[self.signals['positions'] < 0]
        if not sell_signals.empty:
            axes[1].scatter(sell_signals.index, sell_signals['price'], 
                           marker='v', color='red', s=100, label='Sell', zorder=5, alpha=0.8)

        # Risk management exits
        if self.use_risk_management:
            stops = self.portfolio[self.portfolio['stop_loss_triggered'] > 0]
            if not stops.empty:
                axes[1].scatter(stops.index, self.signals.reindex(stops.index)['price'], 
                               marker='x', color='darkred', s=150, label='Stop Loss', zorder=6)
            
            profits = self.portfolio[self.portfolio['take_profit_triggered'] > 0]
            if not profits.empty:
                axes[1].scatter(profits.index, self.signals.reindex(profits.index)['price'], 
                               marker='*', color='gold', s=150, label='Take Profit', zorder=6)

        axes[1].set_ylabel('Price ($)', fontsize=11)
        axes[1].set_title('Trading Signals & Price Action', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper left', fontsize=9, ncol=2)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Drawdown
        returns = self.portfolio['returns'].dropna()
        if not returns.empty:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100

            axes[2].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
            axes[2].plot(drawdown.index, drawdown, color='darkred', linewidth=1.5)
            
            if self.use_risk_management:
                stop_level = -self.risk_manager.stop_loss_pct * 100
                axes[2].axhline(y=stop_level, color='orange', linestyle='--', 
                               label=f'Stop Loss Level ({stop_level:.1f}%)', alpha=0.7)
                axes[2].legend(loc='lower left', fontsize=9)

        axes[2].set_ylabel('Drawdown (%)', fontsize=11)
        axes[2].set_title('Drawdown Analysis', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Rolling Sharpe Ratio
        if not returns.empty and len(returns) > 60:
            rolling_returns = returns.rolling(window=60)
            rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)

            axes[3].plot(rolling_sharpe.index, rolling_sharpe, color='#8b5cf6', linewidth=2)
            axes[3].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good (>1)')
            axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Poor (<0)')
            axes[3].fill_between(rolling_sharpe.index, 0, rolling_sharpe, 
                                where=(rolling_sharpe > 0), alpha=0.2, color='green')
            axes[3].fill_between(rolling_sharpe.index, 0, rolling_sharpe, 
                                where=(rolling_sharpe < 0), alpha=0.2, color='red')

        axes[3].set_xlabel('Date', fontsize=11)
        axes[3].set_ylabel('Sharpe Ratio', fontsize=11)
        axes[3].set_title('Rolling Sharpe Ratio (60-day)', fontsize=13, fontweight='bold')
        axes[3].legend(loc='upper left', fontsize=9)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Chart saved: {save_path}")

        return fig


# ==================== INTERACTIVE MENU ====================

def display_menu():
    """Display main menu"""
    print("\n" + "="*70)
    print("🚀 MTC Trading Bot v1.0 - Ultimate Edition")
    print("="*70)
    print("⚠️  Disclaimer: Past performance does not guarantee future results")
    print("💎 'If you can't hold, you won't be rich.' - CZ")
    print("="*70 + "\n")
    print("MAIN MENU:")
    print("-" * 70)
    print(" 1. 📊 Single Asset Trading (Advanced Risk Management)")
    print(" 2. 💼 Multi-Asset Portfolio Management")
    print(" 3. 🔬 Strategy Comparison (All Strategies)")
    print(" 4. ⚡ Quick Backtest (SPY with defaults)")
    print(" 5. 📚 Strategy Performance Report")
    print(" 6. ❌ Exit")
    print("-" * 70)
    
    choice = input("\n👉 Select option (1-6): ").strip()
    return choice


def single_asset_trading():
    """Single asset trading workflow"""
    print("\n" + "="*70)
    print("📊 SINGLE ASSET TRADING")
    print("="*70)

    ticker = input("Enter ticker (e.g., SPY, AAPL, TSLA): ").strip().upper()
    if not ticker:
        ticker = 'SPY'
        print(f"Using default: {ticker}")
    
    start_date = input("Start date (YYYY-MM-DD) [default: 2022-01-01]: ").strip() or '2022-01-01'
    
    # Use datetime to get most recent data
    from datetime import datetime
    default_end = datetime.now().strftime('%Y-%m-%d')
    end_date_input = input(f"End date (YYYY-MM-DD) [default: {default_end}]: ").strip()
    end_date = end_date_input if end_date_input else default_end
    
    capital_input = input("Initial capital [default: $100,000]: ").strip()
    capital = float(capital_input) if capital_input else 100000.0

    print("\n📋 Select Strategy:")
    print("1. Moving Average Crossover")
    print("2. Momentum Strategy")
    print("3. Mean Reversion (Bollinger Bands)")
    print("4. RSI Strategy")
    
    strategy_choice = input("Strategy (1-4) [default: 1]: ").strip() or '1'

    print("\n⚙️  Configuration:")
    use_risk = input("Enable risk management? (Y/n) [default: Y]: ").strip().lower()
    use_risk = use_risk != 'n'
    
    mode_choice = input("Execution mode (1=Realistic, 2=Optimistic) [default: 1]: ").strip() or '1'
    execution_mode = StrategyMode.OPTIMISTIC if mode_choice == '2' else StrategyMode.REALISTIC

    config = BacktestConfig(
        commission=0.001,
        slippage=0.001,
        execution_mode=execution_mode
    )

    # Initialize bot
    print(f"\n🚀 Initializing bot for {ticker}...")
    bot = TradingBot(ticker, start_date, end_date, capital, config, use_risk)
    
    try:
        bot.fetch_data()
    except Exception as e:
        print(f"\n❌ Error fetching data: {e}")
        return

    # Run selected strategy
    try:
        if strategy_choice == '1':
            short = int(input("\nShort MA window [default: 50]: ").strip() or 50)
            long = int(input("Long MA window [default: 200]: ").strip() or 200)
            bot.moving_average_crossover(short, long)
        elif strategy_choice == '2':
            lookback = int(input("\nLookback period [default: 20]: ").strip() or 20)
            threshold = float(input("Momentum threshold [default: 0.05]: ").strip() or 0.05)
            bot.momentum_strategy(lookback, threshold)
        elif strategy_choice == '3':
            window = int(input("\nBB window [default: 20]: ").strip() or 20)
            std = float(input("Std deviations [default: 2.0]: ").strip() or 2.0)
            bot.mean_reversion_strategy(window, std)
        elif strategy_choice == '4':
            period = int(input("\nRSI period [default: 14]: ").strip() or 14)
            oversold = int(input("Oversold level [default: 30]: ").strip() or 30)
            overbought = int(input("Overbought level [default: 70]: ").strip() or 70)
            bot.rsi_strategy(period, oversold, overbought)
        else:
            print("Invalid choice, using MA Crossover")
            bot.moving_average_crossover(50, 200)

        # Run backtest
        bot.backtest()
        bot.print_metrics()

        # Trade log
        print("\n📋 TRADE LOG (Last 10 trades):")
        trade_log = bot.get_trade_log()
        if trade_log.empty:
            print("No trades executed")
        else:
            print(trade_log.tail(10).to_string(index=False))
            
            # Trade statistics
            if len(trade_log) > 0:
                winning_trades = trade_log[trade_log['Return (%)'] > 0]
                losing_trades = trade_log[trade_log['Return (%)'] <= 0]
                print(f"\n📈 Trade Statistics:")
                print(f"   Total Trades: {len(trade_log)}")
                print(f"   Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trade_log)*100:.1f}%)")
                print(f"   Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trade_log)*100:.1f}%)")
                print(f"   Average Return: {trade_log['Return (%)'].mean():.2f}%")
                print(f"   Best Trade: {trade_log['Return (%)'].max():.2f}%")
                print(f"   Worst Trade: {trade_log['Return (%)'].min():.2f}%")
                
                if 'Duration (days)' in trade_log.columns:
                    avg_duration = trade_log[trade_log['Duration (days)'] > 0]['Duration (days)'].mean()
                    if not np.isnan(avg_duration):
                        print(f"   Avg Trade Duration: {avg_duration:.1f} days")

        # Generate chart
        show_chart = input("\n📊 Generate chart? (Y/n): ").strip().lower()
        if show_chart != 'n':
            print("Generating visualization...")
            filename = f'{ticker}_{bot.strategy_name.replace(" ", "_").replace("/", "-")}_performance.png'
            bot.plot_results(save_path=filename)
            plt.show()
            print(f"✅ Chart saved as: {filename}")

    except ValueError as e:
        print(f"\n❌ Invalid input: {e}")
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()


def multi_asset_portfolio():
    """Multi-asset portfolio workflow"""
    print("\n" + "="*70)
    print("💼 MULTI-ASSET PORTFOLIO MANAGEMENT")
    print("="*70)

    tickers_input = input("Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL): ").strip()
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    if not tickers:
        print("❌ No tickers provided")
        return

    start_date = input("Start date [default: 2022-01-01]: ").strip() or '2022-01-01'
    end_date = input("End date [default: 2025-10-15]: ").strip() or '2025-10-15'
    capital = float(input("Initial capital [default: $100,000]: ").strip() or 100000)

    print("\n📊 Allocation Methods:")
    print("1. Equal Weight")
    print("2. Risk Parity")
    print("3. Momentum Weighted")
    print("4. Market Cap Weighted")
    
    method_choice = input("Select method (1-4): ").strip()
    methods = {
        '1': 'equal_weight',
        '2': 'risk_parity',
        '3': 'momentum_weighted',
        '4': 'market_cap'
    }
    method = methods.get(method_choice, 'equal_weight')

    # Build portfolio
    portfolio = PortfolioManager(tickers, start_date, end_date, capital, method)
    portfolio.fetch_data()

    if not portfolio.data:
        print("❌ No valid data loaded")
        return

    # Show allocations
    print("\n" + "="*70)
    print("📊 PORTFOLIO ALLOCATIONS")
    print("="*70)
    allocations = portfolio.calculate_allocations()
    for ticker, weight in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(weight * 50)
        print(f"{ticker:6s} {weight*100:6.2f}% {bar}")

    # Correlation matrix
    print("\n" + "="*70)
    print("📈 CORRELATION MATRIX")
    print("="*70)
    corr = portfolio.calculate_correlation_matrix()
    if not corr.empty:
        print(corr.round(2).to_string())
    else:
        print("Insufficient data for correlation analysis")

    # Optimized weights
    print("\n" + "="*70)
    print("🎯 OPTIMIZED WEIGHTS (Mean-Variance)")
    print("="*70)
    optimized = portfolio.optimize_portfolio_weights()
    for ticker, weight in sorted(optimized.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(weight * 50)
        print(f"{ticker:6s} {weight*100:6.2f}% {bar}")


def strategy_comparison():
    """Compare all strategies"""
    print("\n" + "="*70)
    print("🔬 STRATEGY COMPARISON")
    print("="*70)

    ticker = input("Enter ticker [default: SPY]: ").strip().upper() or 'SPY'
    
    config = BacktestConfig(execution_mode=StrategyMode.REALISTIC)
    bot = TradingBot(ticker, '2022-01-01', '2025-10-15', 100000, config, True)

    try:
        bot.fetch_data()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    strategies = {
        'MA Crossover (50/200)': lambda: bot.moving_average_crossover(50, 200),
        'Momentum (20d, 5%)': lambda: bot.momentum_strategy(20, 0.05),
        'Mean Reversion (BB 20/2)': lambda: bot.mean_reversion_strategy(20, 2.0),
        'RSI (14/30/70)': lambda: bot.rsi_strategy(14, 30, 70),
    }

    results = {}
    best_strategy = None
    best_sharpe = -999

    print("\n🔄 Testing strategies...")
    for name, strategy_func in strategies.items():
        print(f"\n   Testing: {name}")
        try:
            strategy_func()
            bot.backtest()
            metrics = bot.calculate_metrics()
            results[name] = metrics
            
            if metrics['Sharpe Ratio'] > best_sharpe:
                best_sharpe = metrics['Sharpe Ratio']
                best_strategy = name
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Display results
    print("\n" + "="*70)
    print("🏆 STRATEGY COMPARISON RESULTS")
    print("="*70)
    
    if results:
        comparison = pd.DataFrame(results).T
        key_metrics = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
                      'Win Rate (%)', 'Volatility (%)']
        display_metrics = [m for m in key_metrics if m in comparison.columns]
        print(comparison[display_metrics].to_string())
        
        print("\n" + "="*70)
        print(f"🥇 BEST STRATEGY (by Sharpe): {best_strategy}")
        print(f"   Sharpe Ratio: {best_sharpe:.2f}")
        print("="*70)
    else:
        print("❌ No results to display")


def quick_backtest():
    """Quick backtest with defaults"""
    print("\n" + "="*70)
    print("⚡ QUICK BACKTEST")
    print("="*70)
    print("Configuration: SPY | MA(50/200) | Risk Management ON")
    print("Period: 2022-01-01 to 2024-12-31")
    print("-" * 70)

    config = BacktestConfig(execution_mode=StrategyMode.REALISTIC)
    bot = TradingBot('SPY', '2022-01-01', '2024-12-31', 100000, config, True)

    try:
        bot.fetch_data()
        bot.moving_average_crossover(50, 200)
        bot.backtest()
        bot.print_metrics()

        trade_log = bot.get_trade_log()
        if not trade_log.empty:
            print("\n📋 TRADE LOG:")
            print(trade_log.to_string(index=False))

        bot.plot_results(save_path='quick_backtest.png')
        plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")


def performance_report():
    """Generate detailed performance report"""
    print("\n" + "="*70)
    print("📚 STRATEGY PERFORMANCE REPORT")
    print("="*70)
    
    ticker = input("Enter ticker [default: SPY]: ").strip().upper() or 'SPY'
    
    config = BacktestConfig(execution_mode=StrategyMode.REALISTIC)
    bot = TradingBot(ticker, '2022-01-01', '2025-10-15', 100000, config, True)

    try:
        bot.fetch_data()
        bot.moving_average_crossover(50, 200)
        bot.backtest()
        
        metrics = bot.calculate_metrics()
        trade_log = bot.get_trade_log()
        
        # Generate comprehensive report
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE PERFORMANCE REPORT")
        print("="*70)
        print(f"\nTicker: {ticker}")
        print(f"Strategy: {bot.strategy_name}")
        print(f"Period: {bot.start_date} to {bot.end_date}")
        print(f"Initial Capital: ${bot.initial_capital:,.2f}")
        
        print("\n📈 Returns:")
        print(f"   Total Return: {metrics['Total Return (%)']:.2f}%")
        print(f"   Buy & Hold: {metrics['Buy & Hold Return (%)']:.2f}%")
        print(f"   Outperformance: {metrics['Outperformance (%)']:.2f}%")
        print(f"   Final Value: ${metrics['Final Portfolio Value ($)']:,.2f}")
        
        print("\n📊 Risk Metrics:")
        print(f"   Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"   Sortino Ratio: {metrics['Sortino Ratio']:.2f}")
        print(f"   Calmar Ratio: {metrics['Calmar Ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
        print(f"   Volatility: {metrics['Volatility (%)']:.2f}%")
        
        print("\n💼 Trading Activity:")
        print(f"   Total Trades: {metrics['Total Trades']}")
        print(f"   Win Rate: {metrics['Win Rate (%)']:.2f}%")
        
        if 'Stop Losses Triggered' in metrics:
            print(f"   Stop Losses: {metrics['Stop Losses Triggered']}")
            print(f"   Take Profits: {metrics['Take Profits Triggered']}")
            print(f"   Trailing Stops: {metrics['Trailing Stops Triggered']}")
        
        if not trade_log.empty:
            print("\n📋 Trade Analysis:")
            winning = trade_log[trade_log['Return (%)'] > 0]
            losing = trade_log[trade_log['Return (%)'] <= 0]
            
            print(f"   Winning Trades: {len(winning)}")
            print(f"   Losing Trades: {len(losing)}")
            if len(winning) > 0:
                print(f"   Avg Win: {winning['Return (%)'].mean():.2f}%")
                print(f"   Best Win: {winning['Return (%)'].max():.2f}%")
            if len(losing) > 0:
                print(f"   Avg Loss: {losing['Return (%)'].mean():.2f}%")
                print(f"   Worst Loss: {losing['Return (%)'].min():.2f}%")
            
            if 'Duration (days)' in trade_log.columns:
                valid_durations = trade_log[trade_log['Duration (days)'] > 0]['Duration (days)']
                if not valid_durations.empty:
                    print(f"   Avg Trade Duration: {valid_durations.mean():.1f} days")
        
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== MAIN EXECUTION ====================

def main():
    """Main program loop"""
    print("\n" + "="*70)
    print("🎯 Welcome to MTC Trading Bot v1.0")
    print("="*70)
    
    while True:
        choice = display_menu()
        
        try:
            if choice == '1':
                single_asset_trading()
            elif choice == '2':
                multi_asset_portfolio()
            elif choice == '3':
                strategy_comparison()
            elif choice == '4':
                quick_backtest()
            elif choice == '5':
                performance_report()
            elif choice == '6':
                print("\n" + "="*70)
                print("👋 Thanks for using MTC v1.0!")
                print("💎 Remember: Diamond hands are algorithmic, not emotional!")
                print("="*70 + "\n")
                break
            else:
                print("\n❌ Invalid choice. Please select 1-6.")
                continue
            
            # Ask to continue
            continue_choice = input("\n↩️  Return to main menu? (Y/n): ").strip().lower()
            if continue_choice == 'n':
                print("\n" + "="*70)
                print("👋 Thanks for using MTC v1.0!")
                print("💎 Stay disciplined, trust the algorithm!")
                print("="*70 + "\n")
                break
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or select a different option.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Program terminated.")