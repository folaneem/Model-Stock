import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital for backtesting (default: 100,000)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, float] = {}  # Ticker -> Quantity
        self.positions_history: List[Dict] = []  # Track positions over time
        self.portfolio_values = []  # Track total portfolio value over time
        self.dates = []  # Track dates for portfolio values
        self.trades: List[Dict[str, Any]] = []
        self.metrics: Dict[str, float] = {}
        self.current_date: Optional[datetime] = None
        self.logger = logging.getLogger(__name__)
        
    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """
        Update portfolio value based on current positions and prices
        
        Args:
            current_prices: Dictionary of {ticker: current_price}
        """
        if not self.current_date:
            return
            
        positions_value = 0.0
        positions_snapshot = {}
        
        for ticker, quantity in self.positions.items():
            if ticker in current_prices and quantity > 0:
                value = quantity * current_prices[ticker]
                positions_value += value
                positions_snapshot[ticker] = {
                    'quantity': quantity,
                    'price': current_prices[ticker],
                    'value': value
                }
        
        total_value = self.current_capital + positions_value
        
        # Record portfolio snapshot
        self.portfolio_values.append(total_value)
        self.dates.append(self.current_date)
        self.positions_history.append({
            'date': self.current_date,
            'cash': self.current_capital,
            'positions': positions_snapshot,
            'total_value': total_value
        })


        
    def _calculate_metrics(self):
        """
        Calculate comprehensive backtesting metrics
        """
        if not self.trades or len(self.portfolio_values) < 2:
            self.metrics = {
                'total_trades': 0,
                'win_rate': 0,
                'average_return': 0,
                'sharpe_ratio': 0,
                'maximum_drawdown': 0,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'profit_factor': 0
            }
            return
        
        # Calculate daily returns
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        # Calculate trade returns
        trade_returns = []
        for i in range(len(self.trades) - 1):
            trade1 = self.trades[i]
            trade2 = self.trades[i + 1]
            
            if trade1['action'] == 'buy' and trade2['action'] == 'sell' and trade1['ticker'] == trade2['ticker']:
                profit = (trade2['price'] - trade1['price']) * trade1['quantity']
                trade_returns.append(profit / (trade1['price'] * trade1['quantity']))
        
        # Calculate profit factor
        gains = [r for r in trade_returns if r > 0]
        losses = [abs(r) for r in trade_returns if r < 0]
        profit_factor = sum(gains) / sum(losses) if losses and sum(losses) > 0 else float('inf')
        
        # Calculate Sortino ratio (using risk-free rate of 0.02 for annualized)
        risk_free_rate = 0.02
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf
        downside_returns = np.where(returns < daily_rf, returns - daily_rf, 0)
        downside_volatility = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        sortino_ratio = (np.mean(excess_returns) * 252) / (downside_volatility + 1e-10)
        
        # Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Calmar Ratio (Return / Max Drawdown)
        total_return = (self.portfolio_values[-1] / self.initial_capital) - 1
        years = len(self.dates) / 252  # Approximate trading days in a year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        calmar_ratio = annualized_return / (max_drawdown + 1e-10)
        
        # Calculate metrics
        self.metrics = {
            'total_trades': len(trade_returns),
            'win_rate': len(gains) / len(trade_returns) if trade_returns else 0,
            'average_return': np.mean(trade_returns) if trade_returns else 0,
            'sharpe_ratio': np.sqrt(252) * (np.mean(excess_returns) / (np.std(returns) + 1e-10)) if len(returns) > 1 else 0,
            'maximum_drawdown': max_drawdown,
            'final_capital': self.portfolio_values[-1],
            'total_return': total_return * 100,  # as percentage
            'annualized_return': annualized_return * 100,  # as percentage
            'volatility': np.std(returns) * np.sqrt(252) * 100,  # annualized, as percentage
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'profit_factor': profit_factor
        }
        
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from portfolio values
        
        Returns:
            float: Maximum drawdown as a decimal (e.g., 0.10 for 10%)
        """
        if len(self.portfolio_values) < 2:
            return 0.0
            
        peak = self.portfolio_values[0]
        max_drawdown = 0.0
        
        for value in self.portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return max_drawdown
        
    def get_drawdowns(self) -> Dict[str, List[float]]:
        """
        Calculate drawdown series
        
        Returns:
            dict: Dictionary with 'dates' and 'drawdowns' lists
        """
        if len(self.portfolio_values) < 2 or len(self.dates) != len(self.portfolio_values):
            return {'dates': [], 'drawdowns': []}
            
        peak = self.portfolio_values[0]
        drawdowns = []
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            drawdowns.append(drawdown)
            
        return {
            'dates': self.dates,
            'drawdowns': drawdowns,
            'max_drawdown': max(drawdowns) if drawdowns else 0,
            'max_drawdown_date': self.dates[drawdowns.index(max(drawdowns))] if drawdowns else None
        }
        
    def execute_trade(self, ticker: str, action: str, price: float, quantity: float, current_prices: Dict[str, float] = None):
        """
        Execute a trade and update portfolio
        
        Args:
            ticker: Stock ticker
            action: 'buy' or 'sell'
            price: Execution price
            quantity: Number of shares
            current_prices: Current prices of all positions for portfolio valuation
        """
        if not self.current_date:
            self.current_date = datetime.now()
            
        trade = {
            'ticker': ticker,
            'action': action,
            'price': price,
            'quantity': quantity,
            'timestamp': self.current_date,
            'portfolio_value': None
        }
        
        # Update positions
        if action == 'buy':
            if ticker in self.positions:
                self.positions[ticker] += quantity
            else:
                self.positions[ticker] = quantity
            
            self.current_capital -= price * quantity
            
        elif action == 'sell':
            if ticker in self.positions:
                # Don't sell more than we own
                quantity = min(quantity, self.positions[ticker])
                self.positions[ticker] -= quantity
                if self.positions[ticker] <= 1e-6:  # Account for floating point errors
                    del self.positions[ticker]
                
                self.current_capital += price * quantity
            else:
                self.logger.warning(f"Attempted to sell {ticker} but no position exists")
                return
        
        # Record trade
        self.trades.append(trade)
        
        # Update portfolio value if current_prices is provided
        if current_prices is not None:
            self._update_portfolio_value(current_prices)
            trade['portfolio_value'] = self.portfolio_values[-1] if self.portfolio_values else self.current_capital
        
        self.logger.info(f"Executed trade: {action} {quantity:.2f} shares of {ticker} at ${price:.2f}")
        
        return trade
        
    def evaluate_strategy(self, predictions: pd.DataFrame, data: pd.DataFrame):
        """
        Evaluate trading strategy based on predictions
        
        Args:
            predictions: DataFrame with prediction values
            data: DataFrame with price data
        """
        if not isinstance(predictions.index, pd.DatetimeIndex) or not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Both predictions and data must have datetime indices")
        
        # Ensure we have the same dates in both DataFrames
        common_dates = predictions.index.intersection(data.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between predictions and data")
            
        for date in common_dates:
            self.current_date = date
            current_prices = data.loc[date]
            
            # Get current positions value
            positions_value = 0
            for ticker, quantity in self.positions.items():
                if ticker in current_prices:
                    positions_value += quantity * current_prices[ticker]
            
            # Update portfolio value for this date
            self._update_portfolio_value(current_prices.to_dict())
            
            # Make trading decisions for each ticker
            for ticker in predictions.columns:
                if ticker not in data.columns:
                    continue
                    
                prediction = predictions.loc[date, ticker]
                current_price = current_prices[ticker]
                
                # Skip if we don't have a valid prediction or price
                if pd.isna(prediction) or pd.isna(current_price):
                    continue
                
                # Simple trading logic based on prediction
                if prediction > current_price * 1.01:  # Predicted to go up
                    # Calculate position size (10% of portfolio, limit to 25% per position)
                    position_size = min(0.1, 0.25 / len(predictions.columns))
                    target_value = (self.current_capital + positions_value) * position_size
                    
                    # Don't use more than 50% of available cash
                    max_cash = self.current_capital * 0.5
                    target_value = min(target_value, max_cash)
                    
                    if target_value > 10:  # Minimum trade size
                        quantity = target_value / current_price
                        self.execute_trade(
                            ticker=ticker,
                            action='buy',
                            price=current_price,
                            quantity=quantity,
                            current_prices=current_prices.to_dict()
                        )
                        
                elif prediction < current_price * 0.99:  # Predicted to go down
                    if ticker in self.positions and self.positions[ticker] > 0:
                        self.execute_trade(
                            ticker=ticker,
                            action='sell',
                            price=current_price,
                            quantity=self.positions[ticker],
                            current_prices=current_prices.to_dict()
                        )
        
        # Close any remaining positions at the last price
        if len(common_dates) > 0:
            last_date = common_dates[-1]
            last_prices = data.loc[last_date]
            for ticker in list(self.positions.keys()):
                if ticker in last_prices and self.positions[ticker] > 0:
                    self.execute_trade(
                        ticker=ticker,
                        action='sell',
                        price=last_prices[ticker],
                        quantity=self.positions[ticker],
                        current_prices=last_prices.to_dict()
                    )
        
        # Calculate final metrics
        self._calculate_metrics()
        
    def plot_performance(self):
        """
        Plot backtesting performance
        """
        if not self.trades:
            return
            
        dates = [trade['timestamp'] for trade in self.trades]
        capital_history = []
        current_capital = self.initial_capital
        
        for trade in self.trades:
            if trade['action'] == 'buy':
                current_capital -= trade['price'] * trade['quantity']
            else:
                current_capital += trade['price'] * trade['quantity']
            capital_history.append(current_capital)
            
        plt.figure(figsize=(12, 6))
        plt.plot(dates, capital_history, label='Portfolio Value')
        plt.title('Backtesting Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def get_metrics(self) -> Dict[str, float]:
        """
        Get backtesting metrics
        """
        return self.metrics
        
    def get_trade_history(self) -> List[Dict]:
        """
        Get trade history
        """
        return self.trades

# Example usage:
if __name__ == "__main__":
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=100000.0)
    
    # Example predictions and data
    predictions = pd.DataFrame({
        'AAPL': [150.0, 155.0, 160.0],
        'GOOGL': [2500.0, 2600.0, 2700.0]
    }, index=pd.date_range(start='2023-01-01', periods=3))
    
    data = pd.DataFrame({
        'Close': [145.0, 150.0, 155.0]
    }, index=pd.date_range(start='2023-01-01', periods=3))
    
    # Execute backtest
    engine.evaluate_strategy(predictions, data)
    
    # Show results
    print("\nBacktesting Metrics:")
    print(engine.get_metrics())
    
    # Plot performance
    engine.plot_performance()
