import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

class RiskManager:
    def __init__(self, initial_capital: float = 100000.0, max_risk_per_trade: float = 0.02):
        """
        Initialize risk management system
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.positions: Dict[str, Dict] = {}  # Ticker -> Position details
        self.risk_metrics: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, ticker: str, entry_price: float, stop_loss: float) -> float:
        """
        Calculate optimal position size based on risk management rules
        """
        # Calculate risk per share
        risk_per_share = entry_price - stop_loss
        
        # Calculate maximum shares we can afford to lose
        max_loss = self.current_capital * self.max_risk_per_trade
        max_shares = max_loss / risk_per_share
        
        # Calculate total position value
        position_value = max_shares * entry_price
        
        # Ensure we don't exceed available capital
        max_possible_shares = self.current_capital / entry_price
        shares = min(max_shares, max_possible_shares)
        
        self.logger.info(f"Calculated position size for {ticker}: {shares} shares")
        return shares
        
    def calculate_stop_loss(self, price: float, volatility: float, multiplier: float = 2.0) -> float:
        """
        Calculate stop loss based on volatility
        """
        return price - (volatility * multiplier)
        
    def calculate_take_profit(self, price: float, volatility: float, multiplier: float = 3.0) -> float:
        """
        Calculate take profit based on volatility
        """
        return price + (volatility * multiplier)
        
    def manage_position(self, ticker: str, current_price: float, entry_price: float, volatility: float):
        """
        Manage position based on current market conditions
        """
        if ticker not in self.positions:
            return
            
        position = self.positions[ticker]
        
        # Calculate current risk metrics
        unrealized_pnl = (current_price - entry_price) * position['shares']
        risk_percentage = abs(unrealized_pnl) / self.current_capital
        
        # Update risk metrics
        self.risk_metrics[ticker] = {
            'unrealized_pnl': unrealized_pnl,
            'risk_percentage': risk_percentage,
            'current_price': current_price
        }
        
        # Check for stop loss
        if current_price <= position['stop_loss']:
            self.logger.info(f"Stop loss triggered for {ticker}")
            return 'stop_loss'
            
        # Check for take profit
        if current_price >= position['take_profit']:
            self.logger.info(f"Take profit triggered for {ticker}")
            return 'take_profit'
            
        return None
        
    def calculate_portfolio_risk(self, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics
        """
        total_value = sum(positions.values())
        portfolio_risk = {}
        
        # Calculate individual position risks
        for ticker, value in positions.items():
            portfolio_risk[ticker] = {
                'weight': value / total_value,
                'risk_percentage': value / self.current_capital
            }
        
        # Calculate portfolio concentration
        portfolio_risk['concentration'] = max(
            [info['weight'] for info in portfolio_risk.values()]
        )
        
        return portfolio_risk
        
    def rebalance_portfolio(self, target_weights: Dict[str, float], current_prices: Dict[str, float]):
        """
        Rebalance portfolio based on target weights
        """
        adjustments = {}
        
        # Calculate current portfolio value
        current_value = sum(
            self.positions[ticker]['shares'] * current_prices[ticker]
            for ticker in self.positions
        )
        
        # Calculate target values
        for ticker, weight in target_weights.items():
            target_value = current_value * weight
            current_value = self.positions.get(ticker, {}).get('shares', 0) * current_prices.get(ticker, 0)
            
            if current_value != target_value:
                adjustments[ticker] = {
                    'target_value': target_value,
                    'current_value': current_value,
                    'adjustment_needed': target_value - current_value
                }
                
        return adjustments
        
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics
        """
        return self.risk_metrics
        
    def update_capital(self, new_capital: float):
        """
        Update available capital
        """
        self.current_capital = new_capital
        self.logger.info(f"Updated capital to: {new_capital}")
        
    def add_position(self, ticker: str, shares: float, entry_price: float, volatility: float):
        """
        Add a new position
        """
        stop_loss = self.calculate_stop_loss(entry_price, volatility)
        take_profit = self.calculate_take_profit(entry_price, volatility)
        
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'volatility': volatility
        }
        
        self.logger.info(f"Added position for {ticker}: {shares} shares")
        
    def close_position(self, ticker: str):
        """
        Close a position
        """
        if ticker in self.positions:
            del self.positions[ticker]
            self.logger.info(f"Closed position for {ticker}")

# Example usage:
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager(initial_capital=100000.0)
    
    # Example position management
    ticker = "AAPL"
    entry_price = 150.0
    volatility = 2.5
    
    # Calculate position size
    shares = risk_manager.calculate_position_size(ticker, entry_price, 
        risk_manager.calculate_stop_loss(entry_price, volatility))
    
    # Add position
    risk_manager.add_position(ticker, shares, entry_price, volatility)
    
    # Manage position with updated price
    current_price = 155.0
    action = risk_manager.manage_position(ticker, current_price, entry_price, volatility)
    
    # Get risk metrics
    risk_metrics = risk_manager.get_risk_metrics()
    print("\nRisk Metrics:", risk_metrics)
