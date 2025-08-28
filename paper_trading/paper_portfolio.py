"""
Paper trading portfolio management with position tracking and PnL calculation.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from utils.logger import TradingLogger

logger = TradingLogger(__name__)


@dataclass
class PaperPosition:
    """Represents a paper trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0


class PaperPortfolio:
    """
    Manages paper trading portfolio with position tracking,
    PnL calculation, and balance management.
    """
    
    def __init__(self, initial_balance: float, commission_rate: float = 0.0004, slippage_rate: float = 0.0005):
        """
        Initialize paper portfolio.
        
        Args:
            initial_balance: Starting balance
            commission_rate: Commission rate for trades
            slippage_rate: Slippage rate for trades
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Position tracking
        self.positions: Dict[str, PaperPosition] = {}
        
        # Performance tracking
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.peak_value = initial_balance
        self.max_drawdown = 0.0
        
        # Trade history
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def get_balance(self) -> float:
        """Get current cash balance."""
        return self.balance

    def adjust_balance(self, amount: float) -> None:
        """Adjust the cash balance."""
        self.balance += amount
        logger.debug(f"Balance adjusted by {amount:.2f}, new balance: {self.balance:.2f}")

    def add_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Add a new position to the portfolio.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            quantity: Position quantity
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if position added successfully
        """
        try:
            if symbol in self.positions:
                logger.warning(f"Position already exists for {symbol}")
                return False
            
            position = PaperPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.positions[symbol] = position
            
            logger.info(f"Added {side} position: {quantity} {symbol} @ {entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position for {symbol}: {e}")
            return False

    def close_position(self, symbol: str) -> Optional[PaperPosition]:
        """
        Close a position.
        
        Args:
            symbol: Symbol to close
            
        Returns:
            Closed position or None if not found
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return None
            
            position = self.positions.pop(symbol)
            
            # Update trade statistics
            self.trade_count += 1
            if position.unrealized_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            logger.info(f"Closed {position.side} position for {symbol}, PnL: {position.unrealized_pnl:.2f}")
            return position
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return None

    def update_position_price(self, symbol: str, current_price: float) -> None:
        """
        Update the current price for a position and recalculate PnL.
        
        Args:
            symbol: Symbol to update
            current_price: New current price
        """
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            position.current_price = current_price
            
            # Calculate unrealized PnL
            if position.side == 'long':
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:  # short
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            
        except Exception as e:
            logger.error(f"Error updating position price for {symbol}: {e}")

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position details for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position dictionary or None if not found
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        return {
            'symbol': position.symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'entry_time': position.entry_time,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'unrealized_pnl': position.unrealized_pnl,
            'unrealized_pnl_pct': position.unrealized_pnl / (position.entry_price * position.quantity) if position.quantity > 0 else 0
        }

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all positions as dictionary."""
        positions_dict = {}
        
        for symbol in self.positions:
            position_data = self.get_position(symbol)
            if position_data:
                positions_dict[symbol] = position_data
        
        return positions_dict

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized PnL across all positions."""
        return sum(position.unrealized_pnl for position in self.positions.values())

    def get_total_value(self) -> float:
        """Get total portfolio value (cash + unrealized PnL)."""
        return self.balance + self.get_unrealized_pnl()

    def get_used_margin(self) -> float:
        """Calculate margin used by open positions."""
        used_margin = 0.0
        
        for position in self.positions.values():
            position_value = position.quantity * position.current_price
            used_margin += position_value
        
        return used_margin

    def get_available_margin(self) -> float:
        """Calculate available margin for new positions."""
        return max(0, self.balance - self.get_used_margin() * 0.1)  # Assume 10% margin requirement

    def calculate_drawdown(self) -> Dict[str, float]:
        """Calculate current drawdown metrics."""
        current_value = self.get_total_value()
        
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate drawdown
        drawdown = self.peak_value - current_value
        drawdown_pct = drawdown / self.peak_value if self.peak_value > 0 else 0
        
        # Update max drawdown
        if drawdown_pct > self.max_drawdown:
            self.max_drawdown = drawdown_pct
        
        return {
            'current_drawdown': drawdown,
            'current_drawdown_pct': drawdown_pct,
            'max_drawdown': self.max_drawdown * self.peak_value,
            'max_drawdown_pct': self.max_drawdown,
            'peak_value': self.peak_value
        }

    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics."""
        try:
            current_value = self.get_total_value()
            total_return = current_value - self.initial_balance
            total_return_pct = total_return / self.initial_balance
            
            drawdown_metrics = self.calculate_drawdown()
            
            # Win rate calculation
            win_rate = self.winning_trades / self.trade_count if self.trade_count > 0 else 0
            
            # Position statistics
            open_positions = len(self.positions)
            long_positions = len([p for p in self.positions.values() if p.side == 'long'])
            short_positions = len([p for p in self.positions.values() if p.side == 'short'])
            
            # PnL statistics
            unrealized_pnl = self.get_unrealized_pnl()
            winning_positions = len([p for p in self.positions.values() if p.unrealized_pnl > 0])
            losing_positions = len([p for p in self.positions.values() if p.unrealized_pnl < 0])
            
            return {
                'balance': {
                    'initial_balance': self.initial_balance,
                    'current_balance': self.balance,
                    'total_value': current_value,
                    'unrealized_pnl': unrealized_pnl,
                    'total_return': total_return,
                    'total_return_pct': total_return_pct
                },
                'drawdown': drawdown_metrics,
                'positions': {
                    'open_positions': open_positions,
                    'long_positions': long_positions,
                    'short_positions': short_positions,
                    'winning_positions': winning_positions,
                    'losing_positions': losing_positions
                },
                'trading': {
                    'total_trades': self.trade_count,
                    'winning_trades': self.winning_trades,
                    'losing_trades': self.losing_trades,
                    'win_rate': win_rate,
                    'total_commission': self.total_commission_paid,
                    'total_slippage': self.total_slippage_cost
                },
                'margin': {
                    'used_margin': self.get_used_margin(),
                    'available_margin': self.get_available_margin(),
                    'margin_utilization': self.get_used_margin() / self.balance if self.balance > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio statistics: {e}")
            return {}

    def reset(self, new_balance: Optional[float] = None) -> None:
        """
        Reset the portfolio to initial state.
        
        Args:
            new_balance: New initial balance (optional)
        """
        try:
            if new_balance is not None:
                self.initial_balance = new_balance
            
            self.balance = self.initial_balance
            self.positions.clear()
            
            # Reset performance tracking
            self.total_commission_paid = 0.0
            self.total_slippage_cost = 0.0
            self.peak_value = self.initial_balance
            self.max_drawdown = 0.0
            
            # Reset trade statistics
            self.trade_count = 0
            self.winning_trades = 0
            self.losing_trades = 0
            
            logger.info(f"Portfolio reset with balance: ${self.initial_balance:,.2f}")
            
        except Exception as e:
            logger.error(f"Error resetting portfolio: {e}")

    def validate_order(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """
        Validate if an order can be placed.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Order price
            
        Returns:
            True if order is valid
        """
        try:
            # Check if position already exists
            if symbol in self.positions:
                logger.warning(f"Position already exists for {symbol}")
                return False
            
            # Check if we have enough balance
            order_value = quantity * price
            required_margin = order_value * 0.1  # Assume 10% margin requirement
            
            if required_margin > self.get_available_margin():
                logger.warning(f"Insufficient margin for order: required {required_margin:.2f}, available {self.get_available_margin():.2f}")
                return False
            
            # Basic validation
            if quantity <= 0 or price <= 0:
                logger.warning("Invalid quantity or price")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False

    def export_positions(self) -> List[Dict[str, Any]]:
        """Export all positions as a list of dictionaries."""
        try:
            positions_list = []
            
            for symbol, position in self.positions.items():
                position_data = self.get_position(symbol)
                if position_data:
                    positions_list.append(position_data)
            
            return positions_list
            
        except Exception as e:
            logger.error(f"Error exporting positions: {e}")
            return []
