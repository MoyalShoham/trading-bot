"""
Portfolio simulator for backtesting with realistic trading mechanics.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from utils.logger import TradingLogger

logger = TradingLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    slippage: float
    duration: timedelta


class PortfolioSimulator:
    """
    Simulates a trading portfolio with realistic execution mechanics.
    Includes commission, slippage, margin trading, and position management.
    """
    
    def __init__(
        self,
        initial_balance: float,
        commission: float = 0.0004,
        slippage: float = 0.0005,
        leverage: int = 1
    ):
        """
        Initialize the portfolio simulator.
        
        Args:
            initial_balance: Starting balance
            commission: Commission rate (e.g., 0.0004 = 0.04%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
            leverage: Maximum leverage
        """
        self.initial_balance = initial_balance
        self.commission_rate = commission
        self.slippage_rate = slippage
        self.max_leverage = leverage
        
        # Portfolio state
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[Trade] = []
        self.current_prices: Dict[str, float] = {}
        
        # Metrics tracking
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.peak_equity = initial_balance
        self.max_drawdown = 0.0

    def reset(self, initial_balance: Optional[float] = None) -> None:
        """Reset the portfolio to initial state."""
        if initial_balance is not None:
            self.initial_balance = initial_balance
            
        self.balance = self.initial_balance
        self.positions.clear()
        self.completed_trades.clear()
        self.current_prices.clear()
        
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current market prices."""
        self.current_prices.update(prices)
        
        # Update unrealized PnL for open positions
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                
                if position.side == 'long':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:  # short
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
        
        # Update peak equity and drawdown
        current_equity = self.total_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    @property
    def total_equity(self) -> float:
        """Calculate total portfolio equity including unrealized PnL."""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.balance + unrealized_pnl

    @property
    def margin_used(self) -> float:
        """Calculate total margin used by open positions."""
        return sum(pos.margin_used for pos in self.positions.values())

    @property
    def available_margin(self) -> float:
        """Calculate available margin for new positions."""
        return max(0, self.balance - self.margin_used)

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a symbol."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            return {
                'symbol': pos.symbol,
                'side': pos.side,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'margin_used': pos.margin_used
            }
        return None

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized PnL."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def execute_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: float,
        price: float,
        timestamp: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a trading order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Order price
            timestamp: Execution timestamp
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Trade execution result or None if failed
        """
        try:
            # Apply slippage
            if side == 'buy':
                execution_price = price * (1 + self.slippage_rate)
            else:
                execution_price = price * (1 - self.slippage_rate)
            
            # Calculate trade value and commission
            trade_value = quantity * execution_price
            commission = trade_value * self.commission_rate
            slippage_cost = abs(execution_price - price) * quantity
            
            # Check if we're opening or closing a position
            existing_position = self.positions.get(symbol)
            
            if existing_position is None:
                # Opening new position
                return self._open_position(
                    symbol, side, quantity, execution_price, timestamp,
                    commission, slippage_cost, stop_loss, take_profit
                )
            else:
                # Closing existing position
                return self._close_position(
                    symbol, side, quantity, execution_price, timestamp,
                    commission, slippage_cost
                )
                
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return None

    def _open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        commission: float,
        slippage_cost: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Open a new position."""
        try:
            # Calculate required margin
            position_value = quantity * price
            required_margin = position_value / self.max_leverage
            
            # Check if we have enough available margin
            if required_margin > self.available_margin:
                logger.warning(f"Insufficient margin for {symbol}: need {required_margin}, have {self.available_margin}")
                return None
            
            # Deduct commission and slippage from balance
            total_cost = commission + slippage_cost
            if total_cost > self.balance:
                logger.warning(f"Insufficient balance for costs: {total_cost}")
                return None
            
            self.balance -= total_cost
            self.total_commission_paid += commission
            self.total_slippage_cost += slippage_cost
            
            # Create position
            position_side = 'long' if side == 'buy' else 'short'
            position = Position(
                symbol=symbol,
                side=position_side,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit,
                margin_used=required_margin
            )
            
            self.positions[symbol] = position
            
            logger.debug(f"Opened {position_side} position: {quantity} {symbol} @ {price}")
            
            return {
                'action': 'open',
                'symbol': symbol,
                'side': position_side,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'slippage': slippage_cost,
                'margin_used': required_margin
            }
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None

    def _close_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        commission: float,
        slippage_cost: float
    ) -> Optional[Dict[str, Any]]:
        """Close an existing position."""
        try:
            position = self.positions[symbol]
            
            # Check if this is actually closing the position
            # (buy closes short, sell closes long)
            if (position.side == 'long' and side != 'sell') or \
               (position.side == 'short' and side != 'buy'):
                logger.warning(f"Order side {side} doesn't close {position.side} position")
                return None
            
            # Calculate PnL
            if position.side == 'long':
                pnl = (price - position.entry_price) * quantity
            else:  # short
                pnl = (position.entry_price - price) * quantity
            
            # Adjust for commission and slippage
            net_pnl = pnl - commission - slippage_cost
            
            # Update balance
            self.balance += net_pnl + position.margin_used
            self.total_commission_paid += commission
            self.total_slippage_cost += slippage_cost
            
            # Create completed trade record
            trade = Trade(
                symbol=symbol,
                side=position.side,
                quantity=quantity,
                entry_price=position.entry_price,
                exit_price=price,
                entry_time=position.entry_time,
                exit_time=timestamp,
                pnl=net_pnl,
                commission=commission,
                slippage=slippage_cost,
                duration=timestamp - position.entry_time
            )
            
            self.completed_trades.append(trade)
            
            # Remove position
            del self.positions[symbol]
            
            logger.debug(f"Closed {position.side} position: {quantity} {symbol} @ {price}, PnL: {net_pnl:.2f}")
            
            return {
                'action': 'close',
                'symbol': symbol,
                'side': position.side,
                'quantity': quantity,
                'price': price,
                'pnl': net_pnl,
                'commission': commission,
                'slippage': slippage_cost,
                'trade_duration': trade.duration
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None

    def check_stop_loss_take_profit(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """Check and execute stop loss and take profit orders."""
        executed_orders = []
        
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in self.current_prices:
                continue
                
            current_price = self.current_prices[symbol]
            should_close = False
            exit_reason = ""
            
            # Check stop loss
            if position.stop_loss is not None:
                if position.side == 'long' and current_price <= position.stop_loss:
                    should_close = True
                    exit_reason = "stop_loss"
                elif position.side == 'short' and current_price >= position.stop_loss:
                    should_close = True
                    exit_reason = "stop_loss"
            
            # Check take profit
            if position.take_profit is not None and not should_close:
                if position.side == 'long' and current_price >= position.take_profit:
                    should_close = True
                    exit_reason = "take_profit"
                elif position.side == 'short' and current_price <= position.take_profit:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((symbol, exit_reason, current_price))
        
        # Execute closes
        for symbol, reason, price in positions_to_close:
            position = self.positions[symbol]
            side = 'sell' if position.side == 'long' else 'buy'
            
            result = self.execute_order(
                symbol=symbol,
                side=side,
                quantity=position.quantity,
                price=price,
                timestamp=timestamp
            )
            
            if result:
                result['exit_reason'] = reason
                executed_orders.append(result)
        
        return executed_orders

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary with key metrics."""
        total_trades = len(self.completed_trades)
        winning_trades = len([t for t in self.completed_trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_win = np.mean([t.pnl for t in self.completed_trades if t.pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t.pnl for t in self.completed_trades if t.pnl < 0]) if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'total_equity': self.total_equity,
            'unrealized_pnl': self.get_unrealized_pnl(),
            'total_return': self.total_equity - self.initial_balance,
            'total_return_pct': (self.total_equity - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.positions),
            'margin_used': self.margin_used,
            'available_margin': self.available_margin,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commission': self.total_commission_paid,
            'total_slippage': self.total_slippage_cost
        }
