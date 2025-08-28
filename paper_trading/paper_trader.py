"""
Paper trading implementation that simulates real trading without actual money.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from config import config
from core.exchange import BinanceExchange
from paper_trading.paper_portfolio import PaperPortfolio
from paper_trading.virtual_exchange import VirtualExchange
from utils.logger import TradingLogger
from risk.risk_calculator import RiskCalculator

logger = TradingLogger(__name__)


@dataclass
class PaperTradeResult:
    """Result of a paper trade execution."""
    success: bool
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    message: str = ""


class PaperTrader:
    """
    Paper trading implementation that simulates real trading conditions
    without using real money. Tracks performance and provides realistic
    execution simulation including slippage and commission.
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.0004,
        slippage_rate: float = 0.0005,
        use_real_prices: bool = True
    ):
        """
        Initialize paper trader.
        
        Args:
            initial_balance: Starting virtual balance
            commission_rate: Trading commission rate
            slippage_rate: Slippage simulation rate
            use_real_prices: Whether to use real market prices
        """
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.use_real_prices = use_real_prices
        
        # Initialize components
        self.portfolio = PaperPortfolio(initial_balance, commission_rate, slippage_rate)
        self.virtual_exchange = VirtualExchange()
        self.risk_calculator = RiskCalculator()
        
        # Real exchange for price data (if enabled)
        self.real_exchange: Optional[BinanceExchange] = None
        
        # Trading state
        self.is_active = False
        self.start_time = datetime.now()
        self.trade_history: List[Dict] = []
        self.performance_log: List[Dict] = []
        
        # Price monitoring
        self.current_prices: Dict[str, float] = {}
        self.price_update_interval = 5  # seconds

    async def initialize(self) -> bool:
        """Initialize the paper trader."""
        try:
            # Initialize virtual exchange
            await self.virtual_exchange.initialize()
            
            # Initialize real exchange if using real prices
            if self.use_real_prices:
                self.real_exchange = BinanceExchange()
                await self.real_exchange.initialize()
                
                # Test connection
                if not await self.real_exchange.ping():
                    logger.warning("Real exchange connection failed, falling back to simulated prices")
                    self.use_real_prices = False
            
            logger.info(f"Paper trader initialized with ${self.initial_balance:,.2f} virtual balance")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize paper trader: {e}")
            return False

    async def start(self) -> None:
        """Start paper trading mode."""
        try:
            if not await self.initialize():
                raise RuntimeError("Failed to initialize paper trader")
            
            self.is_active = True
            self.start_time = datetime.now()
            
            logger.info("Paper trading mode started")
            
            # Start price monitoring task
            if self.use_real_prices:
                asyncio.create_task(self._price_monitoring_loop())
            
        except Exception as e:
            logger.error(f"Error starting paper trader: {e}")
            raise

    async def stop(self) -> None:
        """Stop paper trading mode."""
        try:
            self.is_active = False
            
            # Close real exchange connection
            if self.real_exchange:
                await self.real_exchange.close()
            
            # Generate final report
            await self._generate_session_report()
            
            logger.info("Paper trading mode stopped")
            
        except Exception as e:
            logger.error(f"Error stopping paper trader: {e}")

    async def execute_signal(self, signal: Dict[str, Any]) -> Optional[Dict]:
        """
        Execute a trading signal in paper trading mode.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Execution result or None if failed
        """
        try:
            if not self.is_active:
                logger.warning("Paper trader is not active")
                return None
            
            symbol = signal.get('symbol')
            signal_type = signal.get('signal')
            
            if not symbol or signal_type == 'no-trade':
                return None
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            if current_price is None:
                logger.error(f"Could not get current price for {symbol}")
                return None
            
            # Calculate position size using risk management
            position_size = await self._calculate_position_size(signal, current_price)
            if position_size <= 0:
                logger.warning(f"Invalid position size calculated for {symbol}")
                return None
            
            # Determine order side
            if signal_type == 'long_bias':
                side = 'buy'
            elif signal_type == 'short_bias':
                side = 'sell'
            else:
                logger.warning(f"Unknown signal type: {signal_type}")
                return None
            
            # Execute the trade
            trade_result = await self._execute_trade(
                symbol=symbol,
                side=side,
                quantity=position_size,
                price=current_price,
                signal=signal
            )
            
            if trade_result and trade_result.success:
                # Log the trade
                trade_data = {
                    'timestamp': trade_result.timestamp.isoformat(),
                    'symbol': symbol,
                    'side': side,
                    'quantity': position_size,
                    'price': current_price,
                    'commission': trade_result.commission,
                    'slippage': trade_result.slippage,
                    'signal_confidence': signal.get('confidence', 0),
                    'signal_reason': signal.get('reason', ''),
                    'balance_after': self.portfolio.get_balance()
                }
                
                self.trade_history.append(trade_data)
                logger.log_trade(trade_data)
                
                # Update performance tracking
                await self._update_performance_tracking()
                
                return {
                    'success': True,
                    'trade_result': asdict(trade_result),
                    'portfolio_value': self.portfolio.get_total_value(),
                    'unrealized_pnl': self.portfolio.get_unrealized_pnl()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing signal in paper trading: {e}")
            return None

    async def _execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        signal: Dict[str, Any]
    ) -> Optional[PaperTradeResult]:
        """Execute a single trade."""
        try:
            # Check if we already have a position
            current_position = self.portfolio.get_position(symbol)
            
            # For now, only allow one position per symbol
            if current_position:
                logger.info(f"Position already exists for {symbol}, skipping trade")
                return None
            
            # Calculate stop loss and take profit
            stop_loss_price = None
            take_profit_price = None
            
            if side == 'buy':
                stop_loss_price = price * (1 - config.STOP_LOSS_PERCENT)
                take_profit_price = price * (1 + config.TAKE_PROFIT_PERCENT)
            else:  # sell
                stop_loss_price = price * (1 + config.STOP_LOSS_PERCENT)
                take_profit_price = price * (1 - config.TAKE_PROFIT_PERCENT)
            
            # Execute in virtual exchange
            order_result = await self.virtual_exchange.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type='market'
            )
            
            if not order_result:
                return PaperTradeResult(
                    success=False,
                    order_id="",
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    timestamp=datetime.now(),
                    commission=0,
                    slippage=0,
                    message="Virtual exchange order failed"
                )
            
            # Calculate commission and slippage
            trade_value = quantity * price
            commission = trade_value * self.commission_rate
            
            # Apply slippage
            slippage_amount = 0
            execution_price = price
            
            if side == 'buy':
                execution_price = price * (1 + self.slippage_rate)
                slippage_amount = (execution_price - price) * quantity
            else:
                execution_price = price * (1 - self.slippage_rate)
                slippage_amount = (price - execution_price) * quantity
            
            # Update portfolio
            position_side = 'long' if side == 'buy' else 'short'
            
            self.portfolio.add_position(
                symbol=symbol,
                side=position_side,
                quantity=quantity,
                entry_price=execution_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price
            )
            
            # Deduct commission and slippage from balance
            self.portfolio.adjust_balance(-(commission + slippage_amount))
            
            return PaperTradeResult(
                success=True,
                order_id=order_result.get('order_id', 'paper_trade'),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=execution_price,
                timestamp=datetime.now(),
                commission=commission,
                slippage=slippage_amount,
                message="Trade executed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            if self.use_real_prices and self.real_exchange:
                price = await self.real_exchange.get_symbol_price(symbol)
                if price:
                    self.current_prices[symbol] = price
                    return price
            
            # Fallback to virtual exchange
            return await self.virtual_exchange.get_price(symbol)
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    async def _calculate_position_size(self, signal: Dict[str, Any], current_price: float) -> float:
        """Calculate position size using risk management."""
        try:
            symbol = signal['symbol']
            confidence = signal.get('confidence', 0.5)
            
            # Calculate stop loss price
            if signal.get('signal') == 'long_bias':
                stop_loss_price = current_price * (1 - config.STOP_LOSS_PERCENT)
            else:
                stop_loss_price = current_price * (1 + config.STOP_LOSS_PERCENT)
            
            # Use risk calculator
            risk_metrics = self.risk_calculator.calculate_position_size(
                symbol=symbol,
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                account_balance=self.portfolio.get_balance(),
                confidence=confidence
            )
            
            return risk_metrics.position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    async def _price_monitoring_loop(self) -> None:
        """Monitor prices and update portfolio."""
        while self.is_active:
            try:
                symbols = list(self.portfolio.get_positions().keys())
                
                if symbols and self.real_exchange:
                    # Update prices for symbols with positions
                    for symbol in symbols:
                        try:
                            price = await self.real_exchange.get_symbol_price(symbol)
                            if price:
                                self.current_prices[symbol] = price
                                self.portfolio.update_position_price(symbol, price)
                        except Exception as e:
                            logger.debug(f"Error updating price for {symbol}: {e}")
                
                # Check stop loss and take profit
                await self._check_exit_conditions()
                
                await asyncio.sleep(self.price_update_interval)
                
            except Exception as e:
                logger.error(f"Error in price monitoring loop: {e}")
                await asyncio.sleep(self.price_update_interval)

    async def _check_exit_conditions(self) -> None:
        """Check stop loss and take profit conditions."""
        try:
            positions = self.portfolio.get_positions()
            
            for symbol, position in positions.items():
                current_price = self.current_prices.get(symbol)
                if not current_price:
                    continue
                
                should_close = False
                exit_reason = ""
                
                # Check stop loss
                if position['stop_loss']:
                    if position['side'] == 'long' and current_price <= position['stop_loss']:
                        should_close = True
                        exit_reason = "stop_loss"
                    elif position['side'] == 'short' and current_price >= position['stop_loss']:
                        should_close = True
                        exit_reason = "stop_loss"
                
                # Check take profit
                if position['take_profit'] and not should_close:
                    if position['side'] == 'long' and current_price >= position['take_profit']:
                        should_close = True
                        exit_reason = "take_profit"
                    elif position['side'] == 'short' and current_price <= position['take_profit']:
                        should_close = True
                        exit_reason = "take_profit"
                
                if should_close:
                    await self._close_position(symbol, current_price, exit_reason)
                    
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")

    async def _close_position(self, symbol: str, price: float, reason: str) -> None:
        """Close a position."""
        try:
            position = self.portfolio.get_position(symbol)
            if not position:
                return
            
            # Calculate PnL
            if position['side'] == 'long':
                pnl = (price - position['entry_price']) * position['quantity']
                side = 'sell'
            else:
                pnl = (position['entry_price'] - price) * position['quantity']
                side = 'buy'
            
            # Calculate commission
            trade_value = position['quantity'] * price
            commission = trade_value * self.commission_rate
            net_pnl = pnl - commission
            
            # Update portfolio
            self.portfolio.close_position(symbol)
            self.portfolio.adjust_balance(net_pnl)
            
            # Log the closure
            close_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'action': 'close',
                'quantity': position['quantity'],
                'price': price,
                'pnl': net_pnl,
                'commission': commission,
                'reason': reason,
                'balance_after': self.portfolio.get_balance()
            }
            
            self.trade_history.append(close_data)
            logger.log_trade(close_data)
            
            logger.info(f"Closed {position['side']} position for {symbol} due to {reason}, PnL: {net_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    async def _update_performance_tracking(self) -> None:
        """Update performance tracking metrics."""
        try:
            current_value = self.portfolio.get_total_value()
            
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'balance': self.portfolio.get_balance(),
                'total_value': current_value,
                'unrealized_pnl': self.portfolio.get_unrealized_pnl(),
                'total_return': (current_value - self.initial_balance) / self.initial_balance,
                'open_positions': len(self.portfolio.get_positions())
            }
            
            self.performance_log.append(performance_data)
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")

    async def _generate_session_report(self) -> None:
        """Generate a session performance report."""
        try:
            end_time = datetime.now()
            session_duration = end_time - self.start_time
            
            final_value = self.portfolio.get_total_value()
            total_return = final_value - self.initial_balance
            total_return_pct = total_return / self.initial_balance
            
            # Trade statistics
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            report = {
                'session_start': self.start_time.isoformat(),
                'session_end': end_time.isoformat(),
                'duration': str(session_duration),
                'initial_balance': self.initial_balance,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'open_positions': len(self.portfolio.get_positions())
            }
            
            # Save report
            report_file = f"paper_trading_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Paper trading session completed: {total_return_pct:.2%} return, {total_trades} trades")
            
        except Exception as e:
            logger.error(f"Error generating session report: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        try:
            current_value = self.portfolio.get_total_value()
            
            return {
                'initial_balance': self.initial_balance,
                'current_balance': self.portfolio.get_balance(),
                'total_value': current_value,
                'unrealized_pnl': self.portfolio.get_unrealized_pnl(),
                'total_return': current_value - self.initial_balance,
                'total_return_pct': (current_value - self.initial_balance) / self.initial_balance,
                'open_positions': len(self.portfolio.get_positions()),
                'positions': self.portfolio.get_positions(),
                'is_active': self.is_active,
                'session_duration': str(datetime.now() - self.start_time) if self.is_active else None,
                'total_trades': len(self.trade_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}

    async def force_close_all_positions(self) -> None:
        """Force close all open positions."""
        try:
            positions = list(self.portfolio.get_positions().items())
            
            for symbol, position in positions:
                current_price = await self._get_current_price(symbol)
                if current_price:
                    await self._close_position(symbol, current_price, "force_close")
            
            logger.info(f"Force closed {len(positions)} positions")
            
        except Exception as e:
            logger.error(f"Error force closing positions: {e}")
