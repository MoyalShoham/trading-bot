"""
Virtual exchange for paper trading with simulated market conditions.
"""

import asyncio
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import math

from utils.logger import TradingLogger

logger = TradingLogger(__name__)


class VirtualExchange:
    """
    Virtual exchange that simulates trading conditions for paper trading.
    Provides simulated prices, order execution, and market data.
    """
    
    def __init__(self):
        """Initialize the virtual exchange."""
        self.is_initialized = False
        self.order_counter = 0
        
        # Simulated market data
        self.symbol_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Dict]] = {}
        
        # Market simulation parameters
        self.volatility = 0.02  # 2% daily volatility
        self.trend_factor = 0.0001  # Small trend component
        self.price_update_interval = 1.0  # Update prices every second
        
        # Order book simulation
        self.order_books: Dict[str, Dict] = {}
        
        # Default starting prices for common symbols
        self.default_prices = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 3000.0,
            'ADAUSDT': 0.50,
            'DOTUSDT': 25.0,
            'LINKUSDT': 15.0,
            'SOLUSDT': 100.0,
            'MATICUSDT': 1.2,
            'AVAXUSDT': 35.0,
            'ATOMUSDT': 12.0,
            'NEARUSDT': 8.0,
            'XRPUSDT': 0.60,
            'LTCUSDT': 150.0,
            'UNIUSDT': 8.0,
            'AAVEUSDT': 200.0,
            'SUSHIUSDT': 3.0
        }

    async def initialize(self) -> bool:
        """Initialize the virtual exchange."""
        try:
            # Initialize prices
            for symbol, price in self.default_prices.items():
                self.symbol_prices[symbol] = price
                self.price_history[symbol] = []
                self.order_books[symbol] = {
                    'bids': [],
                    'asks': [],
                    'last_price': price
                }
            
            # Start price simulation
            asyncio.create_task(self._price_simulation_loop())
            
            self.is_initialized = True
            logger.info("Virtual exchange initialized with simulated market data")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing virtual exchange: {e}")
            return False

    async def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if not found
        """
        try:
            if symbol in self.symbol_prices:
                return self.symbol_prices[symbol]
            
            # If symbol not found, initialize with a random price
            if symbol.endswith('USDT'):
                base_price = random.uniform(1, 1000)
                self.symbol_prices[symbol] = base_price
                self.price_history[symbol] = []
                return base_price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = 'market'
    ) -> Optional[Dict[str, Any]]:
        """
        Place an order on the virtual exchange.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Order price
            order_type: Order type ('market' or 'limit')
            
        Returns:
            Order result dictionary
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            self.order_counter += 1
            order_id = f"virtual_{self.order_counter}_{int(datetime.now().timestamp())}"
            
            # Get current market price
            current_price = await self.get_price(symbol)
            if current_price is None:
                logger.error(f"No price available for {symbol}")
                return None
            
            # Simulate order execution
            execution_price = current_price
            
            if order_type == 'market':
                # Market orders execute immediately at current price with some slippage
                slippage_factor = random.uniform(0.9999, 1.0001)  # ±0.01% slippage
                execution_price = current_price * slippage_factor
            else:
                # Limit orders would need order book simulation (simplified here)
                execution_price = price
            
            # Simulate execution delay
            await asyncio.sleep(random.uniform(0.01, 0.1))
            
            order_result = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': execution_price,
                'order_type': order_type,
                'status': 'filled',
                'timestamp': datetime.now(),
                'commission': quantity * execution_price * 0.0004  # 0.04% commission
            }
            
            logger.debug(f"Virtual order executed: {side} {quantity} {symbol} @ {execution_price}")
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error placing virtual order: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an order (simplified for virtual exchange).
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            # In a real implementation, this would remove the order from order books
            logger.debug(f"Virtual order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling virtual order: {e}")
            return False

    async def get_order_book(self, symbol: str, limit: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of levels to return
            
        Returns:
            Order book data
        """
        try:
            if symbol not in self.order_books:
                return None
            
            # Generate simulated order book
            current_price = await self.get_price(symbol)
            if current_price is None:
                return None
            
            bids = []
            asks = []
            
            # Generate bid levels (below current price)
            for i in range(limit):
                price_offset = (i + 1) * current_price * 0.001  # 0.1% increments
                bid_price = current_price - price_offset
                bid_quantity = random.uniform(0.1, 10.0)
                bids.append([bid_price, bid_quantity])
            
            # Generate ask levels (above current price)
            for i in range(limit):
                price_offset = (i + 1) * current_price * 0.001  # 0.1% increments
                ask_price = current_price + price_offset
                ask_quantity = random.uniform(0.1, 10.0)
                asks.append([ask_price, ask_quantity])
            
            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return None

    async def get_klines(
        self,
        symbol: str,
        interval: str = '1m',
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical kline/candlestick data.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            limit: Number of klines to return
            
        Returns:
            List of kline data
        """
        try:
            current_price = await self.get_price(symbol)
            if current_price is None:
                return []
            
            klines = []
            base_time = datetime.now() - timedelta(minutes=limit)
            
            # Generate simulated kline data
            price = current_price * 0.99  # Start slightly below current price
            
            for i in range(limit):
                # Simulate price movement
                change_pct = random.gauss(0, self.volatility / math.sqrt(1440))  # Daily vol scaled to minute
                price *= (1 + change_pct)
                
                # Generate OHLC
                open_price = price
                close_price = price * (1 + random.gauss(0, self.volatility / math.sqrt(1440)))
                high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.001)))
                low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.001)))
                volume = random.uniform(100, 10000)
                
                kline_time = base_time + timedelta(minutes=i)
                
                klines.append({
                    'timestamp': int(kline_time.timestamp() * 1000),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                
                price = close_price
            
            return klines
            
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []

    async def _price_simulation_loop(self) -> None:
        """Continuously update simulated prices."""
        while True:
            try:
                await asyncio.sleep(self.price_update_interval)
                
                for symbol in self.symbol_prices:
                    try:
                        await self._update_symbol_price(symbol)
                    except Exception as e:
                        logger.debug(f"Error updating price for {symbol}: {e}")
                        
            except Exception as e:
                logger.error(f"Error in price simulation loop: {e}")
                await asyncio.sleep(5)

    async def _update_symbol_price(self, symbol: str) -> None:
        """Update price for a single symbol."""
        try:
            current_price = self.symbol_prices[symbol]
            
            # Generate price movement using random walk with small trend
            dt = self.price_update_interval / 86400  # Convert to fraction of day
            
            # Random component
            random_component = random.gauss(0, self.volatility * math.sqrt(dt))
            
            # Small trend component (can be positive or negative)
            trend_component = self.trend_factor * dt * random.choice([-1, 1])
            
            # Apply change
            price_change = current_price * (random_component + trend_component)
            new_price = current_price + price_change
            
            # Ensure price doesn't go negative or too extreme
            new_price = max(new_price, current_price * 0.9)  # Max 10% drop
            new_price = min(new_price, current_price * 1.1)  # Max 10% rise
            
            self.symbol_prices[symbol] = new_price
            
            # Store price history (keep last 1000 points)
            price_point = {
                'timestamp': datetime.now(),
                'price': new_price
            }
            
            self.price_history[symbol].append(price_point)
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
                
        except Exception as e:
            logger.error(f"Error updating price for {symbol}: {e}")

    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get price history for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of price points to return
            
        Returns:
            List of price history points
        """
        try:
            if symbol not in self.price_history:
                return []
            
            return self.price_history[symbol][-limit:]
            
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []

    def add_symbol(self, symbol: str, initial_price: float) -> bool:
        """
        Add a new symbol to the virtual exchange.
        
        Args:
            symbol: Trading symbol
            initial_price: Initial price for the symbol
            
        Returns:
            True if added successfully
        """
        try:
            if symbol in self.symbol_prices:
                logger.warning(f"Symbol {symbol} already exists")
                return False
            
            self.symbol_prices[symbol] = initial_price
            self.price_history[symbol] = []
            self.order_books[symbol] = {
                'bids': [],
                'asks': [],
                'last_price': initial_price
            }
            
            logger.info(f"Added new symbol {symbol} with initial price {initial_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            return False

    def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary with all symbols."""
        try:
            summary = {
                'symbols': len(self.symbol_prices),
                'prices': self.symbol_prices.copy(),
                'timestamp': datetime.now()
            }
            
            # Calculate some basic market stats
            if self.symbol_prices:
                prices = list(self.symbol_prices.values())
                summary['avg_price'] = sum(prices) / len(prices)
                summary['min_price'] = min(prices)
                summary['max_price'] = max(prices)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {}
