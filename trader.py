"""
Trader module for the crypto trading bot.
Connects to Binance Futures API for placing/canceling/monitoring orders.
Always checks balance, margin, and open positions.
Implements risk management: stop-loss, take-profit, max leverage.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from config import config
from logger import logger
from symbol_precision import round_quantity

class BinanceTrader:
    async def initialize(self):
        """Ensure aiohttp session is initialized."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
    """Binance Futures trading interface with risk management."""
    
    def __init__(self):
        """Initialize the Binance trader."""
        self.api_key = config.BINANCE_API_KEY
        self.secret_key = config.BINANCE_SECRET_KEY
        self.base_url = config.BINANCE_BASE_URL
        self.testnet = config.BINANCE_TESTNET
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Risk management settings
        self.max_leverage = config.MAX_LEVERAGE
        self.risk_per_trade = config.RISK_PER_TRADE
        self.stop_loss_percent = config.STOP_LOSS_PERCENT
        self.take_profit_percent = config.TAKE_PROFIT_PERCENT
        self.max_position_size = config.MAX_POSITION_SIZE
        self.min_balance_threshold = config.MIN_BALANCE_THRESHOLD
        
        # Trading state
        self.positions = {}
        self.orders = {}
        self.balance = 0.0
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, params: str) -> str:
        """
        Generate HMAC SHA256 signature for Binance API.
        
        Args:
            params: Query parameters string
            
        Returns:
            HMAC signature
        """
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for authenticated requests."""
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None, 
        signed: bool = False
    ) -> Optional[Dict]:
        """
        Make HTTP request to Binance API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            signed: Whether to sign the request
            
        Returns:
            API response or None if failed
        """
        if not self.session:
            logger.log_error("Session not initialized")
            return None
        
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        # Add timestamp for signed requests
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
        
        # Convert params to query string
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        
        # Generate signature if needed
        if signed:
            signature = self._generate_signature(query_string)
            query_string += f"&signature={signature}"
        
        headers = self._get_headers() if signed else {}
        
        try:
            if method.upper() == 'GET':
                if query_string:
                    url += f"?{query_string}"
                async with self.session.get(url, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                async with self.session.post(url, data=query_string, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'DELETE':
                if query_string:
                    url += f"?{query_string}"
                async with self.session.delete(url, headers=headers) as response:
                    return await self._handle_response(response)
            else:
                logger.log_error(f"Unsupported HTTP method: {method}")
                return None
                
        except Exception as e:
            logger.log_error(f"Request error: {str(e)}")
            return None
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Optional[Dict]:
        """Handle API response."""
        try:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.log_error(f"API error {response.status}: {error_text}")
                return None
        except Exception as e:
            logger.log_error(f"Error handling response: {str(e)}")
            return None
    
    async def get_account_info(self) -> Optional[Dict]:
        """Get account information including balances and positions."""
        try:
            response = await self._make_request('GET', '/fapi/v2/account', signed=True)
            if response:
                # Update balance
                for asset in response.get('assets', []):
                    if asset['asset'] == 'USDT':
                        self.balance = float(asset['walletBalance'])
                        break
                
                # Update positions
                self.positions = {}
                for position in response.get('positions', []):
                    if float(position['positionAmt']) != 0:
                        self.positions[position['symbol']] = {
                            'symbol': position['symbol'],
                            'side': 'long' if float(position['positionAmt']) > 0 else 'short',
                            'size': abs(float(position['positionAmt'])),
                            'entry_price': float(position['entryPrice']),
                            'unrealized_pnl': float(position.get('unRealizedProfit', 0.0)),
                            'leverage': int(position['leverage'])
                        }
                
                logger.log_info(f"Account info updated: Balance=${self.balance:.2f}, Positions={len(self.positions)}")
                return response
            return None
            
        except Exception as e:
            logger.log_error(f"Error getting account info: {str(e)}")
            return None
    
    async def get_symbol_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            response = await self._make_request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
            if response:
                return float(response['price'])
            return None
            
        except Exception as e:
            logger.log_error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if leverage > self.max_leverage:
            logger.log_warning(f"Leverage {leverage} exceeds maximum {self.max_leverage}")
            leverage = self.max_leverage
        
        try:
            params = {
                'symbol': symbol,
                'leverage': leverage
            }
            response = await self._make_request('POST', '/fapi/v1/leverage', params, signed=True)
            
            if response:
                logger.log_info(f"Leverage set to {leverage} for {symbol}")
                return True
            return False
            
        except Exception as e:
            logger.log_error(f"Error setting leverage for {symbol}: {str(e)}")
            return False
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Position size in base asset
        """
        try:
            # Calculate risk amount
            risk_amount = self.balance * self.risk_per_trade
            
            # Calculate price difference
            if entry_price > stop_loss_price:  # Long position
                price_diff = entry_price - stop_loss_price
            else:  # Short position
                price_diff = stop_loss_price - entry_price
            
            # Calculate position size
            position_size = risk_amount / price_diff
            
            # Apply maximum position size limit
            max_size = self.balance * self.max_position_size / entry_price
            position_size = min(position_size, max_size)
            
            logger.log_info(f"Calculated position size: {position_size:.6f} for {symbol}")
            return position_size
            
        except Exception as e:
            logger.log_error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    async def place_order(
        self, 
        symbol: str, 
        side: str, 
        order_type: str, 
        quantity: float, 
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Place an order on Binance Futures.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            order_type: MARKET, LIMIT, STOP_MARKET, etc.
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order response or None if failed
        """
        try:
            # Check if we're in dry run mode
            if config.DRY_RUN:
                logger.log_info(f"DRY RUN: Would place {side} order for {quantity} {symbol}")
                return {
                    'orderId': 'dry_run_' + str(int(time.time())),
                    'status': 'DRY_RUN',
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price
                }
            
            # Validate order parameters
            if quantity <= 0:
                logger.log_error(f"Invalid quantity: {quantity}")
                return None
            
            # Set leverage first
            await self.set_leverage(symbol, self.max_leverage)
            
            # Round quantity to allowed precision
            quantity = round_quantity(symbol, quantity)

            # Prepare order parameters
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            if price and order_type != 'MARKET':
                params['price'] = price
            
            if order_type == 'LIMIT':
                params['timeInForce'] = 'GTC'
            
            # Place the order
            response = await self._make_request('POST', '/fapi/v1/order', params, signed=True)
            
            if response:
                order_id = response['orderId']
                self.orders[order_id] = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'status': response['status'],
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.log_info(f"Order placed: {order_id} for {quantity} {symbol} at {price or 'MARKET'}")
                
                # Place stop loss and take profit if specified
                if stop_loss:
                    await self._place_stop_loss(symbol, side, quantity, stop_loss)
                
                if take_profit:
                    await self._place_take_profit(symbol, side, quantity, take_profit)
                
                return response
            
            return None
            
        except Exception as e:
            logger.log_error(f"Error placing order: {str(e)}")
            return None
    
    async def _place_stop_loss(self, symbol: str, side: str, quantity: float, stop_loss_price: float):
        """Place stop loss order."""
        try:
            stop_side = 'SELL' if side == 'BUY' else 'BUY'
            stop_params = {
                'symbol': symbol,
                'side': stop_side,
                'type': 'STOP_MARKET',
                'quantity': quantity,
                'stopPrice': stop_loss_price,
                'reduceOnly': True
            }
            
            response = await self._make_request('POST', '/fapi/v1/order', stop_params, signed=True)
            if response:
                logger.log_info(f"Stop loss placed at {stop_loss_price} for {symbol}")
            
        except Exception as e:
            logger.log_error(f"Error placing stop loss: {str(e)}")
    
    async def _place_take_profit(self, symbol: str, side: str, quantity: float, take_profit_price: float):
        """Place take profit order."""
        try:
            tp_side = 'SELL' if side == 'BUY' else 'BUY'
            tp_params = {
                'symbol': symbol,
                'side': tp_side,
                'type': 'TAKE_PROFIT_MARKET',
                'quantity': quantity,
                'stopPrice': take_profit_price,
                'reduceOnly': True
            }
            
            response = await self._make_request('POST', '/fapi/v1/order', tp_params, signed=True)
            if response:
                logger.log_info(f"Take profit placed at {take_profit_price} for {symbol}")
            
        except Exception as e:
            logger.log_error(f"Error placing take profit: {str(e)}")
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            response = await self._make_request('DELETE', '/fapi/v1/order', params, signed=True)
            
            if response:
                logger.log_info(f"Order {order_id} cancelled for {symbol}")
                return True
            return False
            
        except Exception as e:
            logger.log_error(f"Error cancelling order: {str(e)}")
            return False
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            response = await self._make_request('GET', '/fapi/v1/openOrders', params, signed=True)
            
            if response:
                return response
            return []
            
        except Exception as e:
            logger.log_error(f"Error getting open orders: {str(e)}")
            return []
    
    async def close_position(self, symbol: str, side: str, quantity: float) -> bool:
        """Close an existing position."""
        try:
            close_side = 'SELL' if side == 'BUY' else 'BUY'
            
            response = await self.place_order(
                symbol=symbol,
                side=close_side,
                order_type='MARKET',
                quantity=quantity
            )
            
            if response:
                logger.log_info(f"Position closed for {symbol}: {quantity} {side}")
                return True
            return False
            
        except Exception as e:
            logger.log_error(f"Error closing position: {str(e)}")
            return False
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """Check if current positions meet risk management criteria."""
        risk_status = {
            'balance_ok': True,
            'leverage_ok': True,
            'position_size_ok': True,
            'overall_ok': True
        }
        
        try:
            # Check balance threshold
            if self.balance < (self.balance * self.min_balance_threshold):
                risk_status['balance_ok'] = False
                risk_status['overall_ok'] = False
            
            # Check leverage
            for symbol, position in self.positions.items():
                if position['leverage'] > self.max_leverage:
                    risk_status['leverage_ok'] = False
                    risk_status['overall_ok'] = False
                
                # Check position size
                position_value = position['size'] * position['entry_price']
                if position_value > (self.balance * self.max_position_size):
                    risk_status['position_size_ok'] = False
                    risk_status['overall_ok'] = False
            
            return risk_status
            
        except Exception as e:
            logger.log_error(f"Error checking risk limits: {str(e)}")
            return {'overall_ok': False}
    
    async def execute_signal(self, signal: Dict[str, Any]) -> Optional[Dict]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal from strategy
            
        Returns:
            Execution result or None if failed
        """
        try:
            if not signal or signal.get('signal') == 'no-trade':
                logger.log_info("No trading signal to execute")
                return None
            
            symbol = signal['symbol']
            signal_type = signal['signal']
            
            # Get current price
            current_price = await self.get_symbol_price(symbol)
            if not current_price:
                logger.log_error(f"Could not get price for {symbol}")
                return None
            
            # Check risk limits
            risk_status = self.check_risk_limits()
            if not risk_status['overall_ok']:
                logger.log_warning(f"Risk limits exceeded for {symbol}")
                return None
            
            # Calculate position size
            if signal_type == 'long_bias':
                side = 'BUY'
                stop_loss_price = current_price * (1 - self.stop_loss_percent)
                take_profit_price = current_price * (1 + self.take_profit_percent)
            elif signal_type == 'short_bias':
                side = 'SELL'
                stop_loss_price = current_price * (1 + self.stop_loss_percent)
                take_profit_price = current_price * (1 - self.take_profit_percent)
            else:
                logger.log_warning(f"Unknown signal type: {signal_type}")
                return None
            
            # Calculate quantity
            quantity = self.calculate_position_size(symbol, current_price, stop_loss_price)
            if quantity <= 0:
                logger.log_error(f"Invalid position size calculated: {quantity}")
                return None
            
            # Place the order
            order_result = await self.place_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price
            )
            
            if order_result:
                # Log the trade
                trade_data = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': current_price,
                    'order_id': order_result.get('orderId', 'unknown'),
                    'status': order_result.get('status', 'unknown'),
                    'pnl': 0.0,
                    'balance': self.balance
                }
                
                logger.log_trade(trade_data)
                
                return {
                    'success': True,
                    'order': order_result,
                    'trade_data': trade_data
                }
            
            return None
            
        except Exception as e:
            logger.log_error(f"Error executing signal: {str(e)}")
            return None

# Global trader instance
trader = BinanceTrader()
