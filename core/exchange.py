"""
Binance Futures exchange interface with improved error handling and rate limiting.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import json

from config import config
from utils.decorators import rate_limit, retry_on_failure
from utils.logger import TradingLogger

logger = TradingLogger(__name__)


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    status: str
    timestamp: datetime
    commission: Optional[float] = None
    commission_asset: Optional[str] = None


@dataclass
class Position:
    """Trading position data."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    percentage: float
    leverage: int
    margin_type: str
    isolated_margin: Optional[float] = None


@dataclass
class Balance:
    """Account balance data."""
    asset: str
    wallet_balance: float
    unrealized_pnl: float
    margin_balance: float
    maint_margin: float
    initial_margin: float
    position_initial_margin: float
    open_order_initial_margin: float
    cross_wallet_balance: float
    cross_unrealized_pnl: float
    available_balance: float
    max_withdraw_amount: float


class BinanceExchange:
    """
    Binance Futures exchange interface with comprehensive error handling,
    rate limiting, and connection management.
    """

    def __init__(self):
        """Initialize the Binance exchange interface."""
        self.api_key = config.BINANCE_API_KEY
        self.secret_key = config.BINANCE_SECRET_KEY
        self.base_url = config.BINANCE_BASE_URL
        self.testnet = config.BINANCE_TESTNET
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._rate_limit_delay = 0.1  # 100ms between requests
        self._last_request_time = 0.0
        
        # Connection management
        self._connector = None
        self._timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        # Cache for symbol info
        self._symbol_info_cache: Dict[str, Dict] = {}
        self._cache_expiry = 3600  # 1 hour

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize the exchange connection."""
        try:
            self._connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=self._timeout,
                headers={'User-Agent': 'TradingBot/1.0'}
            )
            
            # Test connection
            await self.ping()
            logger.info("Binance exchange connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange connection: {e}")
            raise

    async def close(self) -> None:
        """Close the exchange connection."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            if self._connector:
                await self._connector.close()
                
            logger.info("Binance exchange connection closed")
            
        except Exception as e:
            logger.error(f"Error closing exchange connection: {e}")

    def _generate_signature(self, params: str) -> str:
        """Generate HMAC SHA256 signature for Binance API."""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self, signed: bool = False) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        if signed:
            headers['X-MBX-APIKEY'] = self.api_key
        return headers

    @rate_limit(calls=1200, period=60)  # Binance rate limit: 1200 requests per minute
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
        timeout: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Make HTTP request to Binance API with rate limiting and error handling.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign the request
            timeout: Custom timeout for this request
            
        Returns:
            API response data or None if failed
        """
        if not self.session or self.session.closed:
            await self.initialize()

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

        headers = self._get_headers(signed)
        
        # Apply rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last)
        
        self._last_request_time = time.time()

        try:
            if method.upper() == 'GET':
                if query_string:
                    url += f"?{query_string}"
                async with self.session.get(url, headers=headers, timeout=timeout) as response:
                    return await self._handle_response(response)
                    
            elif method.upper() == 'POST':
                async with self.session.post(url, data=query_string, headers=headers, timeout=timeout) as response:
                    return await self._handle_response(response)
                    
            elif method.upper() == 'DELETE':
                if query_string:
                    url += f"?{query_string}"
                async with self.session.delete(url, headers=headers, timeout=timeout) as response:
                    return await self._handle_response(response)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None

        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {endpoint}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Client error for {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {e}")
            return None

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Optional[Dict]:
        """Handle API response with proper error handling."""
        try:
            if response.status == 200:
                data = await response.json()
                return data
            elif response.status == 429:
                # Rate limit exceeded
                retry_after = response.headers.get('Retry-After', '60')
                logger.warning(f"Rate limit exceeded, retrying after {retry_after} seconds")
                await asyncio.sleep(int(retry_after))
                return None
            else:
                error_text = await response.text()
                logger.error(f"API error {response.status}: {error_text}")
                
                # Parse error details if available
                try:
                    error_data = json.loads(error_text)
                    error_code = error_data.get('code', 'unknown')
                    error_msg = error_data.get('msg', error_text)
                    logger.error(f"Binance API error - Code: {error_code}, Message: {error_msg}")
                except:
                    pass
                    
                return None
                
        except Exception as e:
            logger.error(f"Error handling response: {e}")
            return None

    async def ping(self) -> bool:
        """Test connectivity to Binance API."""
        try:
            response = await self._make_request('GET', '/fapi/v1/ping')
            return response is not None
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False

    async def get_server_time(self) -> Optional[int]:
        """Get server time from Binance."""
        try:
            response = await self._make_request('GET', '/fapi/v1/time')
            return response.get('serverTime') if response else None
        except Exception as e:
            logger.error(f"Failed to get server time: {e}")
            return None

    @retry_on_failure(max_retries=3, delay=1.0)
    async def get_account_info(self) -> Optional[Dict]:
        """Get account information including balances and positions."""
        try:
            response = await self._make_request('GET', '/fapi/v2/account', signed=True)
            return response
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    async def get_balance(self) -> List[Balance]:
        """Get account balances."""
        try:
            account_info = await self.get_account_info()
            if not account_info:
                return []

            balances = []
            for asset_data in account_info.get('assets', []):
                balance = Balance(
                    asset=asset_data['asset'],
                    wallet_balance=float(asset_data['walletBalance']),
                    unrealized_pnl=float(asset_data['unrealizedProfit']),
                    margin_balance=float(asset_data['marginBalance']),
                    maint_margin=float(asset_data['maintMargin']),
                    initial_margin=float(asset_data['initialMargin']),
                    position_initial_margin=float(asset_data['positionInitialMargin']),
                    open_order_initial_margin=float(asset_data['openOrderInitialMargin']),
                    cross_wallet_balance=float(asset_data['crossWalletBalance']),
                    cross_unrealized_pnl=float(asset_data['crossUnRealizedProfit']),
                    available_balance=float(asset_data['availableBalance']),
                    max_withdraw_amount=float(asset_data['maxWithdrawAmount'])
                )
                balances.append(balance)
                
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return []

    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        try:
            account_info = await self.get_account_info()
            if not account_info:
                return []

            positions = []
            for pos_data in account_info.get('positions', []):
                position_amt = float(pos_data['positionAmt'])
                if position_amt != 0:  # Only include open positions
                    position = Position(
                        symbol=pos_data['symbol'],
                        side='long' if position_amt > 0 else 'short',
                        size=abs(position_amt),
                        entry_price=float(pos_data['entryPrice']),
                        mark_price=float(pos_data['markPrice']),
                        unrealized_pnl=float(pos_data['unRealizedProfit']),
                        percentage=float(pos_data['percentage']),
                        leverage=int(pos_data['leverage']),
                        margin_type=pos_data['marginType'],
                        isolated_margin=float(pos_data['isolatedMargin']) if pos_data['marginType'] == 'isolated' else None
                    )
                    positions.append(position)
                    
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_symbol_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            params = {'symbol': symbol}
            response = await self._make_request('GET', '/fapi/v1/ticker/price', params)
            return float(response['price']) if response else None
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information including filters and precision."""
        try:
            # Check cache first
            if symbol in self._symbol_info_cache:
                cached_data, timestamp = self._symbol_info_cache[symbol]
                if time.time() - timestamp < self._cache_expiry:
                    return cached_data

            response = await self._make_request('GET', '/fapi/v1/exchangeInfo')
            if not response:
                return None

            for symbol_info in response.get('symbols', []):
                if symbol_info['symbol'] == symbol:
                    # Cache the result
                    self._symbol_info_cache[symbol] = (symbol_info, time.time())
                    return symbol_info
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            params = {
                'symbol': symbol,
                'leverage': leverage
            }
            response = await self._make_request('POST', '/fapi/v1/leverage', params, signed=True)
            
            if response and response.get('leverage') == leverage:
                logger.info(f"Leverage set to {leverage}x for {symbol}")
                return True
            else:
                logger.warning(f"Failed to set leverage for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            return False

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = 'GTC',
        reduce_only: bool = False,
        close_position: bool = False
    ) -> Optional[OrderResult]:
        """
        Place an order on Binance Futures.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            order_type: MARKET, LIMIT, STOP_MARKET, etc.
            quantity: Order quantity
            price: Order price (for limit orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            reduce_only: Whether this is a reduce-only order
            close_position: Whether to close entire position
            
        Returns:
            OrderResult if successful, None otherwise
        """
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }

            if price and order_type in ['LIMIT', 'STOP', 'TAKE_PROFIT']:
                params['price'] = price

            if order_type in ['LIMIT', 'STOP', 'TAKE_PROFIT']:
                params['timeInForce'] = time_in_force

            if reduce_only:
                params['reduceOnly'] = 'true'

            if close_position:
                params['closePosition'] = 'true'

            response = await self._make_request('POST', '/fapi/v1/order', params, signed=True)
            
            if response:
                order_result = OrderResult(
                    order_id=str(response['orderId']),
                    symbol=response['symbol'],
                    side=response['side'],
                    quantity=float(response['origQty']),
                    price=float(response['price']) if response.get('price') else None,
                    status=response['status'],
                    timestamp=datetime.fromtimestamp(response['transactTime'] / 1000),
                    commission=float(response.get('commission', 0)),
                    commission_asset=response.get('commissionAsset')
                )
                
                logger.info(f"Order placed: {order_result.order_id} for {quantity} {symbol}")
                return order_result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            response = await self._make_request('DELETE', '/fapi/v1/order', params, signed=True)
            
            if response:
                logger.info(f"Order {order_id} cancelled for {symbol}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            response = await self._make_request('GET', '/fapi/v1/openOrders', params, signed=True)
            return response if response else []
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    async def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Get order status."""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            response = await self._make_request('GET', '/fapi/v1/order', params, signed=True)
            return response
            
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None
