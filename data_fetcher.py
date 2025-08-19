"""
Data fetcher module for the crypto trading bot.
Fetches OHLCV data from Binance Futures and CoinGecko.
Provides cleaned candlestick and indicator-ready data.
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time

from config import config
from logger import logger

class DataFetcher:
    """Fetches market data from multiple sources."""
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.session: Optional[aiohttp.ClientSession] = None
        self.binance_base_url = config.BINANCE_BASE_URL
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        
        # Add connection pooling for better performance
        self.connector = aiohttp.TCPConnector(
            limit=100,  # Max connections
            limit_per_host=30,  # Max per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            enable_cleanup_closed=True
        )
        
        # Cache for symbol mappings
        self._symbol_mapping_cache = {}
        self._mapping_cache_ttl = 3600  # 1 hour
        self._last_mapping_update = 0
        
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(connector=self.connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _get_symbol_mapping(self) -> Dict[str, str]:
        """
        Get dynamic symbol mapping from CoinGecko API.
        Falls back to config-based mapping if API fails.
        """
        current_time = time.time()
        
        # Check cache first
        if (current_time - self._last_mapping_update < self._mapping_cache_ttl 
            and self._symbol_mapping_cache):
            return self._symbol_mapping_cache
        
        try:
            # Try to get supported coins from CoinGecko
            url = f"{self.coingecko_base_url}/coins/list"
            async with self.session.get(url) as response:
                if response.status == 200:
                    coins_data = await response.json()
                    
                    # Create mapping for configured symbols
                    mapping = {}
                    for symbol in config.SYMBOLS:
                        # Extract base asset (remove USDT suffix)
                        base_asset = symbol.replace('USDT', '').lower()
                        
                        # Find matching coin in CoinGecko data
                        for coin in coins_data:
                            if coin['symbol'].lower() == base_asset:
                                mapping[symbol] = coin['id']
                                break
                        
                        # Fallback: use common mappings for popular coins
                        if symbol not in mapping:
                            fallback_mappings = {
                                'ETHUSDT': 'ethereum',
                                'BTCUSDT': 'bitcoin',
                                'SOLUSDT': 'solana',
                                'ADAUSDT': 'cardano',
                                'DOTUSDT': 'polkadot',
                                'LINKUSDT': 'chainlink',
                                'MATICUSDT': 'matic-network',
                                'AVAXUSDT': 'avalanche-2',
                                'UNIUSDT': 'uniswap',
                                'ATOMUSDT': 'cosmos',
                                'XRPUSDT': 'ripple',
                                'TRXUSDT': 'tron'
                            }
                            mapping[symbol] = fallback_mappings.get(symbol, '')
                    
                    self._symbol_mapping_cache = mapping
                    self._last_mapping_update = current_time
                    logger.log_info(f"Updated symbol mapping: {mapping}")
                    return mapping
                    
        except Exception as e:
            logger.log_warning(f"Failed to fetch symbol mapping from CoinGecko: {str(e)}")
        
        # Fallback to config-based mapping
        fallback_mapping = {}
        for symbol in config.SYMBOLS:
            base_asset = symbol.replace('USDT', '').lower()
            # Use common names for popular coins
            if base_asset == 'eth':
                fallback_mapping[symbol] = 'ethereum'
            elif base_asset == 'btc':
                fallback_mapping[symbol] = 'bitcoin'
            elif base_asset == 'sol':
                fallback_mapping[symbol] = 'solana'
            else:
                fallback_mapping[symbol] = base_asset
        
        self._symbol_mapping_cache = fallback_mapping
        self._last_mapping_update = current_time
        logger.log_info(f"Using fallback symbol mapping: {fallback_mapping}")
        return fallback_mapping

    async def fetch_binance_klines(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Fetch klines (OHLCV) data from Binance Futures.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            interval: Time interval (e.g., '1m', '5m')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.session:
            logger.log_error("Session not initialized")
            return None
            
        url = f"{self.binance_base_url}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Clean and convert data types
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Set timestamp as index
                    df.set_index('timestamp', inplace=True)
                    
                    # Remove rows with NaN values
                    df.dropna(inplace=True)
                    
                    logger.log_info(f"Fetched {len(df)} candles for {symbol} {interval}")
                    return df
                    
                else:
                    logger.log_error(f"Failed to fetch data: {response.status}")
                    return None
                    
        except Exception as e:
            logger.log_error(f"Error fetching Binance data: {str(e)}")
            return None
    
    async def fetch_coingecko_data(
        self, 
        coin_id: str, 
        days: int = 1
    ) -> Optional[Dict]:
        """
        Fetch additional market data from CoinGecko.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'ethereum', 'solana')
            days: Number of days of data to fetch
            
        Returns:
            Dictionary with market data or None if failed
        """
        if not self.session:
            logger.log_error("Session not initialized")
            return None
            
        url = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract price and volume data
                    prices = data.get('prices', [])
                    volumes = data.get('total_volumes', [])
                    
                    if prices and volumes:
                        # Convert to more usable format
                        market_data = {
                            'prices': [(pd.to_datetime(price[0], unit='ms'), price[1]) for price in prices],
                            'volumes': [(pd.to_datetime(vol[0], unit='ms'), vol[1]) for vol in volumes],
                            'current_price': prices[-1][1] if prices else None,
                            'current_volume': volumes[-1][1] if volumes else None
                        }
                        
                        logger.log_info(f"Fetched CoinGecko data for {coin_id}")
                        return market_data
                    
                else:
                    logger.log_error(f"Failed to fetch CoinGecko data: {response.status}")
                    return None
                    
        except Exception as e:
            logger.log_error(f"Error fetching CoinGecko data: {str(e)}")
            return None
    
    async def fetch_multiple_timeframes(
        self, 
        symbol: str, 
        timeframes: List[str], 
        limit: int = 100
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for multiple timeframes simultaneously.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to fetch
            limit: Number of candles per timeframe
            
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        # Create tasks for parallel execution
        tasks = []
        for tf in timeframes:
            task = self.fetch_binance_klines(symbol, tf, limit)
            tasks.append((tf, task))
        
        # Execute all tasks concurrently
        results = {}
        for tf, task in tasks:
            try:
                df = await task
                results[tf] = df
            except Exception as e:
                logger.log_error(f"Error fetching {tf} data for {symbol}: {str(e)}")
                results[tf] = None
        
        return results
    
    async def fetch_multiple_symbols_batch(
        self, 
        symbols: List[str], 
        timeframes: List[str] = None,
        limit: int = 100
    ) -> Dict[str, Dict]:
        """
        Fetch data for multiple symbols in parallel batches.
        
        Args:
            symbols: List of symbols to fetch
            timeframes: List of timeframes to fetch (defaults to config timeframes)
            limit: Number of candles per timeframe
            
        Returns:
            Dictionary mapping symbols to their data
        """
        if timeframes is None:
            timeframes = [config.TIMEFRAME_EXECUTION, config.TIMEFRAME_CONFIRMATION]
        
        # Create tasks for all symbols
        tasks = []
        for symbol in symbols:
            task = self.fetch_symbol_data(symbol)
            tasks.append((symbol, task))
        
        # Execute all tasks concurrently
        results = {}
        for symbol, task in tasks:
            try:
                data = await task
                results[symbol] = data
                logger.log_info(f"Batch fetched data for {symbol}")
            except Exception as e:
                logger.log_error(f"Error in batch fetch for {symbol}: {str(e)}")
                results[symbol] = None
        
        return results
    
    async def fetch_symbol_data(
        self, 
        symbol: str
    ) -> Dict[str, any]:
        """
        Fetch comprehensive data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with all available data
        """
        # Get dynamic symbol mapping
        symbol_mapping = await self._get_symbol_mapping()
        coin_id = symbol_mapping.get(symbol, '')
        
        # Fetch data for both timeframes
        timeframes = [config.TIMEFRAME_EXECUTION, config.TIMEFRAME_CONFIRMATION]
        klines_data = await self.fetch_multiple_timeframes(symbol, timeframes)
        
        # Fetch additional market data from CoinGecko
        coingecko_data = None
        if coin_id:
            coingecko_data = await self.fetch_coingecko_data(coin_id)
        
        return {
            'symbol': symbol,
            'klines': klines_data,
            'market_data': coingecko_data,
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Validate the fetched data for quality.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol for logging
            timeframe: Timeframe for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        if df is None or df.empty:
            logger.log_warning(f"No data received for {symbol} {timeframe}")
            return False
        
        # Check for minimum required data
        if len(df) < 20:  # Need at least 20 candles for indicators
            logger.log_warning(f"Insufficient data for {symbol} {timeframe}: {len(df)} candles")
            return False
        
        # Check for data quality issues
        if df['close'].isna().any() or df['volume'].isna().any():
            logger.log_warning(f"Data quality issues detected for {symbol} {timeframe}")
            return False
        
        # Check for reasonable price values
        if (df['close'] <= 0).any():
            logger.log_warning(f"Invalid price values detected for {symbol} {timeframe}")
            return False
        
        logger.log_info(f"Data validation passed for {symbol} {timeframe}")
        return True

# Utility function for standalone usage
async def fetch_data_for_symbols(symbols: List[str]) -> Dict[str, Dict]:
    """
    Fetch data for multiple symbols.
    
    Args:
        symbols: List of symbols to fetch data for
        
    Returns:
        Dictionary mapping symbols to their data
    """
    async with DataFetcher() as fetcher:
        results = {}
        for symbol in symbols:
            try:
                data = await fetcher.fetch_symbol_data(symbol)
                results[symbol] = data
            except Exception as e:
                logger.log_error(f"Failed to fetch data for {symbol}: {str(e)}")
                results[symbol] = None
        
        return results
