
"""
Data fetcher module for the crypto trading bot.
Fetches OHLCV data from Binance Futures (via BinanceProvider).
Provides cleaned candlestick and indicator-ready data.
"""


import asyncio
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from config import config
from logger import logger
from data.binance_provider import BinanceProvider


class DataFetcher:
    """Fetches market data using the selected provider."""
    def __init__(self):
        self.provider_name = getattr(config, "PROVIDER", "BinanceProvider").upper()
        self.provider = None
        if self.provider_name == "BINANCEPROVIDER":
            self.provider = BinanceProvider(
                cache_ttl_prices=int(getattr(config, "CACHE_TTL_PRICES", 5)),
                cache_ttl_klines=int(getattr(config, "CACHE_TTL_KLINES", 60))
            )
        else:
            raise NotImplementedError("Only BinanceProvider is currently supported.")


    async def fetch_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch klines (OHLCV) data from the provider and return as DataFrame."""
        try:
            klines = await self.provider.get_klines(symbol, interval, limit)
            if not klines:
                logger.log_warning(f"No klines data for {symbol} {interval}")
                return None
            df = pd.DataFrame(klines)
            if 'open_time' in df:
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            logger.log_info(f"Fetched {len(df)} candles for {symbol} {interval}")
            return df
        except Exception as e:
            logger.log_error(f"Error fetching klines: {str(e)}")
            return None
    

    async def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch the latest price for a symbol from the provider."""
        try:
            price = await self.provider.get_price(symbol)
            return price
        except Exception as e:
            logger.log_error(f"Error fetching price for {symbol}: {str(e)}")
            return None

    async def fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch latest prices for multiple symbols from the provider."""
        try:
            prices = await self.provider.get_prices(symbols)
            return prices
        except Exception as e:
            logger.log_error(f"Error fetching prices: {str(e)}")
            return {}
    

    async def fetch_multiple_timeframes(self, symbol: str, timeframes: List[str], limit: int = 100) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch data for multiple timeframes using the provider."""
        tasks = [self.fetch_klines(symbol, tf, limit) for tf in timeframes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {tf: (res if not isinstance(res, Exception) else None) for tf, res in zip(timeframes, results)}
    

    async def fetch_multiple_symbols_batch(self, symbols: List[str], timeframes: List[str] = None, limit: int = 100) -> Dict[str, Dict]:
        """Fetch data for multiple symbols in parallel batches using the provider."""
        if timeframes is None:
            timeframes = [config.TIMEFRAME_EXECUTION, config.TIMEFRAME_CONFIRMATION]
        tasks = [self.fetch_symbol_data(symbol, timeframes, limit) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {symbol: (res if not isinstance(res, Exception) else None) for symbol, res in zip(symbols, results)}
    

    async def fetch_symbol_data(self, symbol: str, timeframes: List[str] = None, limit: int = 100) -> Dict[str, any]:
        """Fetch comprehensive data for a symbol using the provider."""
        if timeframes is None:
            timeframes = [config.TIMEFRAME_EXECUTION, config.TIMEFRAME_CONFIRMATION]
        klines_data = await self.fetch_multiple_timeframes(symbol, timeframes, limit)
        price = await self.fetch_price(symbol)
        return {
            'symbol': symbol,
            'klines': klines_data,
            'price': price,
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
    """Fetch data for multiple symbols using the provider."""
    fetcher = DataFetcher()
    results = {}
    for symbol in symbols:
        try:
            data = await fetcher.fetch_symbol_data(symbol)
            results[symbol] = data
        except Exception as e:
            logger.log_error(f"Failed to fetch data for {symbol}: {str(e)}")
            results[symbol] = None
    return results
