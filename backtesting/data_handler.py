"""
Historical data handler for backtesting with support for multiple data sources.
"""

import asyncio
import pandas as pd
import numpy as np
import aiohttp
import csv
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

from config import config
from utils.logger import TradingLogger

logger = TradingLogger(__name__)


class HistoricalDataHandler:
    """
    Handles loading and managing historical data for backtesting.
    Supports CSV files, SQLite database, and Binance API.
    """
    
    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the data handler."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache: Dict[str, pd.DataFrame] = {}
        
        # Database for storing historical data
        self.db_path = self.data_dir / "historical_data.db"
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for historical data storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS klines (
                        symbol TEXT,
                        timeframe TEXT,
                        timestamp INTEGER,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        PRIMARY KEY (symbol, timeframe, timestamp)
                    )
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
                    ON klines (symbol, timeframe, timestamp)
                ''')
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        source: str = "auto"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date for data
            end_date: End date for data
            source: Data source ('csv', 'db', 'api', 'auto')
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        try:
            cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
            
            # Check cache first
            if cache_key in self.cache:
                logger.debug(f"Using cached data for {symbol} {timeframe}")
                return self.cache[cache_key]
            
            data = None
            
            if source == "auto":
                # Try sources in order: db -> csv -> api
                data = await self._load_from_database(symbol, timeframe, start_date, end_date)
                
                if data is None or data.empty:
                    data = await self._load_from_csv(symbol, timeframe, start_date, end_date)
                
                if data is None or data.empty:
                    data = await self._load_from_api(symbol, timeframe, start_date, end_date)
                    
            elif source == "csv":
                data = await self._load_from_csv(symbol, timeframe, start_date, end_date)
            elif source == "db":
                data = await self._load_from_database(symbol, timeframe, start_date, end_date)
            elif source == "api":
                data = await self._load_from_api(symbol, timeframe, start_date, end_date)
            
            if data is not None and not data.empty:
                # Validate and clean data
                data = self._validate_and_clean_data(data)
                
                # Cache the result
                self.cache[cache_key] = data
                
                logger.info(f"Loaded {len(data)} bars for {symbol} {timeframe}")
                return data
            else:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    async def _load_from_csv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load data from CSV file."""
        try:
            csv_file = self.data_dir / f"{symbol}_{timeframe}.csv"
            
            if not csv_file.exists():
                logger.debug(f"CSV file not found: {csv_file}")
                return None
            
            df = pd.read_csv(csv_file)
            
            # Convert timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                # Assume first column is timestamp
                df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
            
            df.set_index('timestamp', inplace=True)
            
            # Filter by date range
            mask = (df.index >= start_date) & (df.index <= end_date)
            df = df.loc[mask]
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"CSV file missing required columns for {symbol}")
                return None
            
            return df[required_columns]
            
        except Exception as e:
            logger.error(f"Error loading CSV data for {symbol}: {e}")
            return None

    async def _load_from_database(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load data from SQLite database."""
        try:
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, open, high, low, close, volume
                    FROM klines
                    WHERE symbol = ? AND timeframe = ?
                    AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                '''
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, timeframe, start_timestamp, end_timestamp)
                )
                
                if df.empty:
                    return None
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
                
        except Exception as e:
            logger.error(f"Error loading database data for {symbol}: {e}")
            return None

    async def _load_from_api(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load data from Binance API."""
        try:
            logger.info(f"Fetching data from API for {symbol} {timeframe}")
            
            base_url = "https://fapi.binance.com"
            endpoint = "/fapi/v1/klines"
            
            # Convert timeframe to Binance format
            binance_interval = self._convert_timeframe(timeframe)
            if not binance_interval:
                logger.error(f"Unsupported timeframe: {timeframe}")
                return None
            
            all_data = []
            current_start = start_date
            
            async with aiohttp.ClientSession() as session:
                while current_start < end_date:
                    # Binance API limit is 1000 bars per request
                    params = {
                        'symbol': symbol,
                        'interval': binance_interval,
                        'startTime': int(current_start.timestamp() * 1000),
                        'endTime': int(end_date.timestamp() * 1000),
                        'limit': 1000
                    }
                    
                    async with session.get(f"{base_url}{endpoint}", params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if not data:
                                break
                            
                            all_data.extend(data)
                            
                            # Update start time for next request
                            last_timestamp = data[-1][0]
                            current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
                            
                            # Rate limiting
                            await asyncio.sleep(0.1)
                        else:
                            logger.error(f"API request failed: {response.status}")
                            break
            
            if not all_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Keep only required columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df.set_index('timestamp', inplace=True)
            
            # Save to database for future use
            await self._save_to_database(symbol, timeframe, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading API data for {symbol}: {e}")
            return None

    def _convert_timeframe(self, timeframe: str) -> Optional[str]:
        """Convert timeframe to Binance format."""
        timeframe_map = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
        return timeframe_map.get(timeframe)

    async def _save_to_database(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Save data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for timestamp, row in df.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO klines
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        timeframe,
                        int(timestamp.timestamp() * 1000),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ))
                conn.commit()
                
            logger.debug(f"Saved {len(df)} bars to database for {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")

    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean historical data."""
        try:
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Check for missing values
            if df.isnull().any().any():
                logger.warning("Found missing values in data, forward filling")
                df = df.fillna(method='ffill')
            
            # Validate OHLCV relationships
            invalid_bars = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close']) |
                (df['volume'] < 0)
            )
            
            if invalid_bars.any():
                logger.warning(f"Found {invalid_bars.sum()} invalid bars, removing them")
                df = df[~invalid_bars]
            
            # Remove zero volume bars (if any)
            df = df[df['volume'] > 0]
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return df

    async def download_historical_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Download and store historical data for multiple symbols."""
        logger.info(f"Downloading historical data for {len(symbols)} symbols")
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    logger.info(f"Downloading {symbol} {timeframe}")
                    
                    data = await self._load_from_api(symbol, timeframe, start_date, end_date)
                    
                    if data is not None:
                        # Also save to CSV
                        csv_file = self.data_dir / f"{symbol}_{timeframe}.csv"
                        data.to_csv(csv_file)
                        logger.info(f"Saved {len(data)} bars to {csv_file}")
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error downloading {symbol} {timeframe}: {e}")

    def get_available_data(self) -> Dict[str, List[str]]:
        """Get list of available data in the system."""
        available = {
            'csv_files': [],
            'database_symbols': []
        }
        
        try:
            # Check CSV files
            for csv_file in self.data_dir.glob("*.csv"):
                available['csv_files'].append(csv_file.stem)
            
            # Check database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT DISTINCT symbol || '_' || timeframe
                    FROM klines
                    ORDER BY symbol, timeframe
                ''')
                available['database_symbols'] = [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting available data: {e}")
        
        return available

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.cache.clear()
        logger.info("Data cache cleared")
