"""
Technical indicators module for the crypto trading bot.
Implements EMA, ATR, RSI, Bollinger Bands, VWAP, and other indicators.
Outputs structured dict for each symbol.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
import time
import hashlib

from logger import logger

class TechnicalIndicators:
    """Technical analysis indicators calculator."""
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        self._cache = {}
        self._cache_ttl = 60  # Cache for 60 seconds
        self._max_cache_size = 1000  # Maximum cache entries
    
    def _get_cache_key(self, df: pd.DataFrame, indicator: str, params: tuple) -> str:
        """Generate cache key for indicator calculation."""
        if df.empty:
            return f"{indicator}_{params}_empty"
        
        # Use last few timestamps and data points for cache key
        last_timestamps = df.index[-5:].strftime('%Y%m%d%H%M%S').tolist()
        last_prices = df['close'].tail(5).round(6).tolist()
        
        # Create hash of the key data
        key_data = f"{indicator}_{params}_{last_timestamps}_{last_prices}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Clean up old cache entries to prevent memory bloat."""
        current_time = time.time()
        expired_keys = []
        
        for key, (timestamp, _) in self._cache.items():
            if current_time - timestamp > self._cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        # If cache is still too large, remove oldest entries
        if len(self._cache) > self._max_cache_size:
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1][0])
            entries_to_remove = len(sorted_entries) - self._max_cache_size
            for i in range(entries_to_remove):
                del self._cache[sorted_entries[i][0]]
    
    def calculate_ema(self, df: pd.DataFrame, periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate Exponential Moving Averages with caching.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of EMA periods to calculate (default: adaptive based on data size)
            
        Returns:
            Dictionary mapping period to EMA series
        """
        if periods is None:
            # Use periods that work with available data
            max_period = min(50, len(df) - 1)
            if max_period >= 21:
                periods = [9, 21, max_period]
            elif max_period >= 9:
                periods = [9, max_period]
            else:
                periods = [max_period] if max_period > 0 else []
        
        # Check cache first
        cache_key = self._get_cache_key(df, 'ema', tuple(periods))
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        emas = {}
        for period in periods:
            if len(df) >= period:
                emas[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            else:
                logger.log_warning(f"Insufficient data for EMA {period}: {len(df)} candles")
                emas[f'ema_{period}'] = pd.Series([np.nan] * len(df), index=df.index)
        
        # Cache the result
        self._cache[cache_key] = (time.time(), emas)
        self._cleanup_cache()
        
        return emas
    
    def calculate_sma(self, df: pd.DataFrame, periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate Simple Moving Averages with caching.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of SMA periods to calculate (default: adaptive based on data size)
            
        Returns:
            Dictionary mapping period to SMA series
        """
        if periods is None:
            # Use periods that work with available data
            max_period = min(200, len(df) - 1)
            if max_period >= 50:
                periods = [20, 50, max_period]
            elif max_period >= 20:
                periods = [20, max_period]
            else:
                periods = [max_period] if max_period > 0 else []
        
        # Check cache first
        cache_key = self._get_cache_key(df, 'sma', tuple(periods))
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        smas = {}
        for period in periods:
            if len(df) >= period:
                smas[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            else:
                logger.log_warning(f"Insufficient data for SMA {period}: {len(df)} candles")
                smas[f'sma_{period}'] = pd.Series([np.nan] * len(df), index=df.index)
        
        # Cache the result
        self._cache[cache_key] = (time.time(), smas)
        self._cleanup_cache()
        
        return smas
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index with caching.
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            
        Returns:
            RSI series
        """
        # Check cache first
        cache_key = self._get_cache_key(df, 'rsi', (period,))
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        if len(df) >= period:
            rsi = ta.momentum.rsi(df['close'], window=period)
        else:
            logger.log_warning(f"Insufficient data for RSI {period}: {len(df)} candles")
            rsi = pd.Series([np.nan] * len(df), index=df.index)
        
        # Cache the result
        self._cache[cache_key] = (time.time(), rsi)
        self._cleanup_cache()
        
        return rsi
    
    def calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands with caching.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for moving average
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        # Check cache first
        cache_key = self._get_cache_key(df, 'bb', (period, std_dev))
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        if len(df) >= period:
            bb = ta.volatility.BollingerBands(
                close=df['close'], 
                window=period, 
                window_dev=std_dev
            )
            
            bands = {
                'bb_upper': bb.bollinger_hband(),
                'bb_middle': bb.bollinger_mavg(),
                'bb_lower': bb.bollinger_lband(),
                'bb_width': bb.bollinger_wband(),
                'bb_percent': bb.bollinger_pband()
            }
        else:
            logger.log_warning(f"Insufficient data for Bollinger Bands {period}: {len(df)} candles")
            bands = {
                'bb_upper': pd.Series([np.nan] * len(df), index=df.index),
                'bb_middle': pd.Series([np.nan] * len(df), index=df.index),
                'bb_lower': pd.Series([np.nan] * len(df), index=df.index),
                'bb_width': pd.Series([np.nan] * len(df), index=df.index),
                'bb_percent': pd.Series([np.nan] * len(df), index=df.index)
            }
        
        # Cache the result
        self._cache[cache_key] = (time.time(), bands)
        self._cleanup_cache()
        
        return bands
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range with caching.
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            ATR series
        """
        # Check cache first
        cache_key = self._get_cache_key(df, 'atr', (period,))
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        if len(df) >= period:
            atr = ta.volatility.average_true_range(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                window=period
            )
        else:
            logger.log_warning(f"Insufficient data for ATR {period}: {len(df)} candles")
            atr = pd.Series([np.nan] * len(df), index=df.index)
        
        # Cache the result
        self._cache[cache_key] = (time.time(), atr)
        self._cleanup_cache()
        
        return atr
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price with caching.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            VWAP series
        """
        # Check cache first
        cache_key = self._get_cache_key(df, 'vwap', ())
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        if len(df) >= 1:
            # Calculate VWAP manually since ta.volume.volume_weighted_average_price has different parameters
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Handle any NaN values
            vwap = vwap.bfill()
        else:
            logger.log_warning("Insufficient data for VWAP")
            vwap = pd.Series([np.nan] * len(df), index=df.index)
        
        # Cache the result
        self._cache[cache_key] = (time.time(), vwap)
        self._cleanup_cache()
        
        return vwap
    
    def calculate_stochastic(
        self, 
        df: pd.DataFrame, 
        k_period: int = 14, 
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator with caching.
        
        Args:
            df: DataFrame with OHLCV data
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with %K and %D lines
        """
        # Check cache first
        cache_key = self._get_cache_key(df, 'stoch', (k_period, d_period))
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        if len(df) >= k_period:
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=k_period,
                smooth_window=d_period
            )
            
            stoch_data = {
                'stoch_k': stoch.stoch(),
                'stoch_d': stoch.stoch_signal()
            }
        else:
            logger.log_warning(f"Insufficient data for Stochastic: {len(df)} candles")
            stoch_data = {
                'stoch_k': pd.Series([np.nan] * len(df), index=df.index),
                'stoch_d': pd.Series([np.nan] * len(df), index=df.index)
            }
        
        # Cache the result
        self._cache[cache_key] = (time.time(), stoch_data)
        self._cleanup_cache()
        
        return stoch_data
    
    def calculate_macd(
        self, 
        df: pd.DataFrame, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD with caching.
        
        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        # Check cache first
        cache_key = self._get_cache_key(df, 'macd', (fast, slow, signal))
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        if len(df) >= slow:
            macd = ta.trend.MACD(
                close=df['close'], 
                window_fast=fast, 
                window_slow=slow, 
                window_sign=signal
            )
            
            macd_data = {
                'macd_line': macd.macd(),
                'macd_signal': macd.macd_signal(),
                'macd_histogram': macd.macd_diff()
            }
        else:
            logger.log_warning(f"Insufficient data for MACD: {len(df)} candles")
            macd_data = {
                'macd_line': pd.Series([np.nan] * len(df), index=df.index),
                'macd_signal': pd.Series([np.nan] * len(df), index=df.index),
                'macd_histogram': pd.Series([np.nan] * len(df), index=df.index)
            }
        
        # Cache the result
        self._cache[cache_key] = (time.time(), macd_data)
        self._cleanup_cache()
        
        return macd_data
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators with caching.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume indicators
        """
        # Check cache first
        cache_key = self._get_cache_key(df, 'volume', ())
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        if len(df) >= 20:
            # Volume SMA - calculate manually since ta.volume.volume_sma doesn't exist
            volume_sma = df['volume'].rolling(window=20).mean()
            
            # On Balance Volume
            obv = ta.volume.on_balance_volume(df['close'], df['volume'])
            
            # Volume Rate of Change - calculate manually
            vroc = df['volume'].pct_change(periods=20) * 100
            
            # Money Flow Index
            mfi = ta.volume.money_flow_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=14
            )
            
            volume_data = {
                'volume_sma': volume_sma,
                'obv': obv,
                'vroc': vroc,
                'mfi': mfi
            }
        else:
            logger.log_warning(f"Insufficient data for volume indicators: {len(df)} candles")
            volume_data = {
                'volume_sma': pd.Series([np.nan] * len(df), index=df.index),
                'obv': pd.Series([np.nan] * len(df), index=df.index),
                'vroc': pd.Series([np.nan] * len(df), index=df.index),
                'mfi': pd.Series([np.nan] * len(df), index=df.index)
            }
        
        # Cache the result
        self._cache[cache_key] = (time.time(), volume_data)
        self._cleanup_cache()
        
        return volume_data
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate all technical indicators with comprehensive caching.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all calculated indicators
        """
        # Check cache first for the complete indicator set
        cache_key = self._get_cache_key(df, 'all_indicators', ())
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        if df.empty or len(df) < 20:
            logger.log_warning(f"Insufficient data for indicators: {len(df)} candles")
            return {}
        
        try:
            # Calculate all indicators
            indicators = {}
            
            # Trend indicators
            indicators.update(self.calculate_ema(df))
            indicators.update(self.calculate_sma(df))
            
            # Momentum indicators
            indicators['rsi'] = self.calculate_rsi(df)
            indicators.update(self.calculate_macd(df))
            indicators.update(self.calculate_stochastic(df))
            
            # Volatility indicators
            indicators.update(self.calculate_bollinger_bands(df))
            indicators['atr'] = self.calculate_atr(df)
            
            # Volume indicators
            indicators.update(self.calculate_volume_indicators(df))
            
            # Price-based indicators
            indicators['vwap'] = self.calculate_vwap(df)
            
            # Remove any None values
            indicators = {k: v for k, v in indicators.items() if v is not None}
            
            # Cache the complete result
            self._cache[cache_key] = (time.time(), indicators)
            self._cleanup_cache()
            
            logger.log_info(f"Calculated {len(indicators)} indicators")
            return indicators
            
        except Exception as e:
            logger.log_error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def get_latest_values(self, indicators: Dict[str, any]) -> Dict[str, float]:
        """
        Get the latest values for all indicators.
        
        Args:
            indicators: Dictionary of indicator series
            
        Returns:
            Dictionary with latest values
        """
        latest_values = {}
        
        for name, series in indicators.items():
            if isinstance(series, pd.Series) and not series.empty:
                latest_values[name] = float(series.iloc[-1])
            else:
                latest_values[name] = np.nan
        
        return latest_values
    
    def get_indicator_summary(self, indicators: Dict[str, any]) -> Dict[str, any]:
        """
        Get a summary of all indicators for analysis.
        
        Args:
            indicators: Dictionary of indicator series
            
        Returns:
            Dictionary with indicator summary
        """
        summary = {
            'latest_values': self.get_latest_values(indicators),
            'trend_signals': {},
            'momentum_signals': {},
            'volatility_signals': {},
            'volume_signals': {}
        }
        
        # Trend signals
        if 'ema_9' in indicators and 'ema_21' in indicators:
            ema_9 = indicators['ema_9'].iloc[-1]
            ema_21 = indicators['ema_21'].iloc[-1]
            summary['trend_signals']['ema_trend'] = 'bullish' if ema_9 > ema_21 else 'bearish'
        
        # Momentum signals
        if 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]
            if rsi > 70:
                summary['momentum_signals']['rsi'] = 'overbought'
            elif rsi < 30:
                summary['momentum_signals']['rsi'] = 'oversold'
            else:
                summary['momentum_signals']['rsi'] = 'neutral'
        
        # Volatility signals
        if 'bb_percent' in indicators:
            bb_percent = indicators['bb_percent'].iloc[-1]
            if bb_percent > 0.8:
                summary['volatility_signals']['bb_position'] = 'upper_band'
            elif bb_percent < 0.2:
                summary['volatility_signals']['bb_position'] = 'lower_band'
            else:
                summary['volatility_signals']['bb_position'] = 'middle_range'
        
        return summary

# Global indicators instance
indicators_calculator = TechnicalIndicators()
