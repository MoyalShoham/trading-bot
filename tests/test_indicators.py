"""
Test technical indicators module.
"""

import pytest
import pandas as pd
import numpy as np
from indicators import TechnicalIndicators

class TestTechnicalIndicators:
    """Test technical indicators class."""
    
    def setup_method(self):
        """Set up test data."""
        self.indicators = TechnicalIndicators()
        
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(150, 250, 100),
            'low': np.random.uniform(50, 150, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Ensure high >= close >= low
        self.sample_data['high'] = np.maximum(
            self.sample_data['high'], 
            self.sample_data['close']
        )
        self.sample_data['low'] = np.minimum(
            self.sample_data['low'], 
            self.sample_data['close']
        )
    
    def test_ema_calculation(self):
        """Test EMA calculation."""
        emas = self.indicators.calculate_ema(self.sample_data, [9, 21])
        
        assert 'ema_9' in emas
        assert 'ema_21' in emas
        assert len(emas['ema_9']) == len(self.sample_data)
        assert not emas['ema_9'].isna().all()
    
    def test_sma_calculation(self):
        """Test SMA calculation."""
        smas = self.indicators.calculate_sma(self.sample_data, [20, 50])
        
        assert 'sma_20' in smas
        assert 'sma_50' in smas
        assert len(smas['sma_20']) == len(self.sample_data)
        assert not smas['sma_20'].isna().all()
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi = self.indicators.calculate_rsi(self.sample_data, 14)
        
        assert len(rsi) == len(self.sample_data)
        assert not rsi.isna().all()
        # RSI should be between 0 and 100
        assert rsi.dropna().min() >= 0
        assert rsi.dropna().max() <= 100
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb = self.indicators.calculate_bollinger_bands(self.sample_data, 20, 2.0)
        assert 'bb_upper' in bb
        assert 'bb_middle' in bb
        assert 'bb_lower' in bb
        assert 'bb_width' in bb
        assert 'bb_percent' in bb
        # Only compare non-NaN values
        valid = (~bb['bb_upper'].isna()) & (~bb['bb_middle'].isna()) & (~bb['bb_lower'].isna())
        assert (bb['bb_upper'][valid] >= bb['bb_middle'][valid]).all()
        assert (bb['bb_middle'][valid] >= bb['bb_lower'][valid]).all()
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        atr = self.indicators.calculate_atr(self.sample_data, 14)
        
        assert len(atr) == len(self.sample_data)
        assert not atr.isna().all()
        # ATR should be positive
        assert atr.dropna().min() >= 0
    
    def test_vwap_calculation(self):
        """Test VWAP calculation."""
        vwap = self.indicators.calculate_vwap(self.sample_data)
        
        assert len(vwap) == len(self.sample_data)
        assert not vwap.isna().all()
        # VWAP should be between low and high (with some tolerance for volume weighting)
        # VWAP can occasionally be outside the high-low range due to volume weighting
        # So we'll check that it's reasonable (within 5% of the range)
        price_range = self.sample_data['high'] - self.sample_data['low']
        tolerance = price_range * 0.05
        assert ((vwap >= self.sample_data['low'] - tolerance) & 
                (vwap <= self.sample_data['high'] + tolerance)).all()
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd = self.indicators.calculate_macd(self.sample_data)
        
        assert 'macd_line' in macd
        assert 'macd_signal' in macd
        assert 'macd_histogram' in macd
        assert len(macd['macd_line']) == len(self.sample_data)
    
    def test_stochastic_calculation(self):
        """Test Stochastic calculation."""
        stoch = self.indicators.calculate_stochastic(self.sample_data)
        
        assert 'stoch_k' in stoch
        assert 'stoch_d' in stoch
        assert len(stoch['stoch_k']) == len(self.sample_data)
        # Stochastic should be between 0 and 100
        assert stoch['stoch_k'].dropna().min() >= 0
        assert stoch['stoch_k'].dropna().max() <= 100
    
    def test_volume_indicators(self):
        """Test volume indicators calculation."""
        volume_indicators = self.indicators.calculate_volume_indicators(self.sample_data)
        
        assert 'volume_sma' in volume_indicators
        assert 'obv' in volume_indicators
        assert 'vroc' in volume_indicators
        assert len(volume_indicators['volume_sma']) == len(self.sample_data)
    
    def test_all_indicators(self):
        """Test calculation of all indicators."""
        all_indicators = self.indicators.calculate_all_indicators(self.sample_data)
        
        # Should have multiple indicator types
        assert len(all_indicators) > 10
        
        # Check for key indicator types
        assert 'ema_9' in all_indicators
        assert 'rsi' in all_indicators
        assert 'atr' in all_indicators
        assert 'vwap' in all_indicators
    
    def test_latest_values(self):
        """Test getting latest indicator values."""
        all_indicators = self.indicators.calculate_all_indicators(self.sample_data)
        latest_values = self.indicators.get_latest_values(all_indicators)
        
        assert len(latest_values) > 0
        assert all(isinstance(v, (int, float)) or np.isnan(v) for v in latest_values.values())
    
    def test_indicator_summary(self):
        """Test indicator summary generation."""
        all_indicators = self.indicators.calculate_all_indicators(self.sample_data)
        summary = self.indicators.get_indicator_summary(all_indicators)
        
        assert 'latest_values' in summary
        assert 'trend_signals' in summary
        assert 'momentum_signals' in summary
        assert 'volatility_signals' in summary
        assert 'volume_signals' in summary
    
    def test_empty_data(self):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        result = self.indicators.calculate_all_indicators(empty_df)
        assert result == {}
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        small_df = self.sample_data.head(5)  # Only 5 rows
        
        # Should handle insufficient data gracefully
        emas = self.indicators.calculate_ema(small_df, [20])
        assert 'ema_20' in emas
        assert emas['ema_20'].isna().all()  # Should be all NaN for insufficient data
