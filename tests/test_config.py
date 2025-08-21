"""
Test configuration module.
"""

import pytest
import os
from unittest.mock import patch
from config import Config, config

class TestConfig:
    """Test configuration class."""
    
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        assert hasattr(config, 'BINANCE_API_KEY')
        assert hasattr(config, 'OPENAI_API_KEY')
        assert hasattr(config, 'TELEGRAM_BOT_TOKEN')
    
    def test_symbols(self):
        """Test trading symbols configuration."""
        assert 'LINKUSDT' in config.SYMBOLS
        assert len(config.SYMBOLS) == 1
    
    def test_timeframes(self):
        """Test timeframe configuration."""
        assert config.TIMEFRAME_EXECUTION == '1m'
        assert config.TIMEFRAME_CONFIRMATION == '5m'
    
    def test_risk_settings(self):
        """Test risk management settings."""
        assert config.MAX_LEVERAGE > 0
        assert 0 < config.RISK_PER_TRADE < 1
        assert 0 < config.STOP_LOSS_PERCENT < 1
        assert 0 < config.TAKE_PROFIT_PERCENT < 1
    
    @patch.dict(os.environ, {
        'BINANCE_API_KEY': 'test_key',
        'BINANCE_SECRET_KEY': 'test_secret',
        'OPENAI_API_KEY': 'test_openai',
        'TELEGRAM_BOT_TOKEN': 'test_telegram',
        'TELEGRAM_CHAT_ID': 'test_chat'
    })
    def test_config_validation(self):
        """Test configuration validation."""
        # Create a new config instance with the mocked environment
        with patch.object(Config, 'BINANCE_API_KEY', 'test_key'), \
             patch.object(Config, 'BINANCE_SECRET_KEY', 'test_secret'), \
             patch.object(Config, 'OPENAI_API_KEY', 'test_openai'), \
             patch.object(Config, 'TELEGRAM_BOT_TOKEN', 'test_telegram'), \
             patch.object(Config, 'TELEGRAM_CHAT_ID', 'test_chat'):
            
            test_config = Config()
            assert test_config.is_valid()
    
    def test_binance_config(self):
        """Test Binance configuration getter."""
        binance_config = config.get_binance_config()
        assert 'api_key' in binance_config
        assert 'api_secret' in binance_config
        assert 'testnet' in binance_config
