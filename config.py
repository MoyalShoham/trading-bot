"""
Configuration module for the crypto trading bot.
Loads environment variables and defines constants.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # === TELEGRAM NOTIFICATIONS ===
    TELEGRAM_NOTIFY_SIGNALS: bool = os.getenv('TELEGRAM_NOTIFY_SIGNALS', 'false').lower() == 'true'
    # === DATA PROVIDER ===
    PROVIDER: str = os.getenv('PROVIDER', 'BinanceProvider')
    CACHE_TTL_PRICES: int = int(os.getenv('CACHE_TTL_PRICES', '5'))
    CACHE_TTL_KLINES: int = int(os.getenv('CACHE_TTL_KLINES', '60'))
    """Configuration class containing all bot settings."""
    
    # === BINANCE FUTURES API ===
    BINANCE_API_KEY: str = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY: str = os.getenv('BINANCE_SECRET_KEY', '')
    BINANCE_TESTNET: bool = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    
    # === OPENAI API ===
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    # === TELEGRAM BOT ===
    TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # === TRADING CONFIGURATION ===
    MAX_LEVERAGE: int = int(os.getenv('MAX_LEVERAGE', '5'))
    RISK_PER_TRADE: float = float(os.getenv('RISK_PER_TRADE', '0.01'))
    STOP_LOSS_PERCENT: float = float(os.getenv('STOP_LOSS_PERCENT', '0.005'))
    TAKE_PROFIT_PERCENT: float = float(os.getenv('TAKE_PROFIT_PERCENT', '0.05'))
    
    # === DRY RUN MODE ===
    DRY_RUN: bool = os.getenv('DRY_RUN', 'false').lower() == 'true'
    
    # === LOGGING ===
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_TO_FILE: bool = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    
    # === TRADING SYMBOLS ===
    SYMBOLS: List[str] = [s.strip() for s in os.getenv('SYMBOLS', 'LINKUSDT').split(',') if s.strip()]
    
    # === TIMEFRAMES ===
    TIMEFRAME_EXECUTION: str = '1m'
    TIMEFRAME_CONFIRMATION: str = '5m'
    
    # === BINANCE API ENDPOINTS ===
    BINANCE_BASE_URL: str = 'https://testnet.binancefuture.com' if BINANCE_TESTNET else 'https://fapi.binance.com'
    BINANCE_WS_URL: str = 'wss://stream.binancefuture.com'
    
    # === RETRY SETTINGS ===
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # === RISK MANAGEMENT ===
    MAX_POSITION_SIZE: float =  4.5  # Max 50% of balance per position
    MIN_BALANCE_THRESHOLD: float = 0.05  # Stop trading if balance drops below 5%
    
    # === PERFORMANCE OPTIMIZATION ===
    BATCH_FETCH_SIZE: int = int(os.getenv('BATCH_FETCH_SIZE', '10'))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv('MAX_CONCURRENT_REQUESTS', '20'))
    CACHE_TTL_INDICATORS: int = int(os.getenv('CACHE_TTL_INDICATORS', '60'))
    CACHE_TTL_ADVISOR: int = int(os.getenv('CACHE_TTL_ADVISOR', '300'))
    ENABLE_PARALLEL_PROCESSING: bool = os.getenv('ENABLE_PARALLEL_PROCESSING', 'true').lower() == 'true'
    ENABLE_INDICATOR_CACHING: bool = os.getenv('ENABLE_INDICATOR_CACHING', 'true').lower() == 'true'
    ENABLE_CONNECTION_POOLING: bool = os.getenv('ENABLE_CONNECTION_POOLING', 'true').lower() == 'true'
    
    # === MEMORY MANAGEMENT ===
    MAX_CACHE_SIZE: int = int(os.getenv('MAX_CACHE_SIZE', '1000'))
    CACHE_CLEANUP_INTERVAL: int = int(os.getenv('CACHE_CLEANUP_INTERVAL', '300'))  # 5 minutes

    EXECUTION_INTERVAL: int = int(os.getenv('EXECUTION_INTERVAL', '15'))  # seconds between bot cycles
    CONFIRMATION_INTERVAL: int = int(os.getenv('CONFIRMATION_INTERVAL', '300'))  # seconds between advisor confirmations
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return any issues."""
        issues = {}
        
        if not cls.BINANCE_API_KEY:
            issues['BINANCE_API_KEY'] = 'Missing Binance API key'
        if not cls.BINANCE_SECRET_KEY:
            issues['BINANCE_SECRET_KEY'] = 'Missing Binance secret key'
        if not cls.OPENAI_API_KEY:
            issues['OPENAI_API_KEY'] = 'Missing OpenAI API key'
        if not cls.TELEGRAM_BOT_TOKEN:
            issues['TELEGRAM_BOT_TOKEN'] = 'Missing Telegram bot token'
        if not cls.TELEGRAM_CHAT_ID:
            issues['TELEGRAM_CHAT_ID'] = 'Missing Telegram chat ID'
            
        return issues
    
    @classmethod
    def is_valid(cls) -> bool:
        """Check if configuration is valid."""
        return len(cls.validate_config()) == 0
    
    @classmethod
    def get_binance_config(cls) -> Dict[str, Any]:
        """Get Binance-specific configuration."""
        return {
            'api_key': cls.BINANCE_API_KEY,
            'api_secret': cls.BINANCE_SECRET_KEY,
            'testnet': cls.BINANCE_TESTNET,
            'base_url': cls.BINANCE_BASE_URL,
            'ws_url': cls.BINANCE_WS_URL
        }

# Global config instance
config = Config()
