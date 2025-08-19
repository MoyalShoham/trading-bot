"""
Logging module for structured JSON/CSV logging with timestamps.
Tracks signals, advisor outputs, trades, and PnL.
"""

import json
import csv
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import os

from config import config

class TradingLogger:
    """Structured logger for trading bot activities."""
    
    def __init__(self):
        """Initialize the trading logger."""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        self._setup_file_logging()
        
        # Setup CSV logging for trades
        self._setup_csv_logging()
    
    def _setup_file_logging(self):
        """Setup file-based logging."""
        log_file = self.log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('TradingBot')
    
    def _setup_csv_logging(self):
        """Setup CSV logging for structured data."""
        self.trades_file = self.log_dir / "trades.csv"
        self.signals_file = self.log_dir / "signals.csv"
        self.advisor_file = self.log_dir / "advisor.csv"
        
        # Create CSV files with headers if they don't exist
        self._create_csv_files()
    
    def _create_csv_files(self):
        """Create CSV files with appropriate headers."""
        # Trades CSV
        if not self.trades_file.exists():
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price', 
                    'order_id', 'status', 'pnl', 'balance'
                ])
        
        # Signals CSV
        if not self.signals_file.exists():
            with open(self.signals_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'signal', 'strength', 'indicators', 
                    'advisor_regime', 'confidence'
                ])
        
        # Advisor CSV
        if not self.advisor_file.exists():
            with open(self.advisor_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'regime', 'factors', 'note', 
                    'confidence', 'response_time'
                ])
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log a trade execution."""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                trade_data.get('symbol', ''),
                trade_data.get('side', ''),
                trade_data.get('quantity', ''),
                trade_data.get('price', ''),
                trade_data.get('order_id', ''),
                trade_data.get('status', ''),
                trade_data.get('pnl', ''),
                trade_data.get('balance', '')
            ])
        
        # Log to file
        self.logger.info(f"Trade executed: {json.dumps(trade_data, indent=2)}")
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log a trading signal."""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        with open(self.signals_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                signal_data.get('symbol', ''),
                signal_data.get('signal', ''),
                signal_data.get('strength', ''),
                json.dumps(signal_data.get('indicators', {})),
                signal_data.get('advisor_regime', ''),
                signal_data.get('confidence', '')
            ])
        
        # Log to file
        self.logger.info(f"Signal generated: {json.dumps(signal_data, indent=2)}")
    
    def log_advisor(self, advisor_data: Dict[str, Any]):
        """Log advisor response."""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        with open(self.advisor_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                advisor_data.get('symbol', ''),
                advisor_data.get('regime', ''),
                json.dumps(advisor_data.get('factors', [])),
                advisor_data.get('note', ''),
                advisor_data.get('confidence', ''),
                advisor_data.get('response_time', '')
            ])
        
        # Log to file
        self.logger.info(f"Advisor response: {json.dumps(advisor_data, indent=2)}")
    
    def log_error(self, error_msg: str, error_data: Optional[Dict[str, Any]] = None):
        """Log an error."""
        if error_data:
            self.logger.error(f"{error_msg}: {json.dumps(error_data, indent=2)}")
        else:
            self.logger.error(error_msg)
    
    def log_info(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        if data:
            self.logger.info(f"{message}: {json.dumps(data, indent=2)}")
        else:
            self.logger.info(message)
    
    def log_warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        if data:
            self.logger.warning(f"{message}: {json.dumps(data, indent=2)}")
        else:
            self.logger.warning(message)
    
    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> list:
        """Get trade history from CSV."""
        trades = []
        with open(self.trades_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if symbol is None or row['symbol'] == symbol:
                    trades.append(row)
                    if len(trades) >= limit:
                        break
        
        return trades
    
    def get_signal_history(self, symbol: Optional[str] = None, limit: int = 100) -> list:
        """Get signal history from CSV."""
        signals = []
        with open(self.signals_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if symbol is None or row['symbol'] == symbol:
                    signals.append(row)
                    if len(signals) >= limit:
                        break
        
        return signals

# Global logger instance
logger = TradingLogger()
