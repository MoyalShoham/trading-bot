"""
Enhanced logging module for the crypto trading bot with structured logging,
CSV exports, and proper error handling.
"""

import logging
import csv
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import traceback

from config import config


@dataclass
class TradeLog:
    """Structured trade log entry."""
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    order_id: str
    status: str
    pnl: float
    balance: float
    commission: Optional[float] = None
    commission_asset: Optional[str] = None


@dataclass
class SignalLog:
    """Structured signal log entry."""
    timestamp: str
    symbol: str
    signal: str
    confidence: float
    reason: str
    risk_level: str
    indicator_score: Optional[float] = None
    advisor_regime: Optional[str] = None


@dataclass
class ErrorLog:
    """Structured error log entry."""
    timestamp: str
    level: str
    module: str
    function: str
    message: str
    exception_type: Optional[str] = None
    traceback: Optional[str] = None


class TradingLogger:
    """
    Enhanced trading logger with structured logging and CSV exports.
    Provides different log levels and specialized logging for trades and signals.
    """
    
    def __init__(self, name: str = "trading_bot"):
        """Initialize the trading logger."""
        self.name = name
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # CSV loggers
        self._setup_csv_loggers()
        
        # Performance tracking
        self._start_time = datetime.now()
        self._log_counts = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'trade': 0,
            'signal': 0
        }

    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if config.LOG_TO_FILE:
            log_file = self.logs_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _setup_csv_loggers(self) -> None:
        """Setup CSV loggers for structured data."""
        self.trades_csv = self.logs_dir / "trades.csv"
        self.signals_csv = self.logs_dir / "signals.csv"
        self.errors_csv = self.logs_dir / "errors.csv"
        
        # Initialize CSV files with headers if they don't exist
        self._init_csv_file(self.trades_csv, TradeLog)
        self._init_csv_file(self.signals_csv, SignalLog)
        self._init_csv_file(self.errors_csv, ErrorLog)

    def _init_csv_file(self, file_path: Path, dataclass_type) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not file_path.exists():
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Get field names from dataclass
                fields = list(dataclass_type.__annotations__.keys())
                writer.writerow(fields)

    def _write_csv_log(self, file_path: Path, log_entry: Union[TradeLog, SignalLog, ErrorLog]) -> None:
        """Write a log entry to CSV file."""
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(asdict(log_entry).values())
        except Exception as e:
            self.logger.error(f"Failed to write CSV log: {e}")

    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self.logger.info(message)
        self._log_counts['info'] += 1
        
        if extra_data:
            self.logger.debug(f"Extra data: {json.dumps(extra_data, default=str)}")

    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self.logger.warning(message)
        self._log_counts['warning'] += 1
        
        if extra_data:
            self.logger.debug(f"Extra data: {json.dumps(extra_data, default=str)}")

    def error(self, message: str, exception: Optional[Exception] = None, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log error message with optional exception details."""
        self.logger.error(message)
        self._log_counts['error'] += 1
        
        # Create structured error log
        error_log = ErrorLog(
            timestamp=datetime.now().isoformat(),
            level="ERROR",
            module=self.name,
            function=traceback.extract_stack()[-2].name,
            message=message,
            exception_type=type(exception).__name__ if exception else None,
            traceback=traceback.format_exc() if exception else None
        )
        
        self._write_csv_log(self.errors_csv, error_log)
        
        if extra_data:
            self.logger.debug(f"Extra data: {json.dumps(extra_data, default=str)}")

    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self.logger.debug(message)
        
        if extra_data:
            self.logger.debug(f"Extra data: {json.dumps(extra_data, default=str)}")

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log trade execution to both main log and CSV."""
        try:
            trade_log = TradeLog(
                timestamp=datetime.now().isoformat(),
                symbol=trade_data.get('symbol', ''),
                side=trade_data.get('side', ''),
                quantity=float(trade_data.get('quantity', 0)),
                price=float(trade_data.get('price', 0)),
                order_id=str(trade_data.get('order_id', '')),
                status=trade_data.get('status', ''),
                pnl=float(trade_data.get('pnl', 0)),
                balance=float(trade_data.get('balance', 0)),
                commission=float(trade_data.get('commission', 0)) if trade_data.get('commission') else None,
                commission_asset=trade_data.get('commission_asset')
            )
            
            self._write_csv_log(self.trades_csv, trade_log)
            self._log_counts['trade'] += 1
            
            self.info(f"TRADE: {trade_log.side} {trade_log.quantity} {trade_log.symbol} @ {trade_log.price} | PnL: {trade_log.pnl}")
            
        except Exception as e:
            self.error(f"Failed to log trade: {e}", e)

    def log_signal(self, signal_data: Dict[str, Any]) -> None:
        """Log trading signal to both main log and CSV."""
        try:
            signal_log = SignalLog(
                timestamp=datetime.now().isoformat(),
                symbol=signal_data.get('symbol', ''),
                signal=signal_data.get('signal', ''),
                confidence=float(signal_data.get('confidence', 0)),
                reason=signal_data.get('reason', ''),
                risk_level=signal_data.get('risk_level', ''),
                indicator_score=float(signal_data.get('indicator_score', 0)) if signal_data.get('indicator_score') else None,
                advisor_regime=signal_data.get('advisor_regime', {}).get('regime') if isinstance(signal_data.get('advisor_regime'), dict) else None
            )
            
            self._write_csv_log(self.signals_csv, signal_log)
            self._log_counts['signal'] += 1
            
            self.info(f"SIGNAL: {signal_log.symbol} - {signal_log.signal} (confidence: {signal_log.confidence:.2f}) - {signal_log.reason}")
            
        except Exception as e:
            self.error(f"Failed to log signal: {e}", e)

    def log_performance_metrics(self) -> None:
        """Log performance metrics."""
        uptime = datetime.now() - self._start_time
        
        metrics = {
            'uptime': str(uptime),
            'log_counts': self._log_counts,
            'avg_logs_per_hour': {
                level: count / max(uptime.total_seconds() / 3600, 1)
                for level, count in self._log_counts.items()
            }
        }
        
        self.info(f"Performance metrics: {json.dumps(metrics, default=str)}")

    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades from CSV log."""
        try:
            trades = []
            if self.trades_csv.exists():
                with open(self.trades_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    trades = list(reader)[-limit:]  # Get last N trades
            return trades
        except Exception as e:
            self.error(f"Failed to get recent trades: {e}", e)
            return []

    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals from CSV log."""
        try:
            signals = []
            if self.signals_csv.exists():
                with open(self.signals_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    signals = list(reader)[-limit:]  # Get last N signals
            return signals
        except Exception as e:
            self.error(f"Failed to get recent signals: {e}", e)
            return []

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours."""
        try:
            if not self.errors_csv.exists():
                return {'total_errors': 0, 'error_types': {}, 'recent_errors': []}
            
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            error_types = {}
            recent_errors = []
            total_errors = 0
            
            with open(self.errors_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp']).timestamp()
                        if timestamp >= cutoff_time:
                            total_errors += 1
                            error_type = row.get('exception_type', 'Unknown')
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                            recent_errors.append(row)
                    except:
                        continue
            
            return {
                'total_errors': total_errors,
                'error_types': error_types,
                'recent_errors': recent_errors[-10:]  # Last 10 errors
            }
            
        except Exception as e:
            self.error(f"Failed to get error summary: {e}", e)
            return {'total_errors': 0, 'error_types': {}, 'recent_errors': []}


# Legacy compatibility functions for existing code
class LegacyLogger:
    """Legacy logger wrapper for backward compatibility."""
    
    def __init__(self):
        self.trading_logger = TradingLogger("legacy")

    def log_info(self, message: str) -> None:
        """Log info message (legacy)."""
        self.trading_logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message (legacy)."""
        self.trading_logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message (legacy)."""
        self.trading_logger.error(message)

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log trade (legacy)."""
        self.trading_logger.log_trade(trade_data)

    def log_signal(self, signal_data: Dict[str, Any]) -> None:
        """Log signal (legacy)."""
        self.trading_logger.log_signal(signal_data)


# Create logger instances
logger = LegacyLogger()
Logger = TradingLogger  # For new code
