"""
Base strategy class providing a framework for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from config import config
from utils.logger import TradingLogger

logger = TradingLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NO_TRADE = "no_trade"


class RiskLevel(Enum):
    """Risk levels for trading signals."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TradingSignal:
    """Structured trading signal."""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    risk_level: RiskLevel
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    timestamp: datetime = None
    reason: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format."""
        return {
            'symbol': self.symbol,
            'signal': self.signal_type.value,
            'confidence': self.confidence,
            'risk_level': self.risk_level.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'quantity': self.quantity,
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason,
            'metadata': self.metadata
        }

    def is_valid(self) -> bool:
        """Validate the trading signal."""
        if not self.symbol or not isinstance(self.confidence, (int, float)):
            return False
        
        if not 0.0 <= self.confidence <= 1.0:
            return False
        
        if self.signal_type == SignalType.NO_TRADE:
            return True
        
        # For actual trade signals, we need entry price
        if self.entry_price is None or self.entry_price <= 0:
            return False
        
        # Validate stop loss and take profit if provided
        if self.stop_loss is not None:
            if self.signal_type in [SignalType.LONG, SignalType.CLOSE_SHORT]:
                if self.stop_loss >= self.entry_price:  # Stop loss should be below entry for long
                    return False
            elif self.signal_type in [SignalType.SHORT, SignalType.CLOSE_LONG]:
                if self.stop_loss <= self.entry_price:  # Stop loss should be above entry for short
                    return False
        
        return True


@dataclass
class StrategyParameters:
    """Strategy configuration parameters."""
    timeframes: List[str]
    indicators: List[str]
    lookback_periods: Dict[str, int]
    thresholds: Dict[str, float]
    risk_management: Dict[str, Any]
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Provides common functionality and enforces strategy interface.
    """
    
    def __init__(self, name: str, parameters: StrategyParameters):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy configuration parameters
        """
        self.name = name
        self.parameters = parameters
        self.is_active = True
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'last_updated': datetime.now()
        }
        
        # Strategy state
        self.last_signals = {}  # symbol -> last signal
        self.position_states = {}  # symbol -> position state
        self.signal_history = []  # Historical signals
        
        logger.info(f"Initialized strategy: {self.name}")

    @abstractmethod
    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        current_position: Optional[Dict[str, Any]] = None
    ) -> TradingSignal:
        """
        Generate trading signal for a given symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            indicators: Technical indicators
            current_position: Current position (if any)
            
        Returns:
            TradingSignal object
        """
        pass

    @abstractmethod
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators for this strategy.
        
        Returns:
            List of indicator names
        """
        return self.parameters.indicators

    def get_required_timeframes(self) -> List[str]:
        """
        Get list of required timeframes for this strategy.
        
        Returns:
            List of timeframe strings
        """
        return self.parameters.timeframes

    def update_performance(self, signal: TradingSignal, result: Dict[str, Any]) -> None:
        """
        Update strategy performance metrics.
        
        Args:
            signal: The trading signal that was executed
            result: Execution result with PnL and other metrics
        """
        try:
            self.performance_metrics['total_signals'] += 1
            
            pnl = result.get('pnl', 0)
            if pnl > 0:
                self.performance_metrics['successful_signals'] += 1
            
            # Update win rate
            total = self.performance_metrics['total_signals']
            successful = self.performance_metrics['successful_signals']
            self.performance_metrics['win_rate'] = successful / total if total > 0 else 0.0
            
            # Update average return (simplified)
            current_avg = self.performance_metrics['avg_return']
            self.performance_metrics['avg_return'] = (current_avg * (total - 1) + pnl) / total
            
            self.performance_metrics['last_updated'] = datetime.now()
            
            logger.debug(f"Updated performance for {self.name}: {self.performance_metrics}")
            
        except Exception as e:
            logger.error(f"Error updating performance for {self.name}: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get strategy performance summary.
        
        Returns:
            Performance metrics dictionary
        """
        return {
            'strategy_name': self.name,
            'is_active': self.is_active,
            'performance': self.performance_metrics.copy(),
            'parameters': {
                'timeframes': self.parameters.timeframes,
                'indicators': self.parameters.indicators,
                'lookback_periods': self.parameters.lookback_periods,
                'thresholds': self.parameters.thresholds
            },
            'recent_signals': len([s for s in self.signal_history if 
                                (datetime.now() - s.timestamp).days < 1])
        }

    def add_signal_to_history(self, signal: TradingSignal) -> None:
        """
        Add signal to history for tracking.
        
        Args:
            signal: Signal to add to history
        """
        self.signal_history.append(signal)
        self.last_signals[signal.symbol] = signal
        
        # Keep only recent history (last 1000 signals)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]

    def get_last_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Get the last signal for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Last TradingSignal for the symbol or None
        """
        return self.last_signals.get(symbol)

    def can_generate_signal(self, symbol: str, cooldown_minutes: int = 15) -> bool:
        """
        Check if enough time has passed to generate a new signal.
        
        Args:
            symbol: Trading symbol
            cooldown_minutes: Minimum minutes between signals
            
        Returns:
            True if can generate signal, False otherwise
        """
        last_signal = self.get_last_signal(symbol)
        if last_signal is None:
            return True
        
        time_diff = datetime.now() - last_signal.timestamp
        return time_diff.total_seconds() >= (cooldown_minutes * 60)

    def calculate_position_size(
        self,
        signal: TradingSignal,
        account_balance: float,
        risk_per_trade: float = None
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            risk_per_trade: Risk per trade as fraction of balance
            
        Returns:
            Position size
        """
        if risk_per_trade is None:
            risk_per_trade = config.RISK_PER_TRADE
        
        if signal.stop_loss is None or signal.entry_price is None:
            logger.warning(f"Cannot calculate position size without stop loss and entry price")
            return 0.0
        
        risk_amount = account_balance * risk_per_trade
        price_diff = abs(signal.entry_price - signal.stop_loss)
        
        if price_diff == 0:
            return 0.0
        
        position_size = risk_amount / price_diff
        
        # Apply confidence adjustment
        confidence_multiplier = 0.5 + (signal.confidence * 0.5)  # 0.5x to 1.0x based on confidence
        position_size *= confidence_multiplier
        
        # Apply risk level adjustment
        risk_multipliers = {
            RiskLevel.LOW: 1.2,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.8
        }
        position_size *= risk_multipliers.get(signal.risk_level, 1.0)
        
        return position_size

    def log_signal(self, signal: TradingSignal) -> None:
        """
        Log the generated signal.
        
        Args:
            signal: Signal to log
        """
        try:
            logger.log_signal(signal.to_dict())
            logger.info(f"{self.name} generated {signal.signal_type.value} signal for {signal.symbol} "
                       f"(confidence: {signal.confidence:.2f}, risk: {signal.risk_level.value})")
        except Exception as e:
            logger.error(f"Error logging signal: {e}")

    def activate(self) -> None:
        """Activate the strategy."""
        self.is_active = True
        logger.info(f"Strategy {self.name} activated")

    def deactivate(self) -> None:
        """Deactivate the strategy."""
        self.is_active = False
        logger.info(f"Strategy {self.name} deactivated")

    def reset_performance(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'last_updated': datetime.now()
        }
        self.signal_history.clear()
        self.last_signals.clear()
        logger.info(f"Performance metrics reset for {self.name}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get strategy configuration.
        
        Returns:
            Strategy configuration dictionary
        """
        return {
            'name': self.name,
            'is_active': self.is_active,
            'parameters': {
                'timeframes': self.parameters.timeframes,
                'indicators': self.parameters.indicators,
                'lookback_periods': self.parameters.lookback_periods,
                'thresholds': self.parameters.thresholds,
                'risk_management': self.parameters.risk_management,
                'custom_params': self.parameters.custom_params
            }
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update strategy configuration.
        
        Args:
            new_config: New configuration dictionary
        """
        try:
            if 'thresholds' in new_config:
                self.parameters.thresholds.update(new_config['thresholds'])
            
            if 'custom_params' in new_config:
                self.parameters.custom_params.update(new_config['custom_params'])
            
            if 'risk_management' in new_config:
                self.parameters.risk_management.update(new_config['risk_management'])
            
            logger.info(f"Configuration updated for {self.name}")
            
        except Exception as e:
            logger.error(f"Error updating configuration for {self.name}: {e}")

    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"Strategy({self.name}, active={self.is_active}, win_rate={self.performance_metrics['win_rate']:.2%})"

    def __repr__(self) -> str:
        """Detailed string representation of the strategy."""
        return (f"Strategy(name='{self.name}', active={self.is_active}, "
                f"signals={self.performance_metrics['total_signals']}, "
                f"win_rate={self.performance_metrics['win_rate']:.2%})")
