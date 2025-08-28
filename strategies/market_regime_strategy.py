"""
Market regime-based strategy that adapts to different market conditions.
"""

from typing import Dict, Optional, Any
from enum import Enum
import numpy as np

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, RiskLevel, StrategyParameters
from utils.logger import TradingLogger

logger = TradingLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down" 
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"


class MarketRegimeStrategy(BaseStrategy):
    """Strategy that adapts based on current market regime."""
    
    def __init__(self):
        """Initialize market regime strategy."""
        parameters = StrategyParameters(
            timeframes=['5m', '15m', '1h'],
            indicators=['atr', 'bb_width', 'adx', 'rsi', 'macd', 'volume_ratio'],
            lookback_periods={'atr': 14, 'adx': 14, 'bb': 20},
            thresholds={
                'trend_strength': 25,  # ADX threshold
                'volatility_high': 2.0,  # ATR multiplier
                'volatility_low': 0.5,
                'breakout_volume': 1.5  # Volume surge
            },
            risk_management={
                'trending_risk': 0.025,     # Higher risk in trends
                'sideways_risk': 0.015,     # Lower risk in chop
                'breakout_risk': 0.035      # Highest risk in breakouts
            }
        )
        
        super().__init__("Market Regime Adaptive", parameters)

    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        current_position: Optional[Dict[str, Any]] = None
    ) -> TradingSignal:
        """Generate regime-adaptive signal."""
        try:
            # Detect current market regime
            regime = self._detect_market_regime(indicators)
            
            # Generate signal based on regime
            signal = self._generate_regime_signal(symbol, regime, indicators, market_data)
            
            # Add to history
            self.add_signal_to_history(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in regime strategy for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.NO_TRADE,
                confidence=0.0,
                risk_level=RiskLevel.HIGH,
                reason=f"Regime strategy error: {str(e)}"
            )

    def _detect_market_regime(self, indicators: Dict) -> MarketRegime:
        """Detect current market regime."""
        try:
            # ATR for volatility
            atr = indicators.get('atr')
            atr_val = float(atr.iloc[-1]) if atr is not None and hasattr(atr, 'iloc') else 0
            atr_avg = float(atr.rolling(20).mean().iloc[-1]) if atr is not None and hasattr(atr, 'iloc') else atr_val
            
            # ADX for trend strength
            adx = indicators.get('adx', indicators.get('atr'))  # Fallback if no ADX
            adx_val = float(adx.iloc[-1]) if adx is not None and hasattr(adx, 'iloc') else 20
            
            # Bollinger Band width for volatility
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            bb_width = 0
            if bb_upper is not None and bb_lower is not None:
                if hasattr(bb_upper, 'iloc') and hasattr(bb_lower, 'iloc'):
                    bb_width = (float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1])) / float(bb_upper.iloc[-1])
            
            # Volume analysis
            volume_change = indicators.get('volume_change', 1.0)
            if hasattr(volume_change, 'iloc'):
                volume_change = float(volume_change.iloc[-1])
            
            # Determine regime
            volatility_ratio = atr_val / atr_avg if atr_avg > 0 else 1.0
            
            # High volatility regime
            if volatility_ratio > self.parameters.thresholds['volatility_high']:
                if volume_change > self.parameters.thresholds['breakout_volume']:
                    return MarketRegime.BREAKOUT
                else:
                    return MarketRegime.HIGH_VOLATILITY
            
            # Low volatility regime
            elif volatility_ratio < self.parameters.thresholds['volatility_low']:
                return MarketRegime.LOW_VOLATILITY
            
            # Trending regimes
            elif adx_val > self.parameters.thresholds['trend_strength']:
                # Check trend direction using EMAs
                ema_9 = indicators.get('ema_9')
                ema_21 = indicators.get('ema_21')
                
                if ema_9 is not None and ema_21 is not None:
                    if hasattr(ema_9, 'iloc') and hasattr(ema_21, 'iloc'):
                        if float(ema_9.iloc[-1]) > float(ema_21.iloc[-1]):
                            return MarketRegime.TRENDING_UP
                        else:
                            return MarketRegime.TRENDING_DOWN
            
            # Default to sideways
            return MarketRegime.SIDEWAYS
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS

    def _generate_regime_signal(
        self,
        symbol: str,
        regime: MarketRegime,
        indicators: Dict,
        market_data: Dict
    ) -> TradingSignal:
        """Generate signal based on market regime."""
        try:
            if regime == MarketRegime.TRENDING_UP:
                return self._trending_up_signal(symbol, indicators)
            elif regime == MarketRegime.TRENDING_DOWN:
                return self._trending_down_signal(symbol, indicators)
            elif regime == MarketRegime.BREAKOUT:
                return self._breakout_signal(symbol, indicators, market_data)
            elif regime == MarketRegime.SIDEWAYS:
                return self._mean_reversion_signal(symbol, indicators)
            else:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.NO_TRADE,
                    confidence=0.0,
                    risk_level=RiskLevel.HIGH,
                    reason=f"Avoiding {regime.value} market conditions"
                )
                
        except Exception as e:
            logger.error(f"Error generating regime signal: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.NO_TRADE,
                confidence=0.0,
                risk_level=RiskLevel.HIGH,
                reason=f"Regime signal error: {str(e)}"
            )

    def _trending_up_signal(self, symbol: str, indicators: Dict) -> TradingSignal:
        """Generate signal for uptrending market."""
        try:
            # Look for pullbacks in uptrend
            rsi = indicators.get('rsi')
            ema_9 = indicators.get('ema_9')
            ema_21 = indicators.get('ema_21')
            
            if not all([rsi, ema_9, ema_21]):
                return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="Missing indicators")
            
            rsi_val = float(rsi.iloc[-1])
            ema_9_val = float(ema_9.iloc[-1])
            ema_21_val = float(ema_21.iloc[-1])
            
            # Buy pullbacks in uptrend
            if 30 < rsi_val < 50 and ema_9_val > ema_21_val:
                confidence = 0.7 + (50 - rsi_val) / 100  # Higher confidence on deeper pullbacks
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    confidence=confidence,
                    risk_level=RiskLevel.MEDIUM,
                    reason=f"Uptrend pullback buy opportunity (RSI: {rsi_val:.1f})",
                    metadata={'regime': 'trending_up', 'rsi': rsi_val}
                )
            
            return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="No pullback opportunity")
            
        except Exception as e:
            logger.error(f"Error in trending up signal: {e}")
            return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="Signal generation error")

    def _trending_down_signal(self, symbol: str, indicators: Dict) -> TradingSignal:
        """Generate signal for downtrending market."""
        try:
            # Look for bounces in downtrend
            rsi = indicators.get('rsi')
            ema_9 = indicators.get('ema_9')
            ema_21 = indicators.get('ema_21')
            
            if not all([rsi, ema_9, ema_21]):
                return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="Missing indicators")
            
            rsi_val = float(rsi.iloc[-1])
            ema_9_val = float(ema_9.iloc[-1])
            ema_21_val = float(ema_21.iloc[-1])
            
            # Short bounces in downtrend
            if 50 < rsi_val < 70 and ema_9_val < ema_21_val:
                confidence = 0.7 + (rsi_val - 50) / 100  # Higher confidence on higher bounces
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    confidence=confidence,
                    risk_level=RiskLevel.MEDIUM,
                    reason=f"Downtrend bounce short opportunity (RSI: {rsi_val:.1f})",
                    metadata={'regime': 'trending_down', 'rsi': rsi_val}
                )
            
            return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="No bounce opportunity")
            
        except Exception as e:
            logger.error(f"Error in trending down signal: {e}")
            return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="Signal generation error")

    def _breakout_signal(self, symbol: str, indicators: Dict, market_data: Dict) -> TradingSignal:
        """Generate signal for breakout conditions."""
        try:
            # Look for volume-confirmed breakouts
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            volume_change = indicators.get('volume_change', 1.0)
            
            if not all([bb_upper, bb_lower]):
                return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="Missing BB indicators")
            
            # Get current price from market data
            current_price = None
            if '5m' in market_data:
                current_price = market_data['5m'].get('close', 0)
            
            if not current_price:
                return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="No current price")
            
            bb_upper_val = float(bb_upper.iloc[-1])
            bb_lower_val = float(bb_lower.iloc[-1])
            volume_val = float(volume_change.iloc[-1]) if hasattr(volume_change, 'iloc') else volume_change
            
            # Breakout conditions
            if current_price > bb_upper_val and volume_val > 1.5:
                confidence = min(0.8, 0.5 + volume_val / 10)
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    confidence=confidence,
                    risk_level=RiskLevel.MEDIUM,
                    reason=f"Bullish breakout with volume confirmation ({volume_val:.1f}x)",
                    metadata={'regime': 'breakout', 'direction': 'bullish', 'volume': volume_val}
                )
            elif current_price < bb_lower_val and volume_val > 1.5:
                confidence = min(0.8, 0.5 + volume_val / 10)
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    confidence=confidence,
                    risk_level=RiskLevel.MEDIUM,
                    reason=f"Bearish breakout with volume confirmation ({volume_val:.1f}x)",
                    metadata={'regime': 'breakout', 'direction': 'bearish', 'volume': volume_val}
                )
            
            return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="No confirmed breakout")
            
        except Exception as e:
            logger.error(f"Error in breakout signal: {e}")
            return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="Breakout signal error")

    def _mean_reversion_signal(self, symbol: str, indicators: Dict) -> TradingSignal:
        """Generate signal for mean reversion in sideways market."""
        try:
            # Look for oversold/overbought in ranging market
            rsi = indicators.get('rsi')
            bb_percent = indicators.get('bb_percent')
            
            if not all([rsi, bb_percent]):
                return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="Missing mean reversion indicators")
            
            rsi_val = float(rsi.iloc[-1])
            bb_val = float(bb_percent.iloc[-1])
            
            # Mean reversion opportunities
            if rsi_val < 25 and bb_val < 0.2:  # Oversold
                confidence = 0.6 + (25 - rsi_val) / 50
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    confidence=confidence,
                    risk_level=RiskLevel.LOW,
                    reason=f"Mean reversion long (RSI: {rsi_val:.1f}, BB: {bb_val:.2f})",
                    metadata={'regime': 'mean_reversion', 'rsi': rsi_val, 'bb_percent': bb_val}
                )
            elif rsi_val > 75 and bb_val > 0.8:  # Overbought
                confidence = 0.6 + (rsi_val - 75) / 50
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    confidence=confidence,
                    risk_level=RiskLevel.LOW,
                    reason=f"Mean reversion short (RSI: {rsi_val:.1f}, BB: {bb_val:.2f})",
                    metadata={'regime': 'mean_reversion', 'rsi': rsi_val, 'bb_percent': bb_val}
                )
            
            return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="No mean reversion opportunity")
            
        except Exception as e:
            logger.error(f"Error in mean reversion signal: {e}")
            return TradingSignal(symbol=symbol, signal_type=SignalType.NO_TRADE, confidence=0.0, risk_level=RiskLevel.HIGH, reason="Mean reversion error")

    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate regime-based signal."""
        if not signal or signal.signal_type == SignalType.NO_TRADE:
            return signal is not None
        
        # Check minimum confidence
        if signal.confidence < 0.4:
            return False
        
        # Regime-specific validation
        regime = signal.metadata.get('regime') if signal.metadata else None
        
        if regime == 'breakout' and signal.confidence < 0.5:
            return False
        elif regime == 'mean_reversion' and signal.risk_level == RiskLevel.HIGH:
            return False
        
        return True
