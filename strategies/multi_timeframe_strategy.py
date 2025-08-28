"""
Multi-timeframe strategy for higher probability trades.
Combines multiple timeframes for better entry timing.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, RiskLevel, StrategyParameters
from config import config
from utils.logger import TradingLogger

logger = TradingLogger(__name__)


class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-timeframe strategy that increases win rate by requiring
    confluence across multiple timeframes.
    """
    
    def __init__(self):
        """Initialize multi-timeframe strategy."""
        parameters = StrategyParameters(
            timeframes=['1m', '5m', '15m', '1h'],  # Multiple timeframes
            indicators=['ema_9', 'ema_21', 'sma_20', 'sma_50', 'rsi', 'macd', 'bb_percent', 'atr'],
            lookback_periods={'short': 20, 'medium': 50, 'long': 200},
            thresholds={
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'trend_strength': 0.6,
                'confluence_required': 0.75  # 75% of timeframes must agree
            },
            risk_management={
                'max_risk_per_trade': 0.02,
                'min_reward_ratio': 2.0,  # 2:1 minimum reward:risk
                'volatility_adjustment': True
            }
        )
        
        super().__init__("MultiTimeframe Pro", parameters)

    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        current_position: Optional[Dict[str, Any]] = None
    ) -> TradingSignal:
        """Generate signal based on multi-timeframe analysis."""
        try:
            # Analyze each timeframe
            timeframe_signals = {}
            timeframe_strength = {}
            
            for tf in self.parameters.timeframes:
                tf_analysis = self._analyze_timeframe(tf, market_data, indicators)
                timeframe_signals[tf] = tf_analysis['signal']
                timeframe_strength[tf] = tf_analysis['strength']
            
            # Calculate confluence
            confluence = self._calculate_confluence(timeframe_signals, timeframe_strength)
            
            # Determine final signal
            final_signal = self._determine_final_signal(confluence, symbol)
            
            # Add to history
            self.add_signal_to_history(final_signal)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe strategy for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.NO_TRADE,
                confidence=0.0,
                risk_level=RiskLevel.HIGH,
                reason=f"Strategy error: {str(e)}"
            )

    def _analyze_timeframe(self, timeframe: str, market_data: Dict, indicators: Dict) -> Dict:
        """Analyze individual timeframe."""
        try:
            # Get timeframe-specific data
            tf_data = market_data.get(timeframe, {})
            if not tf_data:
                return {'signal': 'neutral', 'strength': 0.0}
            
            # Trend analysis
            trend_score = self._calculate_trend_score(indicators, timeframe)
            
            # Momentum analysis  
            momentum_score = self._calculate_momentum_score(indicators, timeframe)
            
            # Volatility analysis
            volatility_score = self._calculate_volatility_score(indicators, timeframe)
            
            # Combined score
            total_score = (trend_score * 0.5 + momentum_score * 0.3 + volatility_score * 0.2)
            
            # Determine signal
            if total_score > 0.6:
                signal = 'bullish'
            elif total_score < -0.6:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            return {
                'signal': signal,
                'strength': abs(total_score),
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'volatility_score': volatility_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {timeframe}: {e}")
            return {'signal': 'neutral', 'strength': 0.0}

    def _calculate_trend_score(self, indicators: Dict, timeframe: str) -> float:
        """Calculate trend strength for timeframe."""
        try:
            score = 0.0
            
            # EMA alignment
            ema_9 = indicators.get('ema_9')
            ema_21 = indicators.get('ema_21')
            
            if ema_9 and ema_21:
                if hasattr(ema_9, 'iloc') and hasattr(ema_21, 'iloc'):
                    ema_9_val = float(ema_9.iloc[-1])
                    ema_21_val = float(ema_21.iloc[-1])
                    
                    if ema_9_val > ema_21_val:
                        score += 0.3
                    else:
                        score -= 0.3
            
            # SMA alignment
            sma_20 = indicators.get('sma_20')
            sma_50 = indicators.get('sma_50')
            
            if sma_20 and sma_50:
                if hasattr(sma_20, 'iloc') and hasattr(sma_50, 'iloc'):
                    sma_20_val = float(sma_20.iloc[-1])
                    sma_50_val = float(sma_50.iloc[-1])
                    
                    if sma_20_val > sma_50_val:
                        score += 0.2
                    else:
                        score -= 0.2
            
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.0

    def _calculate_momentum_score(self, indicators: Dict, timeframe: str) -> float:
        """Calculate momentum strength."""
        try:
            score = 0.0
            
            # RSI analysis
            rsi = indicators.get('rsi')
            if rsi and hasattr(rsi, 'iloc'):
                rsi_val = float(rsi.iloc[-1])
                
                if rsi_val > 70:
                    score -= 0.4  # Overbought
                elif rsi_val < 30:
                    score += 0.4  # Oversold
                elif 45 < rsi_val < 55:
                    score += 0.1  # Neutral momentum
            
            # MACD analysis
            macd_line = indicators.get('macd_line')
            macd_signal = indicators.get('macd_signal')
            
            if macd_line and macd_signal:
                if hasattr(macd_line, 'iloc') and hasattr(macd_signal, 'iloc'):
                    macd_val = float(macd_line.iloc[-1])
                    signal_val = float(macd_signal.iloc[-1])
                    
                    if macd_val > signal_val:
                        score += 0.3
                    else:
                        score -= 0.3
            
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.0

    def _calculate_volatility_score(self, indicators: Dict, timeframe: str) -> float:
        """Calculate volatility-adjusted score."""
        try:
            # Bollinger Band position
            bb_percent = indicators.get('bb_percent')
            if bb_percent and hasattr(bb_percent, 'iloc'):
                bb_val = float(bb_percent.iloc[-1])
                
                if bb_val < 0.2:
                    return 0.3  # Near lower band - potential bounce
                elif bb_val > 0.8:
                    return -0.3  # Near upper band - potential reversal
                else:
                    return 0.0  # Middle range
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {e}")
            return 0.0

    def _calculate_confluence(self, timeframe_signals: Dict, timeframe_strength: Dict) -> Dict:
        """Calculate confluence across timeframes."""
        try:
            # Weight timeframes by importance
            timeframe_weights = {
                '1m': 0.1,   # Short-term noise
                '5m': 0.2,   # Entry timing
                '15m': 0.3,  # Trend confirmation
                '1h': 0.4    # Primary trend
            }
            
            bullish_strength = 0.0
            bearish_strength = 0.0
            total_weight = 0.0
            
            for tf, signal in timeframe_signals.items():
                weight = timeframe_weights.get(tf, 0.1)
                strength = timeframe_strength.get(tf, 0.0)
                
                if signal == 'bullish':
                    bullish_strength += weight * strength
                elif signal == 'bearish':
                    bearish_strength += weight * strength
                
                total_weight += weight
            
            # Normalize
            if total_weight > 0:
                bullish_strength /= total_weight
                bearish_strength /= total_weight
            
            # Calculate confluence percentage
            total_strength = bullish_strength + bearish_strength
            confluence_threshold = self.parameters.thresholds['confluence_required']
            
            return {
                'bullish_strength': bullish_strength,
                'bearish_strength': bearish_strength,
                'net_strength': bullish_strength - bearish_strength,
                'confluence_met': total_strength >= confluence_threshold,
                'dominant_direction': 'bullish' if bullish_strength > bearish_strength else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"Error calculating confluence: {e}")
            return {
                'bullish_strength': 0.0,
                'bearish_strength': 0.0,
                'net_strength': 0.0,
                'confluence_met': False,
                'dominant_direction': 'neutral'
            }

    def _determine_final_signal(self, confluence: Dict, symbol: str) -> TradingSignal:
        """Determine final trading signal based on confluence."""
        try:
            if not confluence['confluence_met']:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.NO_TRADE,
                    confidence=0.0,
                    risk_level=RiskLevel.HIGH,
                    reason="Insufficient timeframe confluence"
                )
            
            net_strength = confluence['net_strength']
            direction = confluence['dominant_direction']
            
            # Determine signal type
            if direction == 'bullish' and net_strength > 0.3:
                signal_type = SignalType.LONG
                confidence = min(0.9, net_strength * 1.5)
                risk_level = RiskLevel.LOW if net_strength > 0.6 else RiskLevel.MEDIUM
                reason = f"Strong bullish confluence across timeframes (strength: {net_strength:.2f})"
                
            elif direction == 'bearish' and net_strength < -0.3:
                signal_type = SignalType.SHORT
                confidence = min(0.9, abs(net_strength) * 1.5)
                risk_level = RiskLevel.LOW if abs(net_strength) > 0.6 else RiskLevel.MEDIUM
                reason = f"Strong bearish confluence across timeframes (strength: {abs(net_strength):.2f})"
                
            else:
                signal_type = SignalType.NO_TRADE
                confidence = 0.0
                risk_level = RiskLevel.HIGH
                reason = "Weak or conflicting signals across timeframes"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                risk_level=risk_level,
                reason=reason,
                metadata={
                    'confluence': confluence,
                    'strategy': 'multi_timeframe',
                    'timeframes_analyzed': len(self.parameters.timeframes)
                }
            )
            
        except Exception as e:
            logger.error(f"Error determining final signal: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.NO_TRADE,
                confidence=0.0,
                risk_level=RiskLevel.HIGH,
                reason=f"Signal determination error: {str(e)}"
            )

    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate the trading signal."""
        if not signal or signal.signal_type == SignalType.NO_TRADE:
            return signal is not None
        
        # Check confidence threshold
        if signal.confidence < self.parameters.thresholds.get('min_confidence', 0.3):
            return False
        
        # Check risk level
        if signal.risk_level == RiskLevel.HIGH:
            return False
        
        return True
