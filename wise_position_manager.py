"""
Wise Position Manager - Intelligent position sizing and risk management.
"""

import math
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

from config import config
from logger import logger


class WisePositionManager:
    """Intelligent position management system."""
    
    def __init__(self):
        """Initialize wise position manager."""
        self.correlation_matrix = {}
        self.position_history = []
        self.max_correlated_exposure = 0.3  # Max 30% in correlated assets
        
    def calculate_kelly_position_size(
        self, 
        balance: float, 
        win_rate: float = 0.6, 
        avg_win: float = 0.03, 
        avg_loss: float = 0.015,
        confidence: float = 0.75
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            balance: Available balance
            win_rate: Historical win rate (default 60%)
            avg_win: Average win percentage (default 3%)
            avg_loss: Average loss percentage (default 1.5%)
            confidence: Signal confidence (0-1)
            
        Returns:
            Optimal position size as fraction of balance
        """
        try:
            # Kelly formula: f = (bp - q) / b
            # where b = odds received (avg_win/avg_loss), p = win_rate, q = loss_rate
            b = avg_win / avg_loss  # Reward to risk ratio
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety multiplier and confidence scaling
            safety_multiplier = config.KELLY_MULTIPLIER if hasattr(config, 'KELLY_MULTIPLIER') else 0.25
            confidence_multiplier = confidence ** 2  # Square confidence for more conservative scaling
            
            optimal_fraction = kelly_fraction * safety_multiplier * confidence_multiplier
            
            # Cap at reasonable limits
            optimal_fraction = max(0.01, min(0.15, optimal_fraction))  # Between 1% and 15%
            
            logger.log_info(f"Kelly calculation: win_rate={win_rate:.2f}, ratio={b:.2f}, kelly={kelly_fraction:.3f}, final={optimal_fraction:.3f}")
            
            return optimal_fraction
            
        except Exception as e:
            logger.log_error(f"Error in Kelly calculation: {e}")
            return 0.02  # Default 2% position size

    def calculate_volatility_adjusted_size(
        self, 
        symbol: str, 
        base_size: float, 
        indicators: Dict[str, Any]
    ) -> float:
        """
        Adjust position size based on volatility.
        
        Args:
            symbol: Trading symbol
            base_size: Base position size
            indicators: Technical indicators
            
        Returns:
            Volatility-adjusted position size
        """
        try:
            # Get ATR (Average True Range) for volatility measure
            atr = indicators.get('atr')
            if atr is None or not hasattr(atr, 'iloc'):
                return base_size
            
            current_atr = float(atr.iloc[-1])
            atr_avg = float(atr.rolling(20).mean().iloc[-1])
            
            # Calculate volatility ratio
            volatility_ratio = current_atr / atr_avg if atr_avg > 0 else 1.0
            
            # Adjust size inversely to volatility
            if volatility_ratio > 1.5:  # High volatility
                adjustment = 0.7  # Reduce size by 30%
                reason = "high volatility"
            elif volatility_ratio > 1.2:  # Medium-high volatility
                adjustment = 0.85  # Reduce size by 15%
                reason = "medium-high volatility"
            elif volatility_ratio < 0.8:  # Low volatility
                adjustment = 1.15  # Increase size by 15%
                reason = "low volatility"
            else:  # Normal volatility
                adjustment = 1.0
                reason = "normal volatility"
            
            adjusted_size = base_size * adjustment
            
            if adjustment != 1.0:
                logger.log_info(f"Volatility adjustment for {symbol}: {adjustment:.2f}x due to {reason} (ATR ratio: {volatility_ratio:.2f})")
            
            return adjusted_size
            
        except Exception as e:
            logger.log_error(f"Error in volatility adjustment: {e}")
            return base_size

    def check_correlation_limits(
        self, 
        symbol: str, 
        current_positions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if new position would violate correlation limits.
        
        Args:
            symbol: Symbol to check
            current_positions: Current open positions
            
        Returns:
            Dictionary with correlation check results
        """
        try:
            # Define symbol correlations (simplified)
            correlation_groups = {
                'memecoins': ['DOGEUSDT', 'DOGSUSDT'],
                'layer1': ['SOLUSDT', 'NEARUSDT', 'LINKUSDT'],
                'altcoins': ['XRPUSDT', 'TRXUSDT', 'FISUSDT']
            }
            
            # Find which group the symbol belongs to
            symbol_group = None
            for group, symbols in correlation_groups.items():
                if symbol in symbols:
                    symbol_group = group
                    break
            
            if not symbol_group:
                return {'allowed': True, 'reason': 'No correlation group defined'}
            
            # Check current exposure in the same group
            group_symbols = correlation_groups[symbol_group]
            current_group_exposure = 0.0
            group_positions = []
            
            for pos_symbol, position in current_positions.items():
                if pos_symbol in group_symbols:
                    position_value = position.get('entry_price', 0) * position.get('quantity', 0)
                    current_group_exposure += position_value
                    group_positions.append(pos_symbol)
            
            # Calculate exposure as percentage of total balance
            total_balance = 100.0  # Approximate from logs
            exposure_percentage = current_group_exposure / total_balance
            
            if exposure_percentage > self.max_correlated_exposure:
                return {
                    'allowed': False,
                    'reason': f'Correlation limit exceeded: {exposure_percentage:.1%} in {symbol_group} group (max {self.max_correlated_exposure:.1%})',
                    'group': symbol_group,
                    'existing_positions': group_positions,
                    'exposure': exposure_percentage
                }
            
            return {
                'allowed': True,
                'reason': f'Correlation check passed: {exposure_percentage:.1%} in {symbol_group} group',
                'group': symbol_group,
                'exposure': exposure_percentage
            }
            
        except Exception as e:
            logger.log_error(f"Error in correlation check: {e}")
            return {'allowed': True, 'reason': 'Correlation check failed, allowing trade'}

    def calculate_wise_position_size(
        self, 
        symbol: str, 
        base_size: float, 
        signal_confidence: float,
        indicators: Dict[str, Any],
        current_positions: Dict[str, Any],
        balance: float
    ) -> Dict[str, Any]:
        """
        Calculate wise position size considering multiple factors.
        
        Args:
            symbol: Trading symbol
            base_size: Base position size from risk calculation
            signal_confidence: Signal confidence (0-1)
            indicators: Technical indicators
            current_positions: Current positions
            balance: Available balance
            
        Returns:
            Dictionary with wise position size and analysis
        """
        try:
            logger.log_info(f"WISE: Calculating wise position size for {symbol}")
            
            # 1. Kelly Criterion optimization
            kelly_fraction = self.calculate_kelly_position_size(
                balance=balance,
                confidence=signal_confidence
            )
            kelly_size = balance * kelly_fraction
            
            # 2. Volatility adjustment
            volatility_adjusted_size = self.calculate_volatility_adjusted_size(
                symbol=symbol,
                base_size=base_size,
                indicators=indicators
            )
            
            # 3. Correlation check
            correlation_check = self.check_correlation_limits(symbol, current_positions)
            
            # 4. Confidence scaling
            confidence_multiplier = min(1.2, signal_confidence * 1.5)  # Up to 20% boost for high confidence
            confidence_adjusted_size = base_size * confidence_multiplier
            
            # 5. Combine all factors (take most conservative)
            candidate_sizes = [
                ('base_risk', base_size),
                ('kelly_optimal', kelly_size),
                ('volatility_adjusted', volatility_adjusted_size),
                ('confidence_adjusted', confidence_adjusted_size)
            ]
            
            # Use the more conservative of Kelly and risk-based sizing
            final_size = min(kelly_size, volatility_adjusted_size, confidence_adjusted_size)
            
            # Apply correlation limits
            if not correlation_check['allowed']:
                final_size = 0.0
            
            analysis = {
                'symbol': symbol,
                'final_size': final_size,
                'confidence': signal_confidence,
                'kelly_fraction': kelly_fraction,
                'kelly_size': kelly_size,
                'volatility_adjusted': volatility_adjusted_size,
                'confidence_adjusted': confidence_adjusted_size,
                'correlation_check': correlation_check,
                'candidates': candidate_sizes,
                'wisdom_applied': True
            }
            
            # Log the wisdom decision
            if final_size > 0:
                logger.log_info(f"WISE: Position for {symbol}: {final_size:.2f} units (confidence: {signal_confidence:.2f}, Kelly: {kelly_fraction:.3f})")
            else:
                logger.log_warning(f"WISE: Position rejected for {symbol}: {correlation_check['reason']}")
            
            return analysis
            
        except Exception as e:
            logger.log_error(f"Error in wise position calculation: {e}")
            return {
                'final_size': base_size,
                'wisdom_applied': False,
                'error': str(e)
            }

    def get_position_health_score(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall portfolio health score.
        
        Args:
            positions: Current positions
            
        Returns:
            Portfolio health analysis
        """
        try:
            total_positions = len(positions)
            total_exposure = sum(
                pos.get('entry_price', 0) * pos.get('quantity', 0) 
                for pos in positions.values()
            )
            
            # Diversification score
            diversification_score = min(1.0, total_positions / 4.0)  # Optimal ~4 positions
            
            # Confidence score
            avg_confidence = sum(
                pos.get('confidence', 0.5) for pos in positions.values()
            ) / max(1, total_positions)
            
            # Risk balance score (prefer positions with good risk/reward)
            risk_scores = []
            for symbol, pos in positions.items():
                confidence = pos.get('confidence', 0.5)
                risk_scores.append(confidence)
            
            avg_risk_score = sum(risk_scores) / max(1, len(risk_scores))
            
            # Overall health score
            health_score = (diversification_score * 0.3 + avg_confidence * 0.4 + avg_risk_score * 0.3)
            
            return {
                'health_score': health_score,
                'total_positions': total_positions,
                'total_exposure': total_exposure,
                'avg_confidence': avg_confidence,
                'diversification_score': diversification_score,
                'recommendations': self._generate_health_recommendations(health_score, total_positions, avg_confidence)
            }
            
        except Exception as e:
            logger.log_error(f"Error calculating portfolio health: {e}")
            return {'health_score': 0.5, 'error': str(e)}

    def _generate_health_recommendations(self, health_score: float, total_positions: int, avg_confidence: float) -> List[str]:
        """Generate portfolio health recommendations."""
        recommendations = []
        
        if health_score < 0.4:
            recommendations.append("ALERT: Portfolio health is poor - consider reducing risk")
        elif health_score < 0.6:
            recommendations.append("WARNING: Portfolio health is moderate - monitor closely")
        else:
            recommendations.append("GOOD: Portfolio health is good")
        
        if total_positions > 6:
            recommendations.append("REDUCE: Too many positions - consider consolidation")
        elif total_positions < 2:
            recommendations.append("DIVERSIFY: Too few positions - consider diversification")
        
        if avg_confidence < 0.5:
            recommendations.append("SELECTIVE: Average signal confidence is low - be more selective")
        elif avg_confidence > 0.8:
            recommendations.append("EXCELLENT: High confidence signals - good strategy execution")
        
        return recommendations


# Global wise position manager
wise_position_manager = WisePositionManager()
