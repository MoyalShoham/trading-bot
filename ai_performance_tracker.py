"""
AI Performance Tracker - Learn from trading results to improve AI recommendations.
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque

from logger import logger


class AIPerformanceTracker:
    """Track AI recommendations vs actual trading results to improve future predictions."""
    
    def __init__(self, max_history: int = 100):
        """Initialize performance tracker."""
        self.max_history = max_history
        self.prediction_history = deque(maxlen=max_history)
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'regime_accuracy': {},
            'confidence_calibration': {},
            'recent_performance': 0.0
        }
    
    def track_prediction(
        self, 
        symbol: str, 
        predicted_regime: str, 
        confidence: float, 
        indicators: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> str:
        """
        Track a new AI prediction for later validation.
        
        Args:
            symbol: Trading symbol
            predicted_regime: AI predicted regime
            confidence: AI confidence score
            indicators: Market indicators at time of prediction
            timestamp: Prediction timestamp
            
        Returns:
            Prediction ID for later tracking
        """
        prediction_id = f"{symbol}_{int(time.time())}"
        
        prediction = {
            'id': prediction_id,
            'symbol': symbol,
            'predicted_regime': predicted_regime,
            'confidence': confidence,
            'indicators': indicators.copy(),
            'timestamp': timestamp or datetime.now().isoformat(),
            'actual_outcome': None,
            'profit_pnl': None,
            'validation_complete': False
        }
        
        self.prediction_history.append(prediction)
        self.performance_stats['total_predictions'] += 1
        
        logger.log_info(f"AI_TRACKER: Tracking prediction {prediction_id} - {predicted_regime} (conf: {confidence:.2f})")
        
        return prediction_id
    
    def validate_prediction(
        self, 
        prediction_id: str, 
        actual_outcome: str, 
        profit_pnl: float
    ) -> bool:
        """
        Validate a prediction with actual trading results.
        
        Args:
            prediction_id: ID from track_prediction
            actual_outcome: What actually happened ('profit', 'loss', 'breakeven')
            profit_pnl: Actual profit/loss amount
            
        Returns:
            True if validation successful
        """
        try:
            # Find the prediction
            for prediction in self.prediction_history:
                if prediction['id'] == prediction_id:
                    prediction['actual_outcome'] = actual_outcome
                    prediction['profit_pnl'] = profit_pnl
                    prediction['validation_complete'] = True
                    
                    # Update performance stats
                    self._update_performance_stats(prediction)
                    
                    logger.log_info(f"AI_TRACKER: Validated {prediction_id} - {actual_outcome} (PnL: {profit_pnl:.2f})")
                    return True
            
            logger.log_warning(f"AI_TRACKER: Prediction {prediction_id} not found for validation")
            return False
            
        except Exception as e:
            logger.log_error(f"Error validating prediction: {e}")
            return False
    
    def _update_performance_stats(self, prediction: Dict) -> None:
        """Update performance statistics based on validated prediction."""
        try:
            regime = prediction['predicted_regime']
            confidence = prediction['confidence']
            outcome = prediction['actual_outcome']
            pnl = prediction['profit_pnl']
            
            # Update regime accuracy
            if regime not in self.performance_stats['regime_accuracy']:
                self.performance_stats['regime_accuracy'][regime] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0.0,
                    'avg_pnl': 0.0,
                    'pnl_sum': 0.0
                }
            
            regime_stats = self.performance_stats['regime_accuracy'][regime]
            regime_stats['total'] += 1
            regime_stats['pnl_sum'] += pnl
            regime_stats['avg_pnl'] = regime_stats['pnl_sum'] / regime_stats['total']
            
            # Define "correct" based on profitability and regime type
            is_correct = False
            if regime in ['trend-up', 'trend-down'] and pnl > 0:
                is_correct = True
            elif regime == 'mean-reversion' and abs(pnl) < 0.5:  # Small profit/loss expected
                is_correct = True
            elif regime in ['chop', 'uncertain'] and abs(pnl) < 1.0:  # Avoid big losses
                is_correct = True
            
            if is_correct:
                regime_stats['correct'] += 1
                self.performance_stats['correct_predictions'] += 1
            
            regime_stats['accuracy'] = regime_stats['correct'] / regime_stats['total']
            
            # Update confidence calibration
            conf_bucket = f"{int(confidence * 10) * 10}%"  # 70%, 80%, etc.
            if conf_bucket not in self.performance_stats['confidence_calibration']:
                self.performance_stats['confidence_calibration'][conf_bucket] = {
                    'predictions': 0,
                    'successes': 0,
                    'accuracy': 0.0,
                    'avg_pnl': 0.0
                }
            
            conf_stats = self.performance_stats['confidence_calibration'][conf_bucket]
            conf_stats['predictions'] += 1
            if is_correct:
                conf_stats['successes'] += 1
            conf_stats['accuracy'] = conf_stats['successes'] / conf_stats['predictions']
            
            # Update recent performance (last 20 predictions)
            recent_predictions = list(self.prediction_history)[-20:]
            validated_recent = [p for p in recent_predictions if p['validation_complete']]
            
            if validated_recent:
                recent_correct = sum(1 for p in validated_recent if self._is_prediction_correct(p))
                self.performance_stats['recent_performance'] = recent_correct / len(validated_recent)
            
        except Exception as e:
            logger.log_error(f"Error updating performance stats: {e}")
    
    def _is_prediction_correct(self, prediction: Dict) -> bool:
        """Determine if a prediction was correct based on outcome."""
        regime = prediction['predicted_regime']
        pnl = prediction.get('profit_pnl', 0)
        
        if regime in ['trend-up', 'trend-down']:
            return pnl > 0
        elif regime == 'mean-reversion':
            return abs(pnl) < 0.5
        elif regime in ['chop', 'uncertain']:
            return abs(pnl) < 1.0
        
        return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        try:
            total = self.performance_stats['total_predictions']
            correct = self.performance_stats['correct_predictions']
            overall_accuracy = correct / total if total > 0 else 0.0
            
            # Recent performance insights
            recent_predictions = list(self.prediction_history)[-10:]
            recent_regimes = [p['predicted_regime'] for p in recent_predictions]
            regime_distribution = {}
            for regime in recent_regimes:
                regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
            
            report = {
                'overall_accuracy': overall_accuracy,
                'recent_performance': self.performance_stats['recent_performance'],
                'total_predictions': total,
                'correct_predictions': correct,
                'regime_accuracy': self.performance_stats['regime_accuracy'],
                'confidence_calibration': self.performance_stats['confidence_calibration'],
                'recent_regime_distribution': regime_distribution,
                'recommendations': self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.log_error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving AI performance."""
        recommendations = []
        
        try:
            stats = self.performance_stats
            
            # Overall accuracy
            total = stats['total_predictions']
            if total > 10:
                accuracy = stats['correct_predictions'] / total
                if accuracy < 0.6:
                    recommendations.append("IMPROVE: Overall accuracy below 60% - consider more conservative thresholds")
                elif accuracy > 0.8:
                    recommendations.append("EXCELLENT: High accuracy - can afford to be more aggressive")
            
            # Regime-specific insights
            for regime, regime_stats in stats['regime_accuracy'].items():
                if regime_stats['total'] >= 3:
                    if regime_stats['accuracy'] < 0.5:
                        recommendations.append(f"WEAK: {regime} predictions underperforming - review criteria")
                    elif regime_stats['avg_pnl'] < -0.5:
                        recommendations.append(f"COSTLY: {regime} predictions losing money - be more selective")
            
            # Confidence calibration
            for conf_level, conf_stats in stats['confidence_calibration'].items():
                if conf_stats['predictions'] >= 3:
                    expected_accuracy = float(conf_level.replace('%', '')) / 100
                    actual_accuracy = conf_stats['accuracy']
                    
                    if actual_accuracy < expected_accuracy - 0.1:
                        recommendations.append(f"OVERCONFIDENT: {conf_level} confidence showing {actual_accuracy:.1%} accuracy")
                    elif actual_accuracy > expected_accuracy + 0.1:
                        recommendations.append(f"UNDERCONFIDENT: {conf_level} confidence could be higher")
            
            # Recent performance
            if stats['recent_performance'] < 0.4:
                recommendations.append("URGENT: Recent performance declining - review recent market conditions")
            elif stats['recent_performance'] > 0.7:
                recommendations.append("STRONG: Recent performance excellent - maintain current approach")
            
            if not recommendations:
                recommendations.append("MONITORING: Continue tracking for pattern identification")
            
            return recommendations
            
        except Exception as e:
            logger.log_error(f"Error generating recommendations: {e}")
            return ["ERROR: Could not generate recommendations"]
    
    def get_prompt_enhancement(self) -> str:
        """Get AI prompt enhancement based on performance data."""
        try:
            report = self.get_performance_report()
            
            if report.get('total_predictions', 0) < 5:
                return ""  # Not enough data yet
            
            # Build performance context for prompt
            enhancement = f"""
PERFORMANCE FEEDBACK:
- Overall accuracy: {report['overall_accuracy']:.1%}
- Recent performance: {report['recent_performance']:.1%}
"""
            
            # Add regime-specific feedback
            best_regime = None
            worst_regime = None
            best_accuracy = 0
            worst_accuracy = 1
            
            for regime, stats in report.get('regime_accuracy', {}).items():
                if stats['total'] >= 3:
                    if stats['accuracy'] > best_accuracy:
                        best_accuracy = stats['accuracy']
                        best_regime = regime
                    if stats['accuracy'] < worst_accuracy:
                        worst_accuracy = stats['accuracy']
                        worst_regime = regime
            
            if best_regime:
                enhancement += f"- STRENGTH: {best_regime} predictions performing well ({best_accuracy:.1%})\n"
            if worst_regime and worst_accuracy < 0.5:
                enhancement += f"- WEAKNESS: {worst_regime} predictions underperforming ({worst_accuracy:.1%}) - be more selective\n"
            
            # Add recommendations
            recommendations = report.get('recommendations', [])[:2]  # Top 2
            for rec in recommendations:
                enhancement += f"- ADVICE: {rec}\n"
            
            return enhancement
            
        except Exception as e:
            logger.log_error(f"Error generating prompt enhancement: {e}")
            return ""


# Global AI performance tracker
ai_performance_tracker = AIPerformanceTracker()
