"""
Strategy module for the crypto trading bot.
Combines indicator signals with advisor regime to decide high-level signals.
Never executes trades directly here.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from config import config
from logger import logger

class TradingStrategy:
    """Trading strategy that combines technical indicators with AI advisor."""
    
    def __init__(self):
        """Initialize the trading strategy."""
        self.signal_thresholds = {
            'rsi_oversold': 35,  # less strict
            'rsi_overbought': 65,  # less strict
            'bb_oversold': 0.25,  # less strict
            'bb_overbought': 0.75,  # less strict
            'macd_threshold': 0.0,
            'volume_threshold': 1.2  # less strict
        }
    
    def analyze_indicators(self, indicators: Dict[str, any]) -> Dict[str, Any]:
        """
        Analyze technical indicators for trading signals.
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary with indicator analysis and signals
        """
        if not indicators:
            return {}
        
        analysis = {
            'trend_signals': {},
            'momentum_signals': {},
            'volatility_signals': {},
            'volume_signals': {},
            'overall_score': 0.0
        }
        
        try:
            # Get latest values
            latest_values = {}
            for name, series in indicators.items():
                if hasattr(series, 'iloc') and not series.empty:
                    latest_values[name] = float(series.iloc[-1])
            
            # Trend Analysis
            if 'ema_9' in latest_values and 'ema_21' in latest_values:
                ema_9 = latest_values['ema_9']
                ema_21 = latest_values['ema_21']
                
                if ema_9 > ema_21:
                    analysis['trend_signals']['ema_trend'] = 'bullish'
                    analysis['overall_score'] += 0.3
                else:
                    analysis['trend_signals']['ema_trend'] = 'bearish'
                    analysis['overall_score'] -= 0.3
            
            if 'sma_20' in latest_values and 'sma_50' in latest_values:
                sma_20 = latest_values['sma_20']
                sma_50 = latest_values['sma_50']
                
                if sma_20 > sma_50:
                    analysis['trend_signals']['sma_trend'] = 'bullish'
                    analysis['overall_score'] += 0.2
                else:
                    analysis['trend_signals']['sma_trend'] = 'bearish'
                    analysis['overall_score'] -= 0.2
            
            # Momentum Analysis
            if 'rsi' in latest_values:
                rsi = latest_values['rsi']
                if rsi < self.signal_thresholds['rsi_oversold']:
                    analysis['momentum_signals']['rsi'] = 'oversold'
                    analysis['overall_score'] += 0.4
                elif rsi > self.signal_thresholds['rsi_overbought']:
                    analysis['momentum_signals']['rsi'] = 'overbought'
                    analysis['overall_score'] -= 0.4
                else:
                    analysis['momentum_signals']['rsi'] = 'neutral'
            
            if 'macd_line' in latest_values and 'macd_signal' in latest_values:
                macd_line = latest_values['macd_line']
                macd_signal = latest_values['macd_signal']
                
                if macd_line > macd_signal and macd_line > self.signal_thresholds['macd_threshold']:
                    analysis['momentum_signals']['macd'] = 'bullish'
                    analysis['overall_score'] += 0.3
                elif macd_line < macd_signal and macd_line < -self.signal_thresholds['macd_threshold']:
                    analysis['momentum_signals']['macd'] = 'bearish'
                    analysis['overall_score'] -= 0.3
                else:
                    analysis['momentum_signals']['macd'] = 'neutral'
            
            # Volatility Analysis
            if 'bb_percent' in latest_values:
                bb_percent = latest_values['bb_percent']
                if bb_percent < self.signal_thresholds['bb_oversold']:
                    analysis['volatility_signals']['bb_position'] = 'oversold'
                    analysis['overall_score'] += 0.2
                elif bb_percent > self.signal_thresholds['bb_overbought']:
                    analysis['volatility_signals']['bb_position'] = 'overbought'
                    analysis['overall_score'] -= 0.2
                else:
                    analysis['volatility_signals']['bb_position'] = 'middle'
            
            # Volume Analysis
            if 'volume_change' in latest_values:
                volume_change = latest_values['volume_change']
                if volume_change > self.signal_thresholds['volume_threshold']:
                    analysis['volume_signals']['volume'] = 'high'
                    analysis['overall_score'] += 0.1
                elif volume_change < -0.5:
                    analysis['volume_signals']['volume'] = 'low'
                    analysis['overall_score'] -= 0.1
                else:
                    analysis['volume_signals']['volume'] = 'normal'
            
            # Normalize overall score to [-1, 1] range
            analysis['overall_score'] = max(-1.0, min(1.0, analysis['overall_score']))
            
        except Exception as e:
            logger.log_error(f"Error analyzing indicators: {str(e)}")
            analysis['overall_score'] = 0.0
        
        return analysis
    
    def combine_with_advisor(
        self, 
        indicator_analysis: Dict[str, Any], 
        advisor_regime: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine technical indicator analysis with AI advisor regime.
        
        Args:
            indicator_analysis: Technical indicator analysis
            advisor_regime: AI advisor regime analysis
            
        Returns:
            Combined analysis and trading signal
        """
        # Initialize with safe defaults
        combined_analysis = {
            'indicator_score': 0.0,
            'advisor_regime': 'uncertain',
            'advisor_confidence': 0.0,
            'signal': 'no-trade',
            'confidence': 0.0,
            'reason': 'No analysis available',
            'risk_level': 'high'
        }
        
        if not indicator_analysis or not advisor_regime:
            logger.log_warning("Missing data for combined analysis")
            return combined_analysis
        
        try:
            # Safely extract values with defaults
            indicator_score = indicator_analysis.get('overall_score', 0.0)
            regime = advisor_regime.get('regime', 'uncertain')
            advisor_confidence = advisor_regime.get('confidence', 0.0)
            
            # Update combined analysis
            combined_analysis.update({
                'indicator_score': indicator_score,
                'advisor_regime': regime,
                'advisor_confidence': advisor_confidence
            })
            
            # Calculate combined confidence with weighting (favor AI advisor more)
            base_confidence = (indicator_score * 0.2 + advisor_confidence * 0.8)
            
            # Boost confidence for high AI advisor scores
            if advisor_confidence > 0.8:
                base_confidence += 0.1  # 10% boost for exceptional AI confidence
            elif advisor_confidence > 0.7:
                base_confidence += 0.05  # 5% boost for strong AI confidence
            
            combined_analysis['confidence'] = min(0.95, base_confidence)  # Cap at 95%
            
            # Determine trading signal based on regime and indicators (optimized for profit)
            if regime == 'trend-up' and indicator_score > 0.05:
                if advisor_confidence > 0.6:  # Higher threshold for quality
                    combined_analysis['signal'] = 'long_bias'
                    combined_analysis['reason'] = 'Bullish trend with sufficient confidence'
                    combined_analysis['risk_level'] = 'low' if advisor_confidence > 0.8 else 'medium'
                else:
                    combined_analysis['signal'] = 'no-trade'
                    combined_analysis['reason'] = 'Bullish trend but low advisor confidence'
                    combined_analysis['risk_level'] = 'high'
            elif regime == 'trend-down' and indicator_score < -0.05:
                if advisor_confidence > 0.6:  # Higher threshold for quality
                    combined_analysis['signal'] = 'short_bias'
                    combined_analysis['reason'] = 'Bearish trend with sufficient confidence'
                    combined_analysis['risk_level'] = 'low' if advisor_confidence > 0.8 else 'medium'
                else:
                    combined_analysis['signal'] = 'no-trade'
                    combined_analysis['reason'] = 'Bearish trend but low advisor confidence'
                    combined_analysis['risk_level'] = 'high'
            elif regime == 'mean-reversion':
                if indicator_score > 0.15:
                    combined_analysis['signal'] = 'long_bias'
                    combined_analysis['reason'] = 'Mean reversion opportunity - oversold conditions'
                    combined_analysis['risk_level'] = 'medium'
                elif indicator_score < -0.15:
                    combined_analysis['signal'] = 'short_bias'
                    combined_analysis['reason'] = 'Mean reversion opportunity - overbought conditions'
                    combined_analysis['risk_level'] = 'medium'
                else:
                    combined_analysis['signal'] = 'no-trade'
                    combined_analysis['reason'] = 'Mean reversion regime but no clear signal'
                    combined_analysis['risk_level'] = 'high'
            elif regime == 'chop':
                combined_analysis['signal'] = 'no-trade'
                combined_analysis['reason'] = 'Choppy market - avoid trading'
                combined_analysis['risk_level'] = 'high'
            elif regime == 'uncertain':
                combined_analysis['signal'] = 'no-trade'
                combined_analysis['reason'] = 'Uncertain market conditions'
                combined_analysis['risk_level'] = 'high'
            # Additional risk checks (optimized for profit)
            min_confidence = config.MIN_SIGNAL_CONFIDENCE if hasattr(config, 'MIN_SIGNAL_CONFIDENCE') else 0.1
            if combined_analysis['confidence'] < min_confidence:
                combined_analysis['signal'] = 'no-trade'
                combined_analysis['reason'] += ' - Low confidence'
                combined_analysis['risk_level'] = 'high'
            
            # Log the combined analysis
            logger.log_info(f"Combined analysis completed: {combined_analysis['signal']}")
            
        except Exception as e:
            logger.log_error(f"Error combining analysis: {str(e)}")
            combined_analysis.update({
                'signal': 'no-trade',
                'reason': f'Error in analysis: {str(e)}',
                'risk_level': 'high',
                'confidence': 0.0
            })
        
        # Final validation - ensure all required fields exist
        required_fields = ['signal', 'confidence', 'reason', 'risk_level']
        for field in required_fields:
            if field not in combined_analysis:
                logger.log_warning(f"Missing {field} in combined analysis, setting default")
                if field == 'signal':
                    combined_analysis[field] = 'no-trade'
                elif field == 'confidence':
                    combined_analysis[field] = 0.0
                elif field == 'reason':
                    combined_analysis[field] = 'No reason available'
                elif field == 'risk_level':
                    combined_analysis[field] = 'high'
        
        return combined_analysis
    
    def generate_trading_signal(
        self, 
        symbol: str, 
        indicators: Dict[str, any], 
        advisor_regime: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate final trading signal by combining all analysis.
        
        Args:
            symbol: Trading symbol
            indicators: Technical indicators
            advisor_regime: AI advisor regime
            
        Returns:
            Complete trading signal
        """
        try:
            # Analyze indicators
            indicator_analysis = self.analyze_indicators(indicators)
            
            # Combine with advisor
            combined_analysis = self.combine_with_advisor(indicator_analysis, advisor_regime)
            
            # Ensure all required fields are present
            if 'risk_level' not in combined_analysis:
                logger.log_warning(f"Missing risk_level in combined analysis for {symbol}, setting default")
                combined_analysis['risk_level'] = 'medium'
            
            if 'signal' not in combined_analysis:
                logger.log_warning(f"Missing signal in combined analysis for {symbol}, setting default")
                combined_analysis['signal'] = 'no-trade'
            
            if 'confidence' not in combined_analysis:
                logger.log_warning(f"Missing confidence in combined analysis for {symbol}, setting default")
                combined_analysis['confidence'] = 0.0
            
            if 'reason' not in combined_analysis:
                logger.log_warning(f"Missing reason in combined analysis for {symbol}, setting default")
                combined_analysis['reason'] = 'No reason provided'
            
            # Create final signal
            signal = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'signal': combined_analysis['signal'],
                'confidence': combined_analysis['confidence'],
                'reason': combined_analysis['reason'],
                'risk_level': combined_analysis['risk_level'],
                'indicators': indicator_analysis,  # for logger compatibility
                'advisor_regime': advisor_regime,
                'combined_analysis': combined_analysis
            }
            # Log the signal
            logger.log_signal(signal)
            
            return signal
            
        except Exception as e:
            logger.log_error(f"Error generating trading signal for {symbol}: {str(e)}")
            # Return a safe default signal
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'signal': 'no-trade',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}',
                'risk_level': 'high',
                'indicator_analysis': {},
                'advisor_regime': {},
                'combined_analysis': {}
            }
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate a trading signal for quality.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not signal:
            return False
        
        # Check required fields
        required_fields = ['symbol', 'signal', 'confidence', 'reason']
        for field in required_fields:
            if field not in signal:
                return False
        
        # Check signal values
        valid_signals = ['long_bias', 'short_bias', 'no-trade']
        if signal['signal'] not in valid_signals:
            return False
        
        # Check confidence range (allow slightly negative for testing)
        if not (-0.1 <= signal['confidence'] <= 1.0):
            return False
        
        # Check risk level
        valid_risk_levels = ['low', 'medium', 'high']
        if signal.get('risk_level') not in valid_risk_levels:
            return False
        
        return True
    
    def get_signal_strength(self, signal: Dict[str, Any]) -> str:
        """
        Get the strength of a trading signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Signal strength description
        """
        if not signal or 'confidence' not in signal:
            return 'unknown'
        
        confidence = signal['confidence']
        
        if confidence >= 0.8:
            return 'very_strong'
        elif confidence >= 0.6:
            return 'strong'
        elif confidence >= 0.4:
            return 'moderate'
        elif confidence >= 0.2:
            return 'weak'
        else:
            return 'very_weak'

# Global strategy instance
strategy = TradingStrategy()
