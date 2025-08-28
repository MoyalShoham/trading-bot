"""
Risk calculation module with advanced position sizing and risk metrics.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from config import config
from utils.logger import TradingLogger

logger = TradingLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for a trading decision."""
    position_size: float
    risk_amount: float
    risk_percentage: float
    stop_loss_distance: float
    take_profit_distance: float
    reward_to_risk_ratio: float
    max_leverage: int
    recommended_leverage: int
    portfolio_heat: float  # Total risk across all positions
    kelly_fraction: Optional[float] = None
    var_95: Optional[float] = None  # Value at Risk
    expected_value: Optional[float] = None


@dataclass
class PositionRisk:
    """Risk assessment for an individual position."""
    symbol: str
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    position_size: float
    current_value: float
    unrealized_pnl: float
    risk_amount: float
    max_loss_percent: float
    time_in_position: timedelta
    
    
class RiskCalculator:
    """Advanced risk calculator with multiple position sizing methods."""
    
    def __init__(self):
        """Initialize the risk calculator."""
        self.max_risk_per_trade = config.RISK_PER_TRADE
        self.max_portfolio_risk = getattr(config, 'MAX_PORTFOLIO_RISK', 0.20)  # 20% max total risk
        self.max_leverage = config.MAX_LEVERAGE
        self.stop_loss_percent = config.STOP_LOSS_PERCENT
        self.take_profit_percent = config.TAKE_PROFIT_PERCENT
        
        # Risk model parameters
        self.confidence_multiplier = 2.0  # Multiplier for high-confidence trades
        self.correlation_adjustment = 0.8  # Reduce position size for correlated assets
        self.volatility_adjustment = True
        self.max_positions = getattr(config, 'MAX_POSITIONS', 5)
        
        # Kelly criterion parameters
        self.use_kelly = getattr(config, 'USE_KELLY_CRITERION', False)
        self.kelly_multiplier = 0.25  # Conservative Kelly fraction
        
        # Historical performance tracking
        self.trade_history: List[Dict] = []
        self.win_rate = 0.5  # Default 50% win rate
        self.avg_win = 0.02  # Default 2% average win
        self.avg_loss = 0.01  # Default 1% average loss

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_balance: float,
        confidence: float = 0.5,
        volatility: Optional[float] = None,
        correlation_factor: float = 1.0
    ) -> RiskMetrics:
        """
        Calculate optimal position size using multiple risk management techniques.
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            account_balance: Current account balance
            confidence: Signal confidence (0-1)
            volatility: Asset volatility (optional)
            correlation_factor: Correlation with existing positions (0-1)
            
        Returns:
            RiskMetrics object with position sizing recommendations
        """
        try:
            # Calculate base risk metrics
            risk_amount = account_balance * self.max_risk_per_trade
            stop_distance = abs(entry_price - stop_loss_price)
            stop_distance_percent = stop_distance / entry_price
            
            # Calculate take profit
            if entry_price > stop_loss_price:  # Long position
                take_profit_price = entry_price * (1 + self.take_profit_percent)
            else:  # Short position
                take_profit_price = entry_price * (1 - self.take_profit_percent)
            
            take_profit_distance = abs(take_profit_price - entry_price)
            reward_to_risk = take_profit_distance / stop_distance
            
            # Base position size (fixed risk method)
            base_position_size = risk_amount / stop_distance
            
            # Apply confidence adjustment
            confidence_adj = min(confidence * self.confidence_multiplier, 2.0)
            adjusted_position_size = base_position_size * confidence_adj
            
            # Apply correlation adjustment
            adjusted_position_size *= correlation_factor
            
            # Apply volatility adjustment
            if volatility and self.volatility_adjustment:
                # Reduce position size for high volatility assets
                volatility_factor = max(0.5, 1.0 - (volatility - 0.02) * 10)
                adjusted_position_size *= volatility_factor
            
            # Calculate recommended leverage
            position_value = adjusted_position_size * entry_price
            required_margin = position_value / self.max_leverage
            recommended_leverage = min(
                self.max_leverage,
                max(1, int(position_value / (account_balance * 0.1)))  # Use max 10% of balance as margin
            )
            
            # Kelly criterion adjustment (if enabled)
            kelly_fraction = None
            if self.use_kelly and self.win_rate > 0:
                kelly_fraction = self._calculate_kelly_fraction()
                if kelly_fraction > 0:
                    kelly_position_size = account_balance * kelly_fraction / stop_distance
                    adjusted_position_size = min(adjusted_position_size, kelly_position_size)
            
            # Ensure position doesn't exceed maximum limits
            max_position_value = account_balance * config.MAX_POSITION_SIZE
            max_position_size = max_position_value / entry_price
            final_position_size = min(adjusted_position_size, max_position_size)
            
            # Calculate final metrics
            final_risk_amount = final_position_size * stop_distance
            final_risk_percentage = final_risk_amount / account_balance
            
            # Calculate portfolio heat (simplified - would need current positions)
            portfolio_heat = final_risk_percentage  # This would be summed across all positions
            
            # Value at Risk calculation (simplified)
            var_95 = final_risk_amount * 1.65  # Assuming normal distribution
            
            # Expected value calculation
            expected_value = None
            if self.win_rate > 0:
                expected_value = (self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss) * final_position_size * entry_price
            
            risk_metrics = RiskMetrics(
                position_size=final_position_size,
                risk_amount=final_risk_amount,
                risk_percentage=final_risk_percentage,
                stop_loss_distance=stop_distance,
                take_profit_distance=take_profit_distance,
                reward_to_risk_ratio=reward_to_risk,
                max_leverage=self.max_leverage,
                recommended_leverage=recommended_leverage,
                portfolio_heat=portfolio_heat,
                kelly_fraction=kelly_fraction,
                var_95=var_95,
                expected_value=expected_value
            )
            
            logger.info(f"Position size calculated for {symbol}: {final_position_size:.6f} (risk: {final_risk_percentage:.2%})")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            # Return conservative fallback
            return RiskMetrics(
                position_size=0.0,
                risk_amount=0.0,
                risk_percentage=0.0,
                stop_loss_distance=stop_distance if 'stop_distance' in locals() else 0.0,
                take_profit_distance=0.0,
                reward_to_risk_ratio=0.0,
                max_leverage=1,
                recommended_leverage=1,
                portfolio_heat=0.0
            )

    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly fraction based on historical performance."""
        if self.win_rate <= 0 or self.avg_loss <= 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win rate, q = loss rate
        b = self.avg_win / self.avg_loss  # odds
        p = self.win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative multiplier
        return max(0.0, kelly_fraction * self.kelly_multiplier)

    def assess_portfolio_risk(self, positions: List[Dict], account_balance: float) -> Dict[str, Any]:
        """
        Assess overall portfolio risk.
        
        Args:
            positions: List of current positions
            account_balance: Current account balance
            
        Returns:
            Portfolio risk assessment
        """
        try:
            total_risk = 0.0
            total_exposure = 0.0
            position_risks = []
            
            for position in positions:
                symbol = position['symbol']
                size = position.get('size', position.get('quantity', 0))
                entry_price = position['entry_price']
                current_price = position.get('current_price', entry_price)
                
                # Calculate position risk
                position_value = size * current_price
                total_exposure += position_value
                
                # Estimate risk based on typical stop loss
                estimated_risk = position_value * self.stop_loss_percent
                total_risk += estimated_risk
                
                position_risk = PositionRisk(
                    symbol=symbol,
                    current_price=current_price,
                    entry_price=entry_price,
                    stop_loss=position.get('stop_loss', 0),
                    take_profit=position.get('take_profit'),
                    position_size=size,
                    current_value=position_value,
                    unrealized_pnl=position.get('unrealized_pnl', 0),
                    risk_amount=estimated_risk,
                    max_loss_percent=estimated_risk / account_balance,
                    time_in_position=timedelta(seconds=position.get('time_in_position', 0))
                )
                position_risks.append(position_risk)
            
            portfolio_risk_percent = total_risk / account_balance
            exposure_percent = total_exposure / account_balance
            
            # Risk assessment
            risk_level = 'low'
            if portfolio_risk_percent > self.max_portfolio_risk:
                risk_level = 'high'
            elif portfolio_risk_percent > self.max_portfolio_risk * 0.7:
                risk_level = 'medium'
            
            return {
                'total_risk_amount': total_risk,
                'total_risk_percent': portfolio_risk_percent,
                'total_exposure': total_exposure,
                'exposure_percent': exposure_percent,
                'position_count': len(positions),
                'max_positions': self.max_positions,
                'risk_level': risk_level,
                'can_add_position': (
                    len(positions) < self.max_positions and 
                    portfolio_risk_percent < self.max_portfolio_risk * 0.8
                ),
                'position_risks': position_risks,
                'recommendations': self._generate_risk_recommendations(portfolio_risk_percent, len(positions))
            }
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return {
                'total_risk_percent': 0.0,
                'risk_level': 'unknown',
                'can_add_position': False,
                'recommendations': ['Unable to assess portfolio risk']
            }

    def _generate_risk_recommendations(self, risk_percent: float, position_count: int) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if risk_percent > self.max_portfolio_risk:
            recommendations.append("Portfolio risk exceeds maximum threshold - consider reducing position sizes")
        
        if risk_percent > self.max_portfolio_risk * 0.8:
            recommendations.append("Approaching maximum portfolio risk - be cautious with new positions")
        
        if position_count >= self.max_positions:
            recommendations.append("Maximum number of positions reached - close existing positions before opening new ones")
        
        if position_count >= self.max_positions * 0.8:
            recommendations.append("Approaching maximum position count - consider consolidating positions")
        
        if risk_percent < self.max_portfolio_risk * 0.3:
            recommendations.append("Portfolio risk is low - consider increasing position sizes for high-confidence signals")
        
        return recommendations

    def update_performance_metrics(self, trade_result: Dict[str, Any]) -> None:
        """
        Update performance metrics with new trade result.
        
        Args:
            trade_result: Dictionary containing trade outcome data
        """
        try:
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': trade_result.get('symbol'),
                'pnl_percent': trade_result.get('pnl_percent', 0),
                'win': trade_result.get('pnl_percent', 0) > 0
            })
            
            # Keep only recent history (last 100 trades)
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
            
            # Recalculate metrics
            self._recalculate_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _recalculate_performance_metrics(self) -> None:
        """Recalculate win rate and average win/loss from trade history."""
        if not self.trade_history:
            return
        
        wins = [trade for trade in self.trade_history if trade['win']]
        losses = [trade for trade in self.trade_history if not trade['win']]
        
        self.win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0.5
        
        if wins:
            self.avg_win = sum(trade['pnl_percent'] for trade in wins) / len(wins)
        
        if losses:
            self.avg_loss = abs(sum(trade['pnl_percent'] for trade in losses)) / len(losses)

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk settings and performance."""
        return {
            'risk_settings': {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_leverage': self.max_leverage,
                'stop_loss_percent': self.stop_loss_percent,
                'take_profit_percent': self.take_profit_percent,
                'max_positions': self.max_positions
            },
            'performance_metrics': {
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'trade_count': len(self.trade_history),
                'kelly_fraction': self._calculate_kelly_fraction() if self.use_kelly else None
            },
            'features': {
                'kelly_criterion': self.use_kelly,
                'volatility_adjustment': self.volatility_adjustment,
                'correlation_adjustment': self.correlation_adjustment != 1.0
            }
        }
