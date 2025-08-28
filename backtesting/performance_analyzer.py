"""
Performance analysis module for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import math

from utils.logger import TradingLogger

logger = TradingLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.
    Calculates various risk and return metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(self, equity_curve: pd.DataFrame, trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: DataFrame with timestamp and equity columns
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            metrics = {}
            
            if equity_curve.empty:
                return self._empty_metrics()
            
            # Basic return metrics
            metrics.update(self._calculate_return_metrics(equity_curve))
            
            # Risk metrics
            metrics.update(self._calculate_risk_metrics(equity_curve))
            
            # Trade-based metrics
            if trades:
                metrics.update(self._calculate_trade_metrics(trades))
            
            # Drawdown analysis
            metrics.update(self._calculate_drawdown_metrics(equity_curve))
            
            # Monthly/yearly analysis
            metrics.update(self._calculate_period_metrics(equity_curve))
            
            # Risk-adjusted returns
            metrics.update(self._calculate_risk_adjusted_metrics(equity_curve))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }

    def _calculate_return_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic return metrics."""
        try:
            initial_equity = equity_curve['equity'].iloc[0]
            final_equity = equity_curve['equity'].iloc[-1]
            
            total_return = (final_equity - initial_equity) / initial_equity
            
            # Calculate daily returns
            equity_curve['daily_return'] = equity_curve['equity'].pct_change().fillna(0)
            
            # Annualize return
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            years = max(days / 365.25, 1/365.25)  # Minimum 1 day
            annual_return = (final_equity / initial_equity) ** (1 / years) - 1
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'initial_equity': initial_equity,
                'final_equity': final_equity,
                'trading_days': days
            }
            
        except Exception as e:
            logger.error(f"Error calculating return metrics: {e}")
            return {'total_return': 0.0, 'annual_return': 0.0}

    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics."""
        try:
            if 'daily_return' not in equity_curve.columns:
                equity_curve['daily_return'] = equity_curve['equity'].pct_change().fillna(0)
            
            daily_returns = equity_curve['daily_return']
            
            # Volatility (annualized)
            volatility = daily_returns.std() * np.sqrt(252)  # 252 trading days
            
            # Downside deviation for Sortino ratio
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # VaR and CVaR
            var_95 = daily_returns.quantile(0.05)
            cvar_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else 0
            
            # Maximum consecutive losing days
            losing_days = (daily_returns < 0).astype(int)
            max_consecutive_losses = self._max_consecutive(losing_days)
            
            return {
                'volatility': volatility,
                'downside_deviation': downside_deviation,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_consecutive_losing_days': max_consecutive_losses,
                'positive_days': (daily_returns > 0).sum(),
                'negative_days': (daily_returns < 0).sum(),
                'flat_days': (daily_returns == 0).sum()
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'volatility': 0.0, 'downside_deviation': 0.0}

    def _calculate_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate drawdown metrics."""
        try:
            equity = equity_curve['equity']
            
            # Calculate running maximum (peak)
            peak = equity.expanding().max()
            
            # Calculate drawdown
            drawdown = equity - peak
            drawdown_pct = drawdown / peak
            
            # Maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_pct = drawdown_pct.min()
            
            # Drawdown duration
            drawdown_periods = self._calculate_drawdown_periods(equity, peak)
            
            max_drawdown_duration = 0
            avg_drawdown_duration = 0
            
            if drawdown_periods:
                max_drawdown_duration = max(drawdown_periods)
                avg_drawdown_duration = np.mean(drawdown_periods)
            
            # Recovery time from max drawdown
            max_dd_start = drawdown_pct.idxmin()
            recovery_time = 0
            
            if max_dd_start in peak.index:
                peak_before_dd = peak.loc[max_dd_start]
                recovery_mask = equity[max_dd_start:] >= peak_before_dd
                
                if recovery_mask.any():
                    recovery_date = equity[max_dd_start:][recovery_mask].index[0]
                    recovery_time = (recovery_date - max_dd_start).days
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'max_drawdown_duration': max_drawdown_duration,
                'avg_drawdown_duration': avg_drawdown_duration,
                'recovery_time': recovery_time,
                'drawdown_periods': len(drawdown_periods)
            }
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0}

    def _calculate_drawdown_periods(self, equity: pd.Series, peak: pd.Series) -> List[int]:
        """Calculate individual drawdown periods."""
        try:
            drawdown_periods = []
            in_drawdown = False
            drawdown_start = None
            
            for i, (eq, pk) in enumerate(zip(equity, peak)):
                if eq < pk and not in_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    drawdown_start = i
                elif eq >= pk and in_drawdown:
                    # End of drawdown
                    in_drawdown = False
                    if drawdown_start is not None:
                        duration = i - drawdown_start
                        drawdown_periods.append(duration)
            
            # Handle case where we end in a drawdown
            if in_drawdown and drawdown_start is not None:
                duration = len(equity) - 1 - drawdown_start
                drawdown_periods.append(duration)
            
            return drawdown_periods
            
        except Exception as e:
            logger.error(f"Error calculating drawdown periods: {e}")
            return []

    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate trade-based metrics."""
        try:
            if not trades:
                return {}
            
            trades_df = pd.DataFrame(trades)
            
            # Filter for trades with PnL data
            pnl_trades = trades_df[trades_df['pnl'].notna()] if 'pnl' in trades_df.columns else trades_df
            
            if pnl_trades.empty:
                return {}
            
            pnls = pnl_trades['pnl']
            winning_trades = pnls[pnls > 0]
            losing_trades = pnls[pnls < 0]
            
            # Basic trade stats
            total_trades = len(pnl_trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            # PnL stats
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
            largest_win = winning_trades.max() if len(winning_trades) > 0 else 0
            largest_loss = losing_trades.min() if len(losing_trades) > 0 else 0
            
            # Profit factor
            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            
            # Consecutive wins/losses
            consecutive_wins = self._max_consecutive((pnls > 0).astype(int))
            consecutive_losses = self._max_consecutive((pnls < 0).astype(int))
            
            # Trade duration analysis
            duration_metrics = {}
            if 'trade_duration' in trades_df.columns:
                durations = pd.to_timedelta(trades_df['trade_duration'])
                duration_metrics = {
                    'avg_trade_duration_hours': durations.mean().total_seconds() / 3600,
                    'min_trade_duration_hours': durations.min().total_seconds() / 3600,
                    'max_trade_duration_hours': durations.max().total_seconds() / 3600
                }
            
            return {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                **duration_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {}

    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive occurrences in a binary series."""
        try:
            if series.empty:
                return 0
            
            groups = (series != series.shift()).cumsum()
            consecutive_counts = series.groupby(groups).sum()
            return consecutive_counts.max() if not consecutive_counts.empty else 0
            
        except Exception as e:
            logger.error(f"Error calculating consecutive: {e}")
            return 0

    def _calculate_period_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate monthly and yearly performance metrics."""
        try:
            equity = equity_curve['equity']
            
            # Resample to monthly returns
            monthly_equity = equity.resample('M').last()
            monthly_returns = monthly_equity.pct_change().dropna()
            
            # Monthly stats
            best_month = monthly_returns.max() if not monthly_returns.empty else 0
            worst_month = monthly_returns.min() if not monthly_returns.empty else 0
            avg_monthly_return = monthly_returns.mean() if not monthly_returns.empty else 0
            positive_months = (monthly_returns > 0).sum() if not monthly_returns.empty else 0
            negative_months = (monthly_returns < 0).sum() if not monthly_returns.empty else 0
            
            # Yearly stats if we have enough data
            yearly_metrics = {}
            if len(equity) > 365:
                yearly_equity = equity.resample('Y').last()
                yearly_returns = yearly_equity.pct_change().dropna()
                
                if not yearly_returns.empty:
                    yearly_metrics = {
                        'best_year': yearly_returns.max(),
                        'worst_year': yearly_returns.min(),
                        'avg_yearly_return': yearly_returns.mean(),
                        'positive_years': (yearly_returns > 0).sum(),
                        'negative_years': (yearly_returns < 0).sum()
                    }
            
            return {
                'best_month': best_month,
                'worst_month': worst_month,
                'avg_monthly_return': avg_monthly_return,
                'positive_months': positive_months,
                'negative_months': negative_months,
                'total_months': len(monthly_returns),
                **yearly_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating period metrics: {e}")
            return {}

    def _calculate_risk_adjusted_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-adjusted return metrics."""
        try:
            if 'daily_return' not in equity_curve.columns:
                equity_curve['daily_return'] = equity_curve['equity'].pct_change().fillna(0)
            
            daily_returns = equity_curve['daily_return']
            
            # Risk-free rate (daily)
            daily_risk_free = self.risk_free_rate / 252
            
            # Excess returns
            excess_returns = daily_returns - daily_risk_free
            
            # Sharpe ratio
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = excess_returns.mean() * np.sqrt(252) / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = daily_returns[daily_returns < daily_risk_free]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            max_drawdown_pct = abs(self._calculate_drawdown_metrics(equity_curve).get('max_drawdown_pct', 0))
            annual_return = self._calculate_return_metrics(equity_curve).get('annual_return', 0)
            calmar_ratio = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
            
            # Information ratio (assuming benchmark return is risk-free rate)
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'excess_return_annual': excess_returns.mean() * 252
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0}

    def generate_performance_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a formatted performance report."""
        try:
            report = []
            report.append("=" * 60)
            report.append("BACKTESTING PERFORMANCE REPORT")
            report.append("=" * 60)
            
            # Returns section
            report.append("\nRETURNS:")
            report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
            report.append(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
            report.append(f"Trading Days: {metrics.get('trading_days', 0)}")
            
            # Risk section
            report.append("\nRISK METRICS:")
            report.append(f"Volatility: {metrics.get('volatility', 0):.2%}")
            report.append(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}")
            report.append(f"VaR (95%): {metrics.get('var_95', 0):.2%}")
            
            # Risk-adjusted returns
            report.append("\nRISK-ADJUSTED RETURNS:")
            report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
            report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
            
            # Trade statistics
            if 'total_trades' in metrics:
                report.append("\nTRADE STATISTICS:")
                report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
                report.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
                report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                report.append(f"Average Win: {metrics.get('avg_win', 0):.4f}")
                report.append(f"Average Loss: {metrics.get('avg_loss', 0):.4f}")
                report.append(f"Largest Win: {metrics.get('largest_win', 0):.4f}")
                report.append(f"Largest Loss: {metrics.get('largest_loss', 0):.4f}")
            
            # Monthly performance
            if 'best_month' in metrics:
                report.append("\nMONTHLY PERFORMANCE:")
                report.append(f"Best Month: {metrics.get('best_month', 0):.2%}")
                report.append(f"Worst Month: {metrics.get('worst_month', 0):.2%}")
                report.append(f"Positive Months: {metrics.get('positive_months', 0)}")
                report.append(f"Negative Months: {metrics.get('negative_months', 0)}")
            
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return "Error generating performance report"
