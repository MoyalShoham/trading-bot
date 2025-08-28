"""
Main backtesting engine for strategy evaluation with historical data.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path

from config import config
from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
from backtesting.data_handler import HistoricalDataHandler
from backtesting.portfolio_simulator import PortfolioSimulator
from backtesting.performance_analyzer import PerformanceAnalyzer
from utils.logger import TradingLogger

logger = TradingLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    symbols: List[str]
    timeframes: List[str]
    commission: float = 0.0004  # 0.04% Binance futures maker fee
    slippage: float = 0.0005   # 0.05% slippage
    max_positions: int = 5
    position_sizing: str = 'fixed_risk'  # 'fixed_risk', 'fixed_amount', 'kelly'
    risk_per_trade: float = 0.02
    leverage: int = 1
    margin_requirement: float = 0.1  # 10% margin for leveraged positions


@dataclass
class BacktestResult:
    """Backtesting result summary."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    annual_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: timedelta
    best_month: float
    worst_month: float
    consecutive_wins: int
    consecutive_losses: int
    commission_paid: float
    slippage_cost: float


class BacktestEngine:
    """
    Comprehensive backtesting engine for strategy evaluation.
    Supports multiple timeframes, commission, slippage, and detailed analytics.
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize the backtest engine."""
        self.config = config
        self.data_handler = HistoricalDataHandler()
        self.portfolio = PortfolioSimulator(
            initial_balance=config.initial_balance,
            commission=config.commission,
            slippage=config.slippage,
            leverage=config.leverage
        )
        self.analyzer = PerformanceAnalyzer()
        
        # Backtest state
        self.current_time = config.start_date
        self.historical_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.strategy_results: Dict[str, Any] = {}
        self.trade_log: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        logger.info(f"Backtest engine initialized: {config.start_date} to {config.end_date}")

    async def load_historical_data(self) -> bool:
        """Load historical data for all symbols and timeframes."""
        try:
            logger.info("Loading historical data...")
            
            for symbol in self.config.symbols:
                self.historical_data[symbol] = {}
                
                for timeframe in self.config.timeframes:
                    data = await self.data_handler.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date
                    )
                    
                    if data is not None and not data.empty:
                        self.historical_data[symbol][timeframe] = data
                        logger.info(f"Loaded {len(data)} {timeframe} bars for {symbol}")
                    else:
                        logger.warning(f"No data loaded for {symbol} {timeframe}")
                        return False
            
            logger.info("Historical data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False

    async def run_backtest(self, strategy: BaseStrategy) -> BacktestResult:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Strategy to backtest
            
        Returns:
            BacktestResult with performance metrics
        """
        try:
            logger.info(f"Starting backtest for strategy: {strategy.name}")
            
            # Load data if not already loaded
            if not self.historical_data:
                if not await self.load_historical_data():
                    raise ValueError("Failed to load historical data")
            
            # Reset state
            self.portfolio.reset(self.config.initial_balance)
            self.trade_log.clear()
            self.equity_curve.clear()
            
            # Get the primary timeframe for iteration
            primary_timeframe = self.config.timeframes[0]
            
            # Combine all timestamps and sort
            all_timestamps = set()
            for symbol in self.config.symbols:
                if symbol in self.historical_data and primary_timeframe in self.historical_data[symbol]:
                    all_timestamps.update(self.historical_data[symbol][primary_timeframe].index)
            
            timestamps = sorted(all_timestamps)
            total_bars = len(timestamps)
            
            logger.info(f"Processing {total_bars} time periods...")
            
            # Process each time period
            for i, timestamp in enumerate(timestamps):
                self.current_time = timestamp
                
                # Update portfolio with current prices
                await self._update_portfolio_prices(timestamp)
                
                # Record equity
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': self.portfolio.total_equity,
                    'balance': self.portfolio.balance,
                    'unrealized_pnl': self.portfolio.get_unrealized_pnl()
                })
                
                # Process signals for each symbol
                for symbol in self.config.symbols:
                    try:
                        await self._process_symbol(strategy, symbol, timestamp)
                    except Exception as e:
                        logger.error(f"Error processing {symbol} at {timestamp}: {e}")
                
                # Log progress
                if i % (total_bars // 20) == 0:  # Log every 5%
                    progress = (i / total_bars) * 100
                    logger.info(f"Backtest progress: {progress:.1f}% - Equity: ${self.portfolio.total_equity:,.2f}")
            
            # Generate final results
            result = await self._generate_results(strategy)
            
            logger.info(f"Backtest completed for {strategy.name}: "
                       f"Final balance: ${result.final_balance:,.2f} "
                       f"({result.total_return_pct:+.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    async def _update_portfolio_prices(self, timestamp: datetime) -> None:
        """Update portfolio with current market prices."""
        current_prices = {}
        
        for symbol in self.config.symbols:
            try:
                # Get current price from primary timeframe
                primary_tf = self.config.timeframes[0]
                if (symbol in self.historical_data and 
                    primary_tf in self.historical_data[symbol]):
                    
                    data = self.historical_data[symbol][primary_tf]
                    if timestamp in data.index:
                        current_prices[symbol] = float(data.loc[timestamp, 'close'])
                        
            except Exception as e:
                logger.debug(f"Error getting price for {symbol} at {timestamp}: {e}")
        
        self.portfolio.update_prices(current_prices)

    async def _process_symbol(self, strategy: BaseStrategy, symbol: str, timestamp: datetime) -> None:
        """Process a single symbol at given timestamp."""
        try:
            # Get market data and indicators for this timestamp
            market_data = self._get_market_data(symbol, timestamp)
            if not market_data:
                return
            
            indicators = self._calculate_indicators(symbol, timestamp)
            if not indicators:
                return
            
            # Get current position
            current_position = self.portfolio.get_position(symbol)
            
            # Generate signal
            signal = await strategy.generate_signal(
                symbol=symbol,
                market_data=market_data,
                indicators=indicators,
                current_position=current_position
            )
            
            if not signal or not strategy.validate_signal(signal):
                return
            
            # Process the signal
            await self._execute_signal(signal, timestamp)
            
        except Exception as e:
            logger.debug(f"Error processing signal for {symbol}: {e}")

    def _get_market_data(self, symbol: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get market data for symbol at timestamp."""
        try:
            market_data = {}
            
            for timeframe in self.config.timeframes:
                if (symbol in self.historical_data and 
                    timeframe in self.historical_data[symbol]):
                    
                    data = self.historical_data[symbol][timeframe]
                    
                    # Get the bar at or before the timestamp
                    available_times = data.index[data.index <= timestamp]
                    if len(available_times) > 0:
                        latest_time = available_times[-1]
                        bar_data = data.loc[latest_time]
                        
                        market_data[timeframe] = {
                            'timestamp': latest_time,
                            'open': float(bar_data['open']),
                            'high': float(bar_data['high']),
                            'low': float(bar_data['low']),
                            'close': float(bar_data['close']),
                            'volume': float(bar_data['volume'])
                        }
            
            return market_data if market_data else None
            
        except Exception as e:
            logger.debug(f"Error getting market data for {symbol}: {e}")
            return None

    def _calculate_indicators(self, symbol: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Calculate technical indicators for symbol at timestamp."""
        try:
            # This is a simplified version - in production you'd use the full indicators module
            primary_tf = self.config.timeframes[0]
            
            if (symbol not in self.historical_data or 
                primary_tf not in self.historical_data[symbol]):
                return None
            
            data = self.historical_data[symbol][primary_tf]
            
            # Get data up to current timestamp
            historical_data = data[data.index <= timestamp]
            if len(historical_data) < 20:  # Need minimum data for indicators
                return None
            
            # Calculate basic indicators (simplified)
            close_prices = historical_data['close']
            
            indicators = {
                'sma_20': close_prices.rolling(20).mean().iloc[-1],
                'sma_50': close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else None,
                'rsi': self._calculate_rsi(close_prices, 14),
                'current_price': float(close_prices.iloc[-1]),
                'volume': float(historical_data['volume'].iloc[-1])
            }
            
            # Remove None values
            indicators = {k: v for k, v in indicators.items() if v is not None}
            
            return indicators
            
        except Exception as e:
            logger.debug(f"Error calculating indicators for {symbol}: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI indicator."""
        try:
            if len(prices) < period + 1:
                return None
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
            
        except:
            return None

    async def _execute_signal(self, signal: TradingSignal, timestamp: datetime) -> None:
        """Execute a trading signal."""
        try:
            if signal.signal_type == SignalType.NO_TRADE:
                return
            
            # Determine order details
            if signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                # Opening position
                side = 'buy' if signal.signal_type == SignalType.LONG else 'sell'
                
                # Calculate position size
                position_size = self._calculate_position_size(signal)
                if position_size <= 0:
                    return
                
                # Execute order
                trade_result = self.portfolio.execute_order(
                    symbol=signal.symbol,
                    side=side,
                    quantity=position_size,
                    price=signal.entry_price,
                    timestamp=timestamp,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
                
                if trade_result:
                    self.trade_log.append({
                        'timestamp': timestamp,
                        'symbol': signal.symbol,
                        'side': side,
                        'action': 'open',
                        'quantity': position_size,
                        'price': signal.entry_price,
                        'commission': trade_result.get('commission', 0),
                        'slippage': trade_result.get('slippage', 0),
                        'signal_confidence': signal.confidence,
                        'signal_reason': signal.reason
                    })
            
            elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                # Closing position
                position = self.portfolio.get_position(signal.symbol)
                if position:
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    
                    trade_result = self.portfolio.execute_order(
                        symbol=signal.symbol,
                        side=side,
                        quantity=position['quantity'],
                        price=signal.entry_price,
                        timestamp=timestamp
                    )
                    
                    if trade_result:
                        self.trade_log.append({
                            'timestamp': timestamp,
                            'symbol': signal.symbol,
                            'side': side,
                            'action': 'close',
                            'quantity': position['quantity'],
                            'price': signal.entry_price,
                            'pnl': trade_result.get('pnl', 0),
                            'commission': trade_result.get('commission', 0),
                            'slippage': trade_result.get('slippage', 0)
                        })
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")

    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on risk management."""
        try:
            if self.config.position_sizing == 'fixed_risk':
                if signal.stop_loss is None or signal.entry_price is None:
                    return 0.0
                
                risk_amount = self.portfolio.balance * self.config.risk_per_trade
                price_diff = abs(signal.entry_price - signal.stop_loss)
                
                if price_diff == 0:
                    return 0.0
                
                position_size = risk_amount / price_diff
                
                # Apply leverage
                position_size *= self.config.leverage
                
                # Ensure we don't exceed available balance
                max_position_value = self.portfolio.balance * 0.8  # Use max 80% of balance
                max_position_size = max_position_value / signal.entry_price
                
                return min(position_size, max_position_size)
                
            elif self.config.position_sizing == 'fixed_amount':
                fixed_amount = self.portfolio.balance * 0.1  # 10% of balance per trade
                return fixed_amount / signal.entry_price
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    async def _generate_results(self, strategy: BaseStrategy) -> BacktestResult:
        """Generate comprehensive backtest results."""
        try:
            # Calculate basic metrics
            initial_balance = self.config.initial_balance
            final_balance = self.portfolio.total_equity
            total_return = final_balance - initial_balance
            total_return_pct = total_return / initial_balance
            
            # Calculate duration
            duration = self.config.end_date - self.config.start_date
            duration_days = duration.days
            
            # Annual return
            years = duration_days / 365.25
            annual_return_pct = (final_balance / initial_balance) ** (1 / years) - 1 if years > 0 else 0
            
            # Create equity series for advanced metrics
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate advanced metrics using analyzer
            metrics = self.analyzer.calculate_metrics(equity_df, self.trade_log)
            
            # Trade statistics
            trades_df = pd.DataFrame(self.trade_log)
            winning_trades = len(trades_df[trades_df.get('pnl', 0) > 0]) if not trades_df.empty else 0
            losing_trades = len(trades_df[trades_df.get('pnl', 0) < 0]) if not trades_df.empty else 0
            total_trades = len(trades_df) if not trades_df.empty else 0
            
            # Commission and slippage
            commission_paid = trades_df['commission'].sum() if 'commission' in trades_df.columns else 0
            slippage_cost = trades_df['slippage'].sum() if 'slippage' in trades_df.columns else 0
            
            result = BacktestResult(
                strategy_name=strategy.name,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                duration_days=duration_days,
                initial_balance=initial_balance,
                final_balance=final_balance,
                total_return=total_return,
                total_return_pct=total_return_pct,
                annual_return_pct=annual_return_pct,
                max_drawdown=metrics.get('max_drawdown', 0),
                max_drawdown_pct=metrics.get('max_drawdown_pct', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                sortino_ratio=metrics.get('sortino_ratio', 0),
                calmar_ratio=metrics.get('calmar_ratio', 0),
                win_rate=winning_trades / total_trades if total_trades > 0 else 0,
                profit_factor=metrics.get('profit_factor', 0),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_win=metrics.get('avg_win', 0),
                avg_loss=metrics.get('avg_loss', 0),
                largest_win=metrics.get('largest_win', 0),
                largest_loss=metrics.get('largest_loss', 0),
                avg_trade_duration=timedelta(hours=metrics.get('avg_trade_duration_hours', 0)),
                best_month=metrics.get('best_month', 0),
                worst_month=metrics.get('worst_month', 0),
                consecutive_wins=metrics.get('consecutive_wins', 0),
                consecutive_losses=metrics.get('consecutive_losses', 0),
                commission_paid=commission_paid,
                slippage_cost=slippage_cost
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating results: {e}")
            raise

    def save_results(self, result: BacktestResult, output_dir: str = "backtest_results") -> None:
        """Save backtest results to files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = result.strategy_name.replace(" ", "_")
            
            # Save summary
            summary_file = output_path / f"{strategy_name}_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            
            # Save trade log
            if self.trade_log:
                trades_file = output_path / f"{strategy_name}_trades_{timestamp}.csv"
                trades_df = pd.DataFrame(self.trade_log)
                trades_df.to_csv(trades_file, index=False)
            
            # Save equity curve
            if self.equity_curve:
                equity_file = output_path / f"{strategy_name}_equity_{timestamp}.csv"
                equity_df = pd.DataFrame(self.equity_curve)
                equity_df.to_csv(equity_file, index=False)
            
            logger.info(f"Backtest results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    async def run_multiple_strategies(self, strategies: List[BaseStrategy]) -> Dict[str, BacktestResult]:
        """Run backtest for multiple strategies."""
        results = {}
        
        for strategy in strategies:
            try:
                logger.info(f"Running backtest for {strategy.name}")
                result = await self.run_backtest(strategy)
                results[strategy.name] = result
                
            except Exception as e:
                logger.error(f"Error backtesting {strategy.name}: {e}")
        
        return results

    def compare_strategies(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """Compare multiple strategy results."""
        try:
            comparison_data = []
            
            for strategy_name, result in results.items():
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Total Return %': result.total_return_pct * 100,
                    'Annual Return %': result.annual_return_pct * 100,
                    'Max Drawdown %': result.max_drawdown_pct * 100,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Win Rate %': result.win_rate * 100,
                    'Total Trades': result.total_trades,
                    'Profit Factor': result.profit_factor,
                    'Final Balance': result.final_balance
                })
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return pd.DataFrame()
