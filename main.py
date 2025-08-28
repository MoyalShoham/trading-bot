import aiohttp
"""
Main orchestrator module for the crypto trading bot.
Loops over symbols, fetches data, calculates indicators, queries advisor,
runs strategy, notifies, and optionally trades.
Uses asyncio for real-time execution every 1m.
"""

import asyncio
import signal
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

from config import config
from logger import logger
from data_fetcher import DataFetcher
from indicators import indicators_calculator
from advisor import MarketAdvisor
from strategy import strategy
from trader import trader
from notifier import notifier

class TradingBot:


    def __init__(self):
        self.symbols = config.SYMBOLS
        self.execution_interval = config.EXECUTION_INTERVAL
        self.confirmation_interval = config.CONFIRMATION_INTERVAL
        self.last_execution = {}
        self.last_confirmation = {}
        self.symbol_data = {}
        self.advisor_cache = {}
        self.running = False
        # Performance tracking
        self.start_time = datetime.now()
        self.total_signals = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        # Memory management
        self._cleanup_interval = 300  # Clean up every 5 minutes
        self._last_cleanup = time.time()
        self._max_cache_age = 600  # Max 10 minutes for cache entries

    async def initialize(self):
        try:
            logger.log_info("Initializing Trading Bot...")

            # Validate configuration
            if not config.is_valid():
                issues = config.validate_config()
                logger.log_error("Configuration validation failed", issues)
                await notifier.notify_error(
                    "Configuration Error",
                    "Bot configuration is invalid",
                    str(issues)
                )
                return False

            # Ensure notifier and trader sessions are initialized
            await notifier.initialize()
            await trader.initialize()

            # Initialize data structures first
            for symbol in self.symbols:
                if symbol:  # Skip empty symbols
                    self.last_execution[symbol] = datetime.now() - timedelta(minutes=5)
                    self.last_confirmation[symbol] = datetime.now() - timedelta(minutes=10)
                    self.symbol_data[symbol] = {}
                    self.advisor_cache[symbol] = {}

            # Test Telegram integration
            logger.log_info("Testing Telegram integration...")
            try:
                telegram_ok = await notifier.send_test_message()
                if not telegram_ok:
                    logger.log_warning("Telegram integration test failed")
            except Exception as e:
                logger.log_warning(f"Telegram integration test failed: {str(e)}")

            # Initialize trader
            if not config.DRY_RUN:
                logger.log_info("Initializing trader...")
                try:
                    account_info = await trader.get_account_info()
                    if account_info:
                        logger.log_info("Trader initialized successfully")
                    else:
                        logger.log_warning("Failed to get account info")
                except Exception as e:
                    logger.log_warning(f"Failed to initialize trader: {str(e)}")

            logger.log_info("Trading Bot initialized successfully")
            await notifier.notify_info(
                "Bot Started",
                f"Trading bot initialized with {len([s for s in self.symbols if s])} symbols"
            )
            return True

        except Exception as e:
            logger.log_error(f"Error initializing bot: {str(e)}")
            await notifier.notify_error(
                "Initialization Error",
                "Failed to initialize trading bot",
                str(e)
            )
            return False

    async def fetch_market_data(self) -> Dict[str, Dict]:
        try:
            logger.log_info("Fetching market data using batch processing...")
            valid_symbols = [symbol for symbol in self.symbols if symbol]
            if not valid_symbols:
                logger.log_warning("No valid symbols configured")
                return {}
            fetcher = DataFetcher()
            if config.ENABLE_PARALLEL_PROCESSING:
                data = await fetcher.fetch_multiple_symbols_batch(valid_symbols)
            else:
                data = {}
                for symbol in valid_symbols:
                    try:
                        symbol_data = await fetcher.fetch_symbol_data(symbol)
                        if symbol_data:
                            data[symbol] = symbol_data
                            logger.log_info(f"Data fetched for {symbol}")
                        else:
                            logger.log_warning(f"Failed to fetch data for {symbol}")
                    except Exception as e:
                        logger.log_error(f"Error fetching data for {symbol}: {str(e)}")
            return data
        except Exception as e:
            logger.log_error(f"Error fetching market data: {str(e)}")
            return {}

    async def calculate_indicators(self, market_data: Dict[str, Dict]) -> Dict[str, Dict]:
        try:
            logger.log_info("Calculating technical indicators...")
            valid_symbols = [symbol for symbol in self.symbols if symbol]
            indicators = {}
            for symbol in valid_symbols:
                try:
                    if symbol in market_data:
                        exec_data = market_data[symbol].get('klines', {}).get(config.TIMEFRAME_EXECUTION)
                        if exec_data is not None:
                            symbol_indicators = indicators_calculator.calculate_all_indicators(exec_data)
                            indicators[symbol] = symbol_indicators
                            logger.log_info(f"Indicators calculated for {symbol}")
                        else:
                            logger.log_warning(f"No execution data for {symbol}")
                    else:
                        logger.log_warning(f"No market data for {symbol}")
                except Exception as e:
                    logger.log_error(f"Error calculating indicators for {symbol}: {str(e)}")
            return indicators
        except Exception as e:
            logger.log_error(f"Error calculating indicators: {str(e)}")
            return {}

    async def get_advisor_analysis(self, indicators: Dict[str, Dict]) -> Dict[str, Dict]:
        try:
            logger.log_info("Getting AI advisor analysis...")
            valid_symbols = [symbol for symbol in self.symbols if symbol]
            async with MarketAdvisor() as advisor:
                analysis = {}
                for symbol in valid_symbols:
                    try:
                        if symbol in indicators:
                            now = datetime.now()
                            last_analysis = self.last_confirmation.get(symbol, datetime.min)
                            if (now - last_analysis).total_seconds() >= self.confirmation_interval:
                                regime = await advisor.get_market_regime(symbol, indicators[symbol])
                                if regime:
                                    analysis[symbol] = regime
                                    self.advisor_cache[symbol] = regime
                                    self.last_confirmation[symbol] = now
                                    logger.log_info(f"Advisor analysis updated for {symbol}")
                                else:
                                    if symbol in self.advisor_cache:
                                        analysis[symbol] = self.advisor_cache[symbol]
                                        logger.log_info(f"Using cached advisor analysis for {symbol}")
                            else:
                                if symbol in self.advisor_cache:
                                    analysis[symbol] = self.advisor_cache[symbol]
                                else:
                                    logger.log_warning(f"No advisor analysis available for {symbol}")
                        else:
                            logger.log_warning(f"No indicators available for {symbol}")
                    except Exception as e:
                        logger.log_error(f"Error getting advisor analysis for {symbol}: {str(e)}")
                return analysis
        except Exception as e:
            logger.log_error(f"Error getting advisor analysis: {str(e)}")
            return {}

    async def generate_signals(self, indicators: Dict[str, Dict], advisor_analysis: Dict[str, Dict]) -> Dict[str, Dict]:
        try:
            logger.log_info("Generating trading signals...")
            valid_symbols = [symbol for symbol in self.symbols if symbol]
            signals = {}
            for symbol in valid_symbols:
                try:
                    if symbol in indicators and symbol in advisor_analysis:
                        signal = strategy.generate_trading_signal(
                            symbol,
                            indicators[symbol],
                            advisor_analysis[symbol]
                        )
                        if signal and strategy.validate_signal(signal):
                            signals[symbol] = signal
                            self.total_signals += 1
                            logger.log_info(f"Signal generated for {symbol}: {signal['signal']}")
                            await notifier.notify_signal(signal)
                        else:
                            logger.log_warning(f"Invalid signal generated for {symbol}")
                    else:
                        logger.log_warning(f"Missing data for signal generation: {symbol}")
                except Exception as e:
                    logger.log_error(f"Error generating signal for {symbol}: {str(e)}")
            return signals
        except Exception as e:
            logger.log_error(f"Error generating signals: {str(e)}")
            return {}

    async def execute_signals(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        try:
            if config.DRY_RUN:
                logger.log_info("DRY RUN MODE: No actual trades will be executed")
                return {}
            logger.log_info("Executing trading signals...")
            valid_symbols = [symbol for symbol in self.symbols if symbol]
            executions = {}
            for symbol in valid_symbols:
                try:
                    if symbol in signals:
                        signal = signals[symbol]
                        if signal['signal'] != 'no-trade':
                            execution = await trader.execute_signal(signal)
                            if execution:
                                executions[symbol] = execution
                                self.total_trades += 1
                                trade_data = execution.get('trade_data', {})
                                await notifier.notify_trade(trade_data)
                                logger.log_info(f"Signal executed for {symbol}")
                            else:
                                logger.log_warning(f"Failed to execute signal for {symbol}")
                        else:
                            logger.log_info(f"No trade signal for {symbol}")
                    else:
                        logger.log_warning(f"No signal available for {symbol}")
                except Exception as e:
                    logger.log_error(f"Error executing signal for {symbol}: {str(e)}")
            return executions
        except Exception as e:
            logger.log_error(f"Error executing signals: {str(e)}")
            return {}

    async def process_symbol(self, symbol: str):
        try:
            logger.log_info(f"Processing {symbol}...")
            fetcher = DataFetcher()
            symbol_data = await fetcher.fetch_symbol_data(symbol)
            if not symbol_data:
                logger.log_warning(f"No data fetched for {symbol}")
                return
            exec_data = symbol_data.get('klines', {}).get(config.TIMEFRAME_EXECUTION)
            if exec_data is not None:
                indicators = indicators_calculator.calculate_all_indicators(exec_data)
            else:
                logger.log_warning(f"No execution data for {symbol}")
                return
            async with MarketAdvisor() as advisor:
                regime = await advisor.get_market_regime(symbol, indicators)
                if not regime:
                    logger.log_warning(f"No advisor analysis for {symbol}")
                    return
                signal = strategy.generate_trading_signal(symbol, indicators, regime)
                if signal and strategy.validate_signal(signal):
                    await notifier.notify_signal(signal)
                    await notifier.notify_regime_change(regime)
                    if not config.DRY_RUN and signal['signal'] != 'no-trade':
                        execution = await trader.execute_signal(signal)
                        if execution:
                            trade_data = execution.get('trade_data', {})
                            await notifier.notify_trade(trade_data)
                    logger.log_info(f"Completed processing {symbol}")
                else:
                    logger.log_warning(f"Invalid signal for {symbol}")
        except Exception as e:
            logger.log_error(f"Error processing {symbol}: {str(e)}")

    async def run_execution_cycle(self):
        try:
            logger.log_info("Starting execution cycle...")
            
            # Real-time order cleanup at start of each cycle
            if not config.DRY_RUN and config.ENABLE_REALTIME_CLEANUP:
                try:
                    cleanup_result = await trader.cleanup_orphaned_orders_realtime()
                    if cleanup_result.get('orphaned_orders_cancelled', 0) > 0:
                        logger.log_info(f"CLEANUP: Freed up capital by cleaning {cleanup_result['orphaned_orders_cancelled']} orphaned orders")
                        # Log available balance increase
                        if 'actual_positions_count' in cleanup_result:
                            logger.log_info(f"CLEANUP: Now tracking {cleanup_result['actual_positions_count']} real positions")
                except Exception as cleanup_e:
                    logger.log_warning(f"Non-critical cleanup error: {str(cleanup_e)}")
            
            market_data = await self.fetch_market_data()
            if not market_data:
                logger.log_warning("No market data received")
                return
            indicators = await self.calculate_indicators(market_data)
            if not indicators:
                logger.log_warning("No indicators calculated")
                return
            advisor_analysis = await self.get_advisor_analysis(indicators)
            if not advisor_analysis:
                logger.log_warning("No advisor analysis received")
                return
            signals = await self.generate_signals(indicators, advisor_analysis)
            if not signals:
                logger.log_info("No trading signals generated")
                return
            executions = await self.execute_signals(signals)
            # Always set SLTP for all open positions after every cycle
            try:
                await trader.set_sltp_for_existing_positions()
            except Exception as sltp_e:
                logger.log_error(f"Error setting SLTP for open positions: {str(sltp_e)}")
            logger.log_info(f"Execution cycle completed: {len(signals)} signals, {len(executions)} executions")
        except Exception as e:
            logger.log_error(f"Error in execution cycle: {str(e)}")
            await notifier.notify_error(
                "Execution Error",
                "Error in main execution cycle",
                str(e)
            )

    async def run(self):
        try:
            if not await self.initialize():
                logger.log_error("Failed to initialize bot")
                return
            self.running = True
            logger.log_info("Bot started successfully")
            while self.running:
                try:
                    start_time = time.time()
                    await self.run_execution_cycle()
                    await self._cleanup_memory()
                    execution_time = time.time() - start_time
                    sleep_time = max(0, self.execution_interval - execution_time)
                    logger.log_info(f"Cycle completed in {execution_time:.2f}s, sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    logger.log_info("Bot execution cancelled")
                    break
                except Exception as e:
                    logger.log_error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(10)
        except Exception as e:
            logger.log_error(f"Fatal error in bot: {str(e)}")
            await notifier.notify_error(
                "Fatal Error",
                "Bot encountered a fatal error",
                str(e)
            )
        finally:
            await self.shutdown()

    async def shutdown(self):
        try:
            logger.log_info("Shutting down bot...")
            self.running = False
            await notifier.notify_info(
                "Bot Shutdown",
                "Trading bot is shutting down"
            )
            runtime = datetime.now() - self.start_time
            summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'runtime': str(runtime),
                'total_signals': self.total_signals,
                'total_trades': self.total_trades,
                'total_pnl': self.total_pnl
            }
            await notifier.send_daily_summary(summary)
            await notifier.close()
            logger.log_info("Bot shutdown completed")
        except Exception as e:
            logger.log_error(f"Error during shutdown: {str(e)}")

    async def _cleanup_memory(self):
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            try:
                expired_symbols = []
                for symbol in list(self.advisor_cache.keys()):
                    if symbol in self.last_confirmation:
                        time_since_confirmation = current_time - self.last_confirmation[symbol].timestamp()
                        if time_since_confirmation > self._max_cache_age:
                            expired_symbols.append(symbol)
                for symbol in expired_symbols:
                    del self.advisor_cache[symbol]
                expired_data_symbols = []
                for symbol in list(self.symbol_data.keys()):
                    if symbol in self.last_execution:
                        time_since_execution = current_time - self.last_execution[symbol].timestamp()
                        if time_since_execution > self._max_cache_age:
                            expired_data_symbols.append(symbol)
                for symbol in expired_data_symbols:
                    del self.symbol_data[symbol]
                expired_executions = []
                for symbol in list(self.last_execution.keys()):
                    time_since_execution = current_time - self.last_execution[symbol].timestamp()
                    if time_since_execution > self._max_cache_age:
                        expired_executions.append(symbol)
                for symbol in expired_executions:
                    del self.last_execution[symbol]
                self._last_cleanup = current_time
                if expired_symbols or expired_data_symbols or expired_executions:
                    logger.log_info(f"Memory cleanup completed: {len(expired_symbols)} advisor entries, "
                                 f"{len(expired_data_symbols)} data entries, {len(expired_executions)} execution entries cleared")
            except Exception as e:
                logger.log_error(f"Error during memory cleanup: {str(e)}")

    def signal_handler(self, signum, frame):
        logger.log_info(f"Received signal {signum}, shutting down...")
        self.running = False

# --- Telegram polling loop (must be outside the class) ---

async def telegram_command_polling_loop():
    """Poll Telegram for new messages and handle commands."""
    from notifier import notifier
    from config import config
    import time
    last_update_id = None
    base_url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"
    chat_id = config.TELEGRAM_CHAT_ID
    session = aiohttp.ClientSession()
    try:
        while True:
            url = f"{base_url}/getUpdates?timeout=30"
            if last_update_id:
                url += f"&offset={last_update_id + 1}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for update in data.get('result', []):
                        last_update_id = update['update_id']
                        message = update.get('message')
                        if not message:
                            continue
                        text = message.get('text', '').strip()
                        if text.startswith('/'):
                            await notifier.handle_command(text)
            await asyncio.sleep(2)
    finally:
        await session.close()


async def main():
    """Main entry point."""
    async def set_telegram_commands():
        """Set Telegram bot command menu for user-friendly / commands."""
        import aiohttp
        from config import config
        commands = [
            {"command": "portfolio", "description": "Show portfolio balance"},
            {"command": "status", "description": "Show bot status"},
            {"command": "positions", "description": "Show open positions"},
            {"command": "closeallactivepositions", "description": "Close all open positions"}
        ]
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/setMyCommands"
        data = {"commands": commands}
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=data)

    try:
        bot = TradingBot()
        signal.signal(signal.SIGINT, bot.signal_handler)
        signal.signal(signal.SIGTERM, bot.signal_handler)
        await set_telegram_commands()
        # Run bot and Telegram polling concurrently
        await asyncio.gather(
            bot.run(),
            telegram_command_polling_loop()
        )
    except KeyboardInterrupt:
        logger.log_info("Bot interrupted by user")
    except Exception as e:
        logger.log_error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())
