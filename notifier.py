"""
Notifier module for the crypto trading bot.
Sends structured messages to Telegram for alerts and notifications.
Examples: new signal, regime change, order filled, error alerts.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from config import config
from logger import logger

class TelegramNotifier:
    def __init__(self):
        self.session = None
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.templates = {
            'signal': (
                "<b>Signal</b> <code>{symbol}</code> | <b>{signal}</b> | Confidence: <code>{confidence:.2f}</code> | Risk: <code>{risk_level}</code>\n"
                "Reason: <code>{reason}</code>\n"
                "Indicators: <code>{indicator_summary}</code>\n"
                "Regime: <code>{regime}</code> | Factors: <code>{factors}</code>\n"
                "Leverage: <code>{leverage}x</code>\n"
                "Time: <code>{timestamp}</code>"
            ),
            'trade': (
                "<b>Trade</b> <code>{symbol}</code> | <b>{side}</b> | Qty: <code>{quantity}</code> | Price: <code>{price:.{price_precision}f}</code> | Leverage: <code>{leverage}x</code>\n"
                "Order ID: <code>{order_id}</code> | Status: <code>{status}</code>\n"
                "Balance: <code>${balance:,.2f}</code> | PnL: <code>${pnl:,.2f}</code>\n"
                "Time: <code>{timestamp}</code>"
            ),
            'regime': (
                "<b>Regime Change</b> <code>{symbol}</code> | <b>{regime}</b> | Confidence: <code>{confidence:.2f}</code>\n"
                "Factors: <code>{factors}</code>\n"
                "Note: <code>{note}</code>\n"
                "Time: <code>{timestamp}</code>"
            ),
            'error': (
                "<b>Error</b> <code>{error_type}</code>\n"
                "Message: <code>{message}</code>\n"
                "Details: <code>{details}</code>\n"
                "Time: <code>{timestamp}</code>"
            ),
            'info': (
                "<b>Info</b>\n"
                "Message: <code>{message}</code>\n"
                "Details: <code>{details}</code>\n"
                "Time: <code>{timestamp}</code>"
            ),
        }
    async def close(self):
        """Close the aiohttp session if open."""
        if self.session and not self.session.closed:
            await self.session.close()
    def escape_markdown(self, text: str) -> str:
        """Escape Telegram MarkdownV2 special characters in text."""
        import re
        if not isinstance(text, str):
            text = str(text)
        return re.sub(r'([_\*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)
    async def initialize(self):
        """Ensure aiohttp session is initialized."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def handle_command(self, command: str) -> bool:
        """
        Handle Telegram bot commands: /portfolio, /status, /positions, /closeAllActivePositions
        """
        from trader import trader
        import aiohttp
        command = command.strip().lower()
        if command in ['/portfolio', 'portfolio']:
            # Fetch latest account info for margin, balance
            margin_balance = None
            total_unrealized_pnl = 0.0
            try:
                response = await trader._make_request('GET', '/fapi/v2/account', signed=True)
                if response:
                    margin_balance = float(response.get('totalMarginBalance', 0.0))
                    # Recalculate unrealized PnL using latest mark price
                    for pos in response.get('positions', []):
                        amt = float(pos.get('positionAmt', 0.0))
                        entry = float(pos.get('entryPrice', 0.0))
                        symbol = pos.get('symbol', None)
                        if symbol and amt != 0:
                            # Fetch mark price
                            mark_price = None
                            try:
                                mark_resp = await trader._make_request('GET', '/fapi/v1/premiumIndex', {'symbol': symbol}, signed=False)
                                if mark_resp and 'markPrice' in mark_resp:
                                    mark_price = float(mark_resp['markPrice'])
                            except Exception:
                                pass
                            if mark_price is not None:
                                if amt > 0:
                                    pnl = (mark_price - entry) * amt
                                else:
                                    pnl = (entry - mark_price) * abs(amt)
                                total_unrealized_pnl += pnl
                            else:
                                total_unrealized_pnl += float(pos.get('unRealizedProfit', 0.0))
            except Exception:
                pass
            balance = getattr(trader, 'balance', 0.0)
            msg = (
                f"\U0001F4B0 <b>Portfolio Overview</b>\n\n"
                f"<b>Wallet Balance:</b> <code>${balance:,.2f}</code>\n"
                f"<b>Margin Balance:</b> <code>${margin_balance if margin_balance is not None else 'N/A':,}</code>\n"
                f"<b>Unrealized PnL:</b> <code>${total_unrealized_pnl:,.2f}</code>\n"
                f"\n<i>Use /positions to see open trades.</i>"
            )
            return await self._send_message(msg, parse_mode='HTML')
        if command in ['/status', 'status']:
            # Try to get stats from TradingBot (import main and get instance if possible)
            total_signals = getattr(trader, 'total_signals', 0)
            total_trades = getattr(trader, 'total_trades', 0)
            total_pnl = getattr(trader, 'total_pnl', 0.0)
            start_time = getattr(trader, 'start_time', None)
            if start_time:
                from datetime import datetime
                runtime = datetime.now() - start_time
                runtime_str = str(runtime).split('.')[0]
            else:
                runtime_str = 'N/A'
            msg = (
                f"\U0001F4CA <b>Bot Status</b>\n\n"
                f"<b>Signals:</b> <code>{total_signals}</code>\n"
                f"<b>Trades:</b> <code>{total_trades}</code>\n"
                f"<b>PnL:</b> <code>${total_pnl:,.2f}</code>\n"
                f"<b>Uptime:</b> <code>{runtime_str}</code>\n"
                f"\n<i>Use /portfolio or /positions for more info.</i>"
            )
            return await self._send_message(msg, parse_mode='HTML')
        if command in ['/positions', 'positions']:
            positions = getattr(trader, 'positions', {})
            if not positions:
                msg = "\U0001F4CB <b>Positions</b>\nNo open positions."
            else:
                lines = ["\U0001F4CB <b>Open Positions</b>\n"]
                async def get_mark_price(symbol):
                    try:
                        resp = await trader._make_request('GET', '/fapi/v1/premiumIndex', {'symbol': symbol}, signed=False)
                        if resp and 'markPrice' in resp:
                            return float(resp['markPrice'])
                    except Exception:
                        pass
                    return None
                for sym, pos in positions.items():
                    side = pos.get('side', 'N/A')
                    qty = pos.get('quantity', pos.get('size', 0))
                    entry = pos.get('entry_price', 0)
                    lev = pos.get('leverage', getattr(trader, 'max_leverage', 1))
                    # Recalculate unrealized PnL and ROI using mark price
                    mark_price = await get_mark_price(sym)
                    if mark_price is not None:
                        if side == 'long':
                            unrealized = (mark_price - entry) * float(qty)
                            roi = ((mark_price - entry) / entry) * 100 if entry else 0.0
                        elif side == 'short':
                            unrealized = (entry - mark_price) * float(qty)
                            roi = ((entry - mark_price) / entry) * 100 if entry else 0.0
                        else:
                            unrealized = 0.0
                            roi = 0.0
                    else:
                        unrealized = pos.get('unrealized_pnl', 0.0)
                        roi = 0.0
                    lines.append(
                        f"<b>{sym}</b>\n"
                        f"  Side: <b>{side.upper()}</b>\n"
                        f"  Qty: <code>{qty}</code>\n"
                        f"  Entry: <code>${entry}</code>\n"
                        f"  Mark: <code>${mark_price if mark_price is not None else 'N/A'}</code>\n"
                        f"  Lev: <code>{lev}x</code>\n"
                        f"  Unrealized: <code>${unrealized:,.2f}</code>\n"
                        f"  ROI: <code>{roi:.2f}%</code>\n"
                        "----------------------"
                    )
                msg = '\n'.join(lines)
            return await self._send_message(msg, parse_mode='HTML')
        if command in ['/closeallactivepositions', 'closeallactivepositions']:
            positions = getattr(trader, 'positions', {})
            closed = 0
            for sym, pos in list(positions.items()):
                side = pos.get('side', 'N/A')
                qty = pos.get('quantity', pos.get('size', 0))
                if qty > 0:
                    await trader.close_position(sym, side, qty)
                    closed += 1
            msg = f"\U00002705 <b>Closed {closed} active positions.</b>"
            return await self._send_message(msg, parse_mode='HTML')
        msg = "\U00002753 <b>Unknown command.</b>\nAvailable: /portfolio, /status, /positions, /closeallactivepositions"
        return await self._send_message(msg, parse_mode='HTML')

    
    def _get_regime_template(self) -> str:
        """Get regime change template."""
        return """
📊 *MARKET REGIME UPDATE* 📊

*Symbol:* {symbol}
*Regime:* {regime}
*Confidence:* {confidence:.1%}

*Key Factors:*
{factors}

*Risk Note:* {note}

*Time:* {timestamp}
"""
    
    def _get_error_template(self) -> str:
        """Get error alert template."""
        return """
⚠️ *ERROR ALERT* ⚠️

*Type:* {error_type}
*Message:* {message}

*Details:*
{details}

*Time:* {timestamp}
"""
    
    def _get_info_template(self) -> str:
        """Get info message template."""
        return """
ℹ️ *INFO UPDATE* ℹ️

*Message:* {message}

*Details:*
{details}

*Time:* {timestamp}
"""
    
    async def _send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a message to Telegram.
        
        Args:
            text: Message text
            parse_mode: Parse mode (HTML only)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.session:
            logger.log_error("Session not initialized")
            return False
        if not self.bot_token or not self.chat_id:
            logger.log_error("Telegram credentials not configured")
            return False
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('ok'):
                        logger.log_info("Telegram message sent successfully")
                        return True
                    else:
                        logger.log_error(f"Telegram API error: {result}")
                        return False
                else:
                    error_text = await response.text()
                    logger.log_error(f"Telegram HTTP error {response.status}: {error_text}")
                    return False
        except Exception as e:
            logger.log_error(f"Error sending Telegram message: {str(e)}")
            return False
    
    def _format_indicator_summary(self, indicators: Dict[str, Any]) -> str:
        """Format indicator summary for Telegram message."""
        if not indicators:
            return "No indicator data available"
        
        summary_lines = []
        
        # Trend signals
        trend_signals = indicators.get('trend_signals', {})
        if trend_signals:
            summary_lines.append("*Trend:*")
            for signal, value in trend_signals.items():
                summary_lines.append(f"  • {signal}: {value}")
        
        # Momentum signals
        momentum_signals = indicators.get('momentum_signals', {})
        if momentum_signals:
            summary_lines.append("*Momentum:*")
            for signal, value in momentum_signals.items():
                summary_lines.append(f"  • {signal}: {value}")
        
        # Volatility signals
        volatility_signals = indicators.get('volatility_signals', {})
        if volatility_signals:
            summary_lines.append("*Volatility:*")
            for signal, value in volatility_signals.items():
                summary_lines.append(f"  • {signal}: {value}")
        
        return '\n'.join(summary_lines) if summary_lines else "No clear signals"
    
    def _format_factors(self, factors: List[str]) -> str:
        """Format factors list for Telegram message."""
        if not factors:
            return "No factors provided"
        
        return '\n'.join([f"• {factor}" for factor in factors])
    
    async def notify_signal(self, signal: Dict[str, Any], leverage: int = None) -> bool:
        """
        Send trading signal notification.
        
        Args:
            signal: Trading signal data
            
        Returns:
            True if notification sent successfully
        """
        from config import config
        if not getattr(config, 'TELEGRAM_NOTIFY_SIGNALS', False):
            return False
        try:
            if not signal:
                logger.log_warning("No signal data provided for notification")
                return False
            # Extract data
            symbol = signal.get('symbol', 'UNKNOWN')
            signal_type = signal.get('signal', 'no-trade')
            confidence = signal.get('confidence', 0.0)
            risk_level = signal.get('risk_level', 'unknown')
            reason = signal.get('reason', 'No reason provided')
            # Get indicator summary
            indicator_analysis = signal.get('indicator_analysis', {})
            indicator_summary = self._format_indicator_summary(indicator_analysis)
            # Get advisor data
            advisor_regime = signal.get('advisor_regime', {})
            regime = advisor_regime.get('regime', 'unknown')
            factors = self._format_factors(advisor_regime.get('factors', []))
            timestamp = signal.get('timestamp', datetime.now().isoformat())
            # Leverage
            if leverage is None:
                from trader import trader
                trader_positions = getattr(trader, 'positions', {})
                lev = None
                if symbol in trader_positions:
                    lev = trader_positions[symbol].get('leverage', None)
                leverage = lev if lev is not None else getattr(trader, 'max_leverage', 1)
            # Format message
            message = self.templates['signal'].format(
                symbol=symbol,
                signal=signal_type.upper(),
                confidence=confidence,
                risk_level=risk_level.upper(),
                reason=reason,
                indicator_summary=indicator_summary,
                regime=regime.upper(),
                factors=factors,
                leverage=leverage,
                timestamp=timestamp
            )
            # Send message
            return await self._send_message(message, parse_mode='HTML')
        except Exception as e:
            logger.log_error(f"Error sending signal notification: {str(e)}")
            return False
    
    async def notify_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Send trade execution notification.
        
        Args:
            trade_data: Trade execution data
            
        Returns:
            True if notification sent successfully
        """
        if not trade_data:
            logger.log_warning("No trade data provided for notification")
            return False
        # Extract data
        symbol = trade_data.get('symbol', 'UNKNOWN')
        side = trade_data.get('side', 'UNKNOWN')
        quantity = trade_data.get('quantity', 0)
        price = trade_data.get('price', 0)
        order_id = trade_data.get('order_id', 'UNKNOWN')
        status = trade_data.get('status', 'UNKNOWN')
        balance = trade_data.get('balance', 0)
        pnl = trade_data.get('pnl', 0)
        timestamp = trade_data.get('timestamp', datetime.now().isoformat())
        # Leverage
        from trader import trader
        trader_positions = getattr(trader, 'positions', {})
        lev = None
        if symbol in trader_positions:
            lev = trader_positions[symbol].get('leverage', None)
        leverage = lev if lev is not None else getattr(trader, 'max_leverage', 1)
        # Price precision
        from symbol_precision import SYMBOL_PRECISION
        price_precision = SYMBOL_PRECISION.get(symbol, 2)
        # Format message
        message = self.templates['trade'].format(
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            price=price,
            price_precision=price_precision,
            order_id=order_id,
            status=status,
            balance=balance,
            pnl=pnl,
            leverage=leverage,
            timestamp=timestamp
        )
        # Send message
        return await self._send_message(message, parse_mode='HTML')
    
    async def notify_regime_change(self, regime_data: Dict[str, Any]) -> bool:
        """
        Send market regime change notification.
        
        Args:
            regime_data: Regime analysis data
            
        Returns:
            True if notification sent successfully
        """
        try:
            if not regime_data:
                logger.log_warning("No regime data provided for notification")
                return False
            
            # Extract data
            symbol = regime_data.get('symbol', 'UNKNOWN')
            regime = regime_data.get('regime', 'unknown')
            confidence = regime_data.get('confidence', 0.0)
            factors = self._format_factors(regime_data.get('factors', []))
            note = regime_data.get('note', 'No additional notes')
            timestamp = regime_data.get('timestamp', datetime.now().isoformat())
            
            # Format message
            message = self.templates['regime'].format(
                symbol=symbol,
                regime=regime.upper(),
                confidence=confidence,
                factors=factors,
                note=note,
                timestamp=timestamp
            )
            
            # Send message
            return await self._send_message(message)
            
        except Exception as e:
            logger.log_error(f"Error sending regime notification: {str(e)}")
            return False
    
    async def notify_error(self, error_type: str, message: str, details: Optional[str] = None) -> bool:
        """
        Send error alert notification.
        
        Args:
            error_type: Type of error
            message: Error message
            details: Additional error details
            
        Returns:
            True if notification sent successfully
        """
        try:
            if not details:
                details = "No additional details available"
            
            timestamp = datetime.now().isoformat()
            
            # Format message
            message_text = self.templates['error'].format(
                error_type=error_type.upper(),
                message=message,
                details=details,
                timestamp=timestamp
            )
            
            # Send message
            return await self._send_message(message_text)
            
        except Exception as e:
            logger.log_error(f"Error sending error notification: {str(e)}")
            return False
    
    async def notify_info(self, message: str, details: Optional[str] = None) -> bool:
        """
        Send info update notification.
        
        Args:
            message: Info message
            details: Additional details
            
        Returns:
            True if notification sent successfully
        """
        try:
            if not details:
                details = "No additional details available"
            
            timestamp = datetime.now().isoformat()
            
            # Format message
            message_text = self.templates['info'].format(
                message=message,
                details=details,
                timestamp=timestamp
            )
            
            # Send message
            return await self._send_message(message_text)
            
        except Exception as e:
            logger.log_error(f"Error sending info notification: {str(e)}")
            return False
    
    async def send_test_message(self) -> bool:
        """Send a test message to verify Telegram integration."""
        try:
            test_message = """
🧪 *BOT TEST MESSAGE* 🧪

This is a test message to verify that the Telegram bot integration is working correctly.

*Bot Status:* ✅ Active
*Time:* {timestamp}

If you receive this message, the notification system is working properly.
""".format(timestamp=datetime.now().isoformat())
            
            success = await self._send_message(test_message)
            if success:
                logger.log_info("Test message sent successfully")
            return success
            
        except Exception as e:
            logger.log_error(f"Error sending test message: {str(e)}")
            return False
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Send daily trading summary.
        
        Args:
            summary_data: Daily summary data
            
        Returns:
            True if notification sent successfully
        """
        try:
            if not summary_data:
                logger.log_warning("No summary data provided")
                return False
            
            # Create summary message
            summary_message = f"""
📈 *DAILY TRADING SUMMARY* 📈

*Date:* {summary_data.get('date', 'Unknown')}
*Total Trades:* {summary_data.get('total_trades', 0)}
*Winning Trades:* {summary_data.get('winning_trades', 0)}
*Losing Trades:* {summary_data.get('losing_trades', 0)}
*Win Rate:* {summary_data.get('win_rate', 0):.1%}

*Total PnL:* ${summary_data.get('total_pnl', 0):,.2f}
*Starting Balance:* ${summary_data.get('starting_balance', 0):,.2f}
*Ending Balance:* ${summary_data.get('ending_balance', 0):,.2f}

*Top Performing Symbol:* {summary_data.get('top_symbol', 'None')}
*Regime Changes:* {summary_data.get('regime_changes', 0)}

*Time:* {datetime.now().isoformat()}
"""
            
            return await self._send_message(summary_message)
            
        except Exception as e:
            logger.log_error(f"Error sending daily summary: {str(e)}")
            return False

# Global notifier instance
notifier = TelegramNotifier()
