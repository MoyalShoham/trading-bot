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
    """Telegram bot notifier for trading alerts."""
    
    def __init__(self):
        """Initialize the Telegram notifier."""
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Message templates
        self.templates = {
            'signal': self._get_signal_template(),
            'trade': self._get_trade_template(),
            'regime': self._get_regime_template(),
            'error': self._get_error_template(),
            'info': self._get_info_template()
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _get_signal_template(self) -> str:
        """Get signal message template."""
        return """
🚨 *TRADING SIGNAL* 🚨

*Symbol:* {symbol}
*Signal:* {signal}
*Confidence:* {confidence:.1%}
*Risk Level:* {risk_level}

*Reason:* {reason}

*Indicators:*
{indicator_summary}

*Advisor Regime:* {regime}
*Factors:* {factors}

*Time:* {timestamp}
"""
    
    def _get_trade_template(self) -> str:
        """Get trade execution template."""
        return """
💰 *TRADE EXECUTED* 💰

*Symbol:* {symbol}
*Side:* {side}
*Quantity:* {quantity}
*Price:* ${price:,.2f}
*Order ID:* {order_id}
*Status:* {status}

*Balance:* ${balance:,.2f}
*PnL:* ${pnl:,.2f}

*Time:* {timestamp}
"""
    
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
    
    async def _send_message(self, text: str, parse_mode: str = 'Markdown') -> bool:
        """
        Send a message to Telegram.
        
        Args:
            text: Message text
            parse_mode: Parse mode (Markdown, HTML, etc.)
            
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
    
    async def notify_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Send trading signal notification.
        
        Args:
            signal: Trading signal data
            
        Returns:
            True if notification sent successfully
        """
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
                timestamp=timestamp
            )
            
            # Send message
            return await self._send_message(message)
            
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
        try:
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
            
            # Format message
            message = self.templates['trade'].format(
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                price=price,
                order_id=order_id,
                status=status,
                balance=balance,
                pnl=pnl,
                timestamp=timestamp
            )
            
            # Send message
            return await self._send_message(message)
            
        except Exception as e:
            logger.log_error(f"Error sending trade notification: {str(e)}")
            return False
    
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
