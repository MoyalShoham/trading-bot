"""
Order Cleanup Utility - Clean up orphaned orders and sync positions.
Run this to clean up orders for symbols without actual positions.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
from typing import Dict, List, Optional
import json

from config import config
from logger import logger


class OrderCleanup:
    """Utility to clean up orphaned orders."""
    
    def __init__(self):
        """Initialize cleanup utility."""
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_SECRET_KEY
        self.base_url = config.BINANCE_BASE_URL
        
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Optional[Dict]:
        """Make API request to Binance."""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if params is None:
                params = {}
            
            if signed:
                params['timestamp'] = int(time.time() * 1000)
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                signature = hmac.new(
                    self.api_secret.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                params['signature'] = signature
            
            headers = {'X-MBX-APIKEY': self.api_key}
            
            async with aiohttp.ClientSession() as session:
                if method == 'GET':
                    async with session.get(url, params=params, headers=headers) as response:
                        return await response.json()
                elif method == 'DELETE':
                    async with session.delete(url, params=params, headers=headers) as response:
                        return await response.json()
                        
        except Exception as e:
            logger.log_error(f"API request error: {e}")
            return None

    async def get_actual_positions(self) -> Dict[str, Dict]:
        """Get actual positions from Binance."""
        try:
            response = await self._make_request('GET', '/fapi/v2/positionRisk', signed=True)
            
            actual_positions = {}
            if response:
                for pos in response:
                    symbol = pos['symbol']
                    position_amt = float(pos['positionAmt'])
                    
                    if abs(position_amt) > 0:  # Only positions with actual quantity
                        actual_positions[symbol] = {
                            'symbol': symbol,
                            'quantity': abs(position_amt),
                            'side': 'long' if position_amt > 0 else 'short',
                            'entry_price': float(pos['entryPrice']),
                            'unrealized_pnl': float(pos['unRealizedProfit']),
                            'percentage': float(pos['percentage'])
                        }
            
            logger.log_info(f"Found {len(actual_positions)} actual positions: {list(actual_positions.keys())}")
            return actual_positions
            
        except Exception as e:
            logger.log_error(f"Error getting actual positions: {e}")
            return {}

    async def get_all_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        try:
            response = await self._make_request('GET', '/fapi/v1/openOrders', signed=True)
            
            if response:
                logger.log_info(f"Found {len(response)} open orders")
                for order in response:
                    logger.log_info(f"Order: {order['symbol']} - {order['type']} - {order['side']} - ID: {order['orderId']}")
                return response
            
            return []
            
        except Exception as e:
            logger.log_error(f"Error getting open orders: {e}")
            return []

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order."""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            response = await self._make_request('DELETE', '/fapi/v1/order', params, signed=True)
            
            if response and 'orderId' in response:
                logger.log_info(f"✅ Cancelled order {order_id} for {symbol}")
                return True
            else:
                logger.log_warning(f"❌ Failed to cancel order {order_id} for {symbol}: {response}")
                return False
                
        except Exception as e:
            logger.log_error(f"Error cancelling order {order_id} for {symbol}: {e}")
            return False

    async def cleanup_orphaned_orders(self, dry_run: bool = True) -> Dict:
        """Clean up orders for symbols without actual positions."""
        try:
            logger.log_info("🧹 Starting order cleanup process...")
            
            # Get actual positions and open orders
            actual_positions = await self.get_actual_positions()
            open_orders = await self.get_all_open_orders()
            
            orphaned_orders = []
            position_orders = []
            
            # Categorize orders
            for order in open_orders:
                symbol = order['symbol']
                order_id = order['orderId']
                
                if symbol in actual_positions:
                    position_orders.append(order)
                    logger.log_info(f"✅ Keeping order {order_id} for {symbol} (has position)")
                else:
                    orphaned_orders.append(order)
                    logger.log_warning(f"🚫 Found orphaned order {order_id} for {symbol} (no position)")
            
            # Summary
            logger.log_info(f"📊 Order Analysis:")
            logger.log_info(f"   • Total open orders: {len(open_orders)}")
            logger.log_info(f"   • Orders with positions: {len(position_orders)}")
            logger.log_info(f"   • Orphaned orders: {len(orphaned_orders)}")
            
            # Cancel orphaned orders
            cancelled_count = 0
            failed_count = 0
            
            if orphaned_orders:
                if dry_run:
                    logger.log_info("🔍 DRY RUN - Would cancel these orphaned orders:")
                    for order in orphaned_orders:
                        logger.log_info(f"   • {order['symbol']} - {order['type']} - ID: {order['orderId']}")
                else:
                    logger.log_info("🗑️ Cancelling orphaned orders...")
                    for order in orphaned_orders:
                        symbol = order['symbol']
                        order_id = order['orderId']
                        
                        success = await self.cancel_order(symbol, order_id)
                        if success:
                            cancelled_count += 1
                        else:
                            failed_count += 1
                        
                        # Small delay to avoid rate limits
                        await asyncio.sleep(0.1)
            
            # Final summary
            result = {
                'total_orders': len(open_orders),
                'position_orders': len(position_orders),
                'orphaned_orders': len(orphaned_orders),
                'cancelled': cancelled_count,
                'failed': failed_count,
                'actual_positions': actual_positions,
                'dry_run': dry_run
            }
            
            if not dry_run:
                logger.log_info(f"🎯 Cleanup Complete:")
                logger.log_info(f"   • Cancelled: {cancelled_count}")
                logger.log_info(f"   • Failed: {failed_count}")
                logger.log_info(f"   • Remaining orders: {len(position_orders)}")
            
            return result
            
        except Exception as e:
            logger.log_error(f"Error in cleanup process: {e}")
            return {'error': str(e)}

    async def get_cleanup_report(self) -> Dict:
        """Get a detailed report of current orders and positions."""
        try:
            logger.log_info("📋 Generating cleanup report...")
            
            actual_positions = await self.get_actual_positions()
            open_orders = await self.get_all_open_orders()
            
            # Group orders by symbol
            orders_by_symbol = {}
            for order in open_orders:
                symbol = order['symbol']
                if symbol not in orders_by_symbol:
                    orders_by_symbol[symbol] = []
                orders_by_symbol[symbol].append(order)
            
            # Create detailed report
            report = {
                'timestamp': time.time(),
                'summary': {
                    'total_positions': len(actual_positions),
                    'total_orders': len(open_orders),
                    'symbols_with_positions': list(actual_positions.keys()),
                    'symbols_with_orders': list(orders_by_symbol.keys())
                },
                'details': {}
            }
            
            # Analyze each symbol
            all_symbols = set(actual_positions.keys()) | set(orders_by_symbol.keys())
            
            for symbol in all_symbols:
                has_position = symbol in actual_positions
                has_orders = symbol in orders_by_symbol
                
                symbol_info = {
                    'has_position': has_position,
                    'has_orders': has_orders,
                    'status': 'unknown'
                }
                
                if has_position and has_orders:
                    symbol_info['status'] = 'normal'
                    symbol_info['position'] = actual_positions[symbol]
                    symbol_info['orders'] = orders_by_symbol[symbol]
                elif has_position and not has_orders:
                    symbol_info['status'] = 'no_protection'  # Position without SL/TP
                    symbol_info['position'] = actual_positions[symbol]
                    symbol_info['orders'] = []
                elif not has_position and has_orders:
                    symbol_info['status'] = 'orphaned_orders'  # Orders without position
                    symbol_info['orders'] = orders_by_symbol[symbol]
                
                report['details'][symbol] = symbol_info
            
            # Log summary
            logger.log_info("📊 Current Status:")
            logger.log_info(f"   • Positions: {len(actual_positions)}")
            logger.log_info(f"   • Open orders: {len(open_orders)}")
            logger.log_info(f"   • Symbols with both: {len([s for s in all_symbols if s in actual_positions and s in orders_by_symbol])}")
            logger.log_info(f"   • Orphaned order symbols: {len([s for s in all_symbols if s not in actual_positions and s in orders_by_symbol])}")
            
            return report
            
        except Exception as e:
            logger.log_error(f"Error generating report: {e}")
            return {'error': str(e)}


async def main():
    """Main cleanup function."""
    cleanup = OrderCleanup()
    
    # First, get a detailed report
    report = await cleanup.get_cleanup_report()
    
    if 'error' not in report:
        orphaned_symbols = [
            symbol for symbol, info in report['details'].items() 
            if info['status'] == 'orphaned_orders'
        ]
        
        if orphaned_symbols:
            print(f"\n🚫 Found orphaned orders for: {orphaned_symbols}")
            print("\n🔍 Running DRY RUN first...")
            
            # Dry run first
            await cleanup.cleanup_orphaned_orders(dry_run=True)
            
            # Ask for confirmation
            response = input("\n❓ Proceed with actual cleanup? (y/N): ").strip().lower()
            
            if response == 'y':
                print("\n🗑️ Performing actual cleanup...")
                result = await cleanup.cleanup_orphaned_orders(dry_run=False)
                print(f"\n✅ Cleanup complete! Cancelled {result.get('cancelled', 0)} orders.")
            else:
                print("\n❌ Cleanup cancelled by user.")
        else:
            print("\n✅ No orphaned orders found. All orders have corresponding positions.")
    else:
        print(f"\n❌ Error generating report: {report['error']}")


if __name__ == "__main__":
    asyncio.run(main())
