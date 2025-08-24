"""
Accountant module for generating tax reports from Binance position history.
Fetches trade/position history and generates a tax report based on Israeli tax rules.
"""

import pandas as pd
from typing import List, Dict, Optional
from data.binance_provider import BinanceProvider
from config import config
from logger import logger

class Accountant:
    def __init__(self):
        self.provider = BinanceProvider()

    async def fetch_trade_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch trade history from Binance for the account.
        If symbol is None, fetches for all symbols.
        """
        try:
            trades = await self.provider.get_account_trades(symbol)
            df = pd.DataFrame(trades)
            logger.log_info(f"Fetched {len(df)} trades{' for ' + symbol if symbol else ''}.")
            return df
        except Exception as e:
            logger.log_error(f"Error fetching trade history: {str(e)}")
            return pd.DataFrame()

    def calculate_tax(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate realized gains/losses and tax liability based on Israeli tax rules (FIFO, 25% CGT).
        Returns a DataFrame with tax calculations per trade.
        """
        # Placeholder: implement FIFO and 25% capital gains tax logic
        # This is a simplified example, real implementation should handle all edge cases
        if trades_df.empty:
            return pd.DataFrame()
        trades_df = trades_df.sort_values('time')
        # Example columns: ['symbol', 'side', 'price', 'qty', 'commission', 'time']
        # Implement FIFO matching and gain/loss calculation here
        # ...
        trades_df['gain'] = 0  # Placeholder
        trades_df['tax'] = trades_df['gain'] * 0.25  # 25% tax
        return trades_df

    def generate_report(self, tax_df: pd.DataFrame, output_path: str = 'tax_report.csv'):
        """
        Generate a CSV tax report from the tax DataFrame with more precision and a total row.
        """
        if tax_df.empty:
            logger.log_warning("No tax data to report.")
            return
        # Set precision for gain/tax columns
        for col in ['gain', 'tax']:
            if col in tax_df:
                tax_df[col] = tax_df[col].astype(float).round(16)
        # Add total row
        total_row = {col: '' for col in tax_df.columns}
        if 'gain' in tax_df and 'tax' in tax_df:
            total_row['gain'] = tax_df['gain'].sum().round(16)
            total_row['tax'] = tax_df['tax'].sum().round(16)
            total_row['symbol'] = 'TOTAL' if 'symbol' in tax_df else ''
        import pandas as pd
        tax_df = pd.concat([tax_df, pd.DataFrame([total_row])], ignore_index=True)
        tax_df.to_csv(output_path, index=False)
        logger.log_info(f"Tax report generated: {output_path}")

# Example usage (to be run in an async context):
# accountant = Accountant()
# trades = await accountant.fetch_trade_history()
# tax_df = accountant.calculate_tax(trades)
# accountant.generate_report(tax_df)
