from dotenv import load_dotenv
load_dotenv()
"""
Script to run the Accountant module and generate a tax report from Binance trade history.
"""

import asyncio
from accountant import Accountant

async def main():
    accountant = Accountant()
    trades = await accountant.fetch_trade_history()
    tax_df = accountant.calculate_tax(trades)
    accountant.generate_report(tax_df)

if __name__ == "__main__":
    asyncio.run(main())
