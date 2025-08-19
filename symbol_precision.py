# Symbol precision info for Binance USDT Perpetual Futures (example values, update as needed)
SYMBOL_PRECISION = {
    'ADAUSDT': 0,
    'XRPUSDT': 1,
    'TRXUSDT': 0,
    'SPELLUSDT': 0,
    # Add more symbols as needed
}

import math

def round_quantity(symbol: str, quantity: float) -> float:
    """Floor quantity to allowed precision for the symbol (Binance style)."""
    precision = SYMBOL_PRECISION.get(symbol, 0)
    factor = 10 ** precision
    return math.floor(quantity * factor) / factor
