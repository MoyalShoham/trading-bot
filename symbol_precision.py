# Symbol precision info for Binance USDT Perpetual Futures (example values, update as needed)
SYMBOL_PRECISION = {
    'ADAUSDT':   {'quantity': 0, 'price': 4},
    'XRPUSDT':   {'quantity': 1, 'price': 4},
    'TRXUSDT':   {'quantity': 0, 'price': 5},
    'SPELLUSDT': {'quantity': 0, 'price': 6},
    'ETHUSDT':   {'quantity': 3, 'price': 2},
    'SOLUSDT':   {'quantity': 2, 'price': 2},
    'BNBUSDT':   {'quantity': 3, 'price': 2},
    'LINKUSDT':  {'quantity': 2, 'price': 3},
    'NEARUSDT':  {'quantity': 1, 'price': 3},
    'XNYUSDT':   {'quantity': 5, 'price': 5},
}

import math


def round_quantity(symbol: str, quantity: float) -> float:
    """Floor quantity to allowed precision for the symbol (Binance style)."""
    precision = SYMBOL_PRECISION.get(symbol, {}).get('quantity', 0)
    factor = 10 ** precision
    return math.floor(quantity * factor) / factor

def round_price(symbol: str, price: float) -> float:
    """Truncate price to allowed precision for the symbol (Binance style)."""
    precision = SYMBOL_PRECISION.get(symbol, {}).get('price', 2)
    factor = 10 ** precision
    return math.floor(price * factor) / factor
