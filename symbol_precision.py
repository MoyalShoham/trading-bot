# Symbol precision info for Binance USDT Perpetual Futures (example values, update as needed)
SYMBOL_PRECISION = {
    # Your trading symbols with corrected precision
    'SOLUSDT':    {'quantity': 0, 'price': 4},    # SOL - whole numbers
    'XRPUSDT':    {'quantity': 1, 'price': 4},    # XRP - 1 decimal place
    'LINKUSDT':   {'quantity': 2, 'price': 3},    # LINK - 2 decimal places
    'TRXUSDT':    {'quantity': 0, 'price': 5},    # TRX - whole numbers
    'FISUSDT':    {'quantity': 0, 'price': 6},    # FIS - whole numbers, 6 decimal price
    'DOGEUSDT':   {'quantity': 0, 'price': 5},    # DOGE - whole numbers
    'DOGSUSDT':   {'quantity': 0, 'price': 7},    # DOGS - whole numbers, 7 decimal price
    'NEARUSDT':   {'quantity': 0, 'price': 4},    # NEAR - whole numbers (fixed!)
    
    # Other common pairs
    'ADAUSDT':    {'quantity': 0, 'price': 4},
    'ETHUSDT':    {'quantity': 3, 'price': 2},
    'BTCUSDT':    {'quantity': 3, 'price': 1},
    'BNBUSDT':    {'quantity': 3, 'price': 2},
    'SPELLUSDT':  {'quantity': 0, 'price': 6},
    'XNYUSDT':    {'quantity': 5, 'price': 5},
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
