import asyncio
import pytest
from data.binance_provider import BinanceProvider

@pytest.mark.asyncio
async def test_get_price(monkeypatch):
    provider = BinanceProvider(cache_ttl_prices=1)
    async def mock_fetch(url, params=None):
        return {"symbol": "BTCUSDT", "price": "50000.0"}
    provider._fetch = mock_fetch
    price = await provider.get_price("BTCUSDT")
    assert price == 50000.0
    # Test cache
    price2 = await provider.get_price("BTCUSDT")
    assert price2 == 50000.0

@pytest.mark.asyncio
async def test_get_prices(monkeypatch):
    provider = BinanceProvider(cache_ttl_prices=1)
    async def mock_fetch(url, params=None):
        return [
            {"symbol": "BTCUSDT", "price": "50000.0"},
            {"symbol": "ETHUSDT", "price": "4000.0"},
        ]
    provider._fetch = mock_fetch
    prices = await provider.get_prices(["BTCUSDT", "ETHUSDT"])
    assert prices["BTCUSDT"] == 50000.0
    assert prices["ETHUSDT"] == 4000.0

@pytest.mark.asyncio
async def test_get_klines(monkeypatch):
    provider = BinanceProvider(cache_ttl_klines=1)
    sample_kline = [
        1625097600000, "34000.0", "35000.0", "33000.0", "34500.0", "100.0",
        1625097659999, "3400000.0", 1000, "50.0", "1700000.0", "0"
    ]
    async def mock_fetch(url, params=None):
        return [sample_kline] * 2
    provider._fetch = mock_fetch
    klines = await provider.get_klines("BTCUSDT", interval="1m", limit=2)
    assert len(klines) == 2
    assert klines[0]["open"] == 34000.0
    # Test cache
    klines2 = await provider.get_klines("BTCUSDT", interval="1m", limit=2)
    assert klines2 == klines
