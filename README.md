# 🚀 Crypto Trading Bot V2

A modular, safe, and extendable real-time trading agent for Binance Futures with AI-powered market regime classification.

## 🌟 Features

- **Real-time Trading**: 1-minute execution cycles with 5-minute confirmation timeframes
- **AI-Powered Analysis**: OpenAI GPT-4 integration for market regime classification
- **Technical Indicators**: Comprehensive suite including EMA, RSI, Bollinger Bands, MACD, VWAP, ATR
- **Risk Management**: Configurable stop-loss, take-profit, leverage limits, and position sizing
- **Telegram Notifications**: Real-time alerts for signals, trades, and regime changes
- **Structured Logging**: JSON/CSV logging for all trading activities
- **Dry Run Mode**: Paper trading mode for testing strategies
- **Modular Architecture**: Clean separation of concerns for easy extension

## 🏗️ Architecture

```
trading-bot/
├── .env                    # API keys and configuration (never commit!)
├── requirements.txt        # Python dependencies
├── main.py                # Main orchestrator
├── config.py              # Configuration and environment variables
├── data_fetcher.py        # Market data from Binance + CoinGecko
├── indicators.py          # Technical analysis indicators
├── advisor.py             # OpenAI LLM market regime analysis
├── strategy.py            # Signal generation and strategy logic
├── trader.py              # Binance Futures execution
├── notifier.py            # Telegram notifications
├── logger.py              # Structured logging
└── tests/                 # Unit and integration tests
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- Binance Futures account with API keys
- OpenAI API key
- Telegram bot token

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd crypto-bot-v2

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp env.example .env
```

### 3. Configuration

Edit `.env` file with your API credentials:

```bash
# Binance Futures API
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_TESTNET=true  # Start with testnet!

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Trading Configuration
MAX_LEVERAGE=3
RISK_PER_TRADE=0.02
STOP_LOSS_PERCENT=0.05
TAKE_PROFIT_PERCENT=0.10

# Start in dry run mode
DRY_RUN=true
```

### 4. Run the Bot

```bash
# Start the bot
python main.py

# Or run in background
nohup python main.py > bot.log 2>&1 &
```

## 📊 Trading Strategy

The bot combines technical indicators with AI-powered market regime analysis:

### Market Regimes
- **Trend-Up**: Strong upward momentum
- **Trend-Down**: Strong downward momentum  
- **Mean-Reversion**: Price oscillating around central value
- **Chop**: Sideways movement
- **Uncertain**: Mixed signals

### Signal Generation
1. **Data Collection**: Fetch OHLCV data from Binance Futures
2. **Indicator Calculation**: Compute technical indicators
3. **AI Analysis**: Query OpenAI for regime classification
4. **Signal Combination**: Merge indicators with AI insights
5. **Risk Assessment**: Apply position sizing and risk limits
6. **Execution**: Place orders with stop-loss and take-profit

## 🔒 Risk Management

- **Position Sizing**: Based on account balance and risk per trade
- **Stop Loss**: Configurable percentage-based stop losses
- **Take Profit**: Automatic take profit orders
- **Leverage Limits**: Maximum leverage restrictions
- **Balance Thresholds**: Stop trading if balance drops below threshold

## 📱 Notifications

The bot sends Telegram notifications for:

- 🚨 **Trading Signals**: New buy/sell signals with confidence levels
- 💰 **Trade Execution**: Order fills and position updates
- 📊 **Regime Changes**: Market regime updates from AI analysis
- ⚠️ **Error Alerts**: System errors and API failures
- 📈 **Daily Summaries**: Performance and PnL summaries

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_indicators.py

# Run with coverage
pytest --cov=. tests/
```

### Test Structure

- **Unit Tests**: Individual module functionality
- **Integration Tests**: Module interaction testing
- **Mock Data**: Simulated API responses for testing

## 🔧 Configuration Options

### Trading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_LEVERAGE` | 3 | Maximum allowed leverage |
| `RISK_PER_TRADE` | 0.02 | Risk per trade (2% of balance) |
| `STOP_LOSS_PERCENT` | 0.05 | Stop loss percentage (5%) |
| `TAKE_PROFIT_PERCENT` | 0.10 | Take profit percentage (10%) |

### Timeframes

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TIMEFRAME_EXECUTION` | 1m | Signal execution timeframe |
| `TIMEFRAME_CONFIRMATION` | 5m | AI analysis confirmation timeframe |

### Symbols

Currently supports:
- `ETHUSDT` (Ethereum)
- `SOLUSDT` (Solana)

## 📈 Performance Monitoring

The bot tracks:
- Total signals generated
- Trade execution success rate
- PnL performance
- API response times
- Error rates

## 🚨 Safety Features

- **Dry Run Mode**: Test strategies without real money
- **API Rate Limiting**: Respects Binance API limits
- **Error Handling**: Graceful degradation on failures
- **Logging**: Complete audit trail of all activities
- **Validation**: Input validation and sanitization

## 🔮 Future Enhancements

- [ ] Additional trading pairs
- [ ] More technical indicators
- [ ] Machine learning models
- [ ] Backtesting framework
- [ ] Web dashboard
- [ ] Mobile app
- [ ] Social trading features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This software is for educational purposes only. Use at your own risk.**

- Cryptocurrency trading involves substantial risk
- Past performance does not guarantee future results
- Always test thoroughly in dry run mode first
- Never invest more than you can afford to lose
- Consider consulting with a financial advisor

## 🆘 Support

- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check the code comments and docstrings

## 🎯 Getting Help

1. Check the configuration in `.env`
2. Review the logs in the `logs/` directory
3. Test individual modules with the test suite
4. Verify API credentials and permissions
5. Check network connectivity and firewall settings

---

**Happy Trading! 🚀📈**
