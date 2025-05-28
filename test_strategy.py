import yfinance as yf
import pandas as pd
import datetime

def simulate_trade_strategy(ticker: str) -> str:
    try:
        print(f"🔧 simulate_trade_strategy() CALLED with ticker: {ticker}")  # Debug marker

        end = datetime.datetime.today()
        start = end - datetime.timedelta(days=365)
        data = yf.download(ticker, start=start, end=end)

        if data.empty:
            return f"❌ No historical data available for {ticker}."

        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['Signal'] = 0
        data.loc[data.index[50]:, 'Signal'] = (
            data['SMA_50'][50:] > data['SMA_200'][50:]
        ).astype(int)
        data['Position'] = data['Signal'].diff()

        buy_signals = data[data['Position'] == 1.0]
        sell_signals = data[data['Position'] == -1.0]
        latest_close = round(data['Close'].iloc[-1], 2)

        return f"""📊 {ticker.upper()} Strategy Backtest (1 Year)
- Buy signals: {len(buy_signals)}
- Sell signals: {len(sell_signals)}
- Latest Close: ${latest_close}
- Strategy: 50-day vs. 200-day SMA crossover
Note: This is a simple simulation and doesn’t include transaction costs or slippage."""

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Exception in simulate_trade_strategy: {error_details}")
        return f"❌ Strategy simulation failed with error:\n{error_details}"

