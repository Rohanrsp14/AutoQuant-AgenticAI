import os
import requests
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema import SystemMessage
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
import os
import chromadb
import plotly.graph_objs as go
import plotly.io as pio
import datetime
import yfinance as yf
import pandas as pd


def get_stock_summary(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        name = info.get("shortName", "N/A")
        price = info.get("currentPrice", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        market_cap = info.get("marketCap", "N/A")
        volume = info.get("volume", "N/A")
        day_high = info.get("dayHigh", "N/A")
        day_low = info.get("dayLow", "N/A")

        return (f"Stock: {name} ({ticker.upper()})\n"
                f"Current Price: ${price}\n"
                f"P/E Ratio: {pe_ratio}\n"
                f"Market Cap: {market_cap}\n"
                f"Volume: {volume}\n"
                f"Day High/Low: {day_high} / {day_low}")
    except Exception as e:
        return f"Failed to retrieve stock data: {e}"


load_dotenv()

import yfinance as yf
import pandas as pd
import datetime


def simulate_trade_strategy(ticker: str) -> str:
    try:
        print(f"🔧 simulate_trade_strategy() CALLED with ticker: {ticker}"
              )  # Debug marker

        end = datetime.datetime.today()
        start = end - datetime.timedelta(days=365)
        data = yf.download(ticker, start=start, end=end)

        if data.empty:
            return f"❌ No historical data available for {ticker}."

        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['Signal'] = 0
        data.loc[data.index[50]:,
                 'Signal'] = (data['SMA_50'][50:]
                              > data['SMA_200'][50:]).astype(int)
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


def compare_trade_strategies(ticker1: str, ticker2: str) -> str:

    def run_strategy(ticker):
        try:
            end = datetime.datetime.today()
            start = end - datetime.timedelta(days=365)
            data = yf.download(ticker, start=start, end=end)

            if data.empty:
                return f"{ticker.upper()}: No data."

            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            data['Signal'] = 0
            data.loc[data.index[50]:,
                     'Signal'] = (data['SMA_50'][50:]
                                  > data['SMA_200'][50:]).astype(int)
            data['Position'] = data['Signal'].diff()

            buys = len(data[data['Position'] == 1.0])
            sells = len(data[data['Position'] == -1.0])
            close = round(data['Close'].iloc[-1], 2)

            return f"{ticker.upper()}: {buys} Buy | {sells} Sell | Latest Close: ${close}"
        except Exception as e:
            return f"{ticker.upper()}: Strategy failed. Error: {str(e)}"

    result1 = run_strategy(ticker1)
    result2 = run_strategy(ticker2)

    return f"""📊 Comparison of {ticker1.upper()} vs {ticker2.upper()} (50/200 SMA)
{result1}
{result2}
Note: Simple SMA crossover strategy. No slippage or transaction costs included."""


import yfinance as yf
import pandas as pd


def smart_swing_signal(ticker: str) -> str:
    try:
        end = datetime.datetime.today()
        start = end - datetime.timedelta(days=365)
        data = yf.download(ticker, start=start, end=end)

        if data.empty or len(data) < 250:
            return f"Not enough data to analyze {ticker.upper()}."

        # Indicators
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()

        delta = data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        # ✅ ensure RSI is a scalar float, not a Pandas object
        rsi_value = float(data['RSI'].iloc[-1])

        # Trade conditions
        trend_up = latest['SMA_50'] > latest['SMA_200']
        rsi_ok = 45 <= rsi_value <= 60
        inside_bar = (latest['High'] < prev['High']) and (latest['Low']
                                                          > prev['Low'])

        entry_price = round(prev['High'], 2)
        stop_loss = round(prev['Low'], 2)
        target_price = round(entry_price + (entry_price - stop_loss), 2)
        rrr = round(
            (target_price - entry_price) /
            (entry_price - stop_loss), 2) if (entry_price -
                                              stop_loss) != 0 else 'N/A'

        # Debug: Confirm function is running
        print(f"✅ SmartSwingSignal executed for {ticker.upper()}")

        if trend_up and rsi_ok and inside_bar:
            return f"""📈 {ticker.upper()} SmartSwingSignal

✅ Trend: Uptrend confirmed (50 > 200 SMA)
✅ RSI: {round(rsi_value, 1)} → healthy pullback zone
✅ Inside Bar Detected: Yes ({data.index[-1].strftime('%Y-%m-%d')})
🔔 Entry Trigger: Break above ${entry_price}
📉 Stop Loss: Below ${stop_loss}
🎯 Target: ${target_price} (based on inside bar range)
📊 Volume Confirmation Required: Yes
🧮 Risk-Reward Ratio: ~{rrr}R
📅 Exit Strategy: Trail 10EMA or RSI > 70
"""
        else:
            return f"""📉 {ticker.upper()} does not meet SmartSwingSignal criteria.

Trend Up: {'✅' if trend_up else '❌'}
RSI: {round(rsi_value, 1)} {'✅' if rsi_ok else '❌'}
Inside Bar: {'✅' if inside_bar else '❌'}

📊 Latest Prices:
- High: {round(latest['High'], 2)}
- Low: {round(latest['Low'], 2)}
- Close: {round(latest['Close'], 2)}

Try again when setup conditions are met."""
    except Exception as e:
        import traceback
        return "SmartSwingSignal failed:\n" + traceback.format_exc()

from fpdf import FPDF

def export_summary_to_pdf(summary: str, filename: str = "SmartSwingSignal_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in summary.strip().split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(filename)
    print(f"✅ PDF generated: {filename}")


# Setup OpenRouter (acts like OpenAI)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Use an LLM from OpenRouter (e.g., Mistral, Claude, LLaMA3)
llm = ChatOpenAI(
    temperature=0.5,
    model=
    "mistralai/mistral-7b-instruct",  # You can change to another OpenRouter-supported model
    openai_api_base=os.environ["OPENAI_API_BASE"])

# Add a real tool - DuckDuckGo Search
search = DuckDuckGoSearchRun()

# Define tools for the agent
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description=
        "Useful for answering questions about current events or public info."),
    Tool(
        name="Stock Summary Tool",
        func=get_stock_summary,
        description=
        "Use this to get financial summary of a stock. Input should be the stock symbol (e.g., AAPL, MSFT, TSLA)."
    ),
    Tool(
        name="TradeStrategyTool",
        func=simulate_trade_strategy,
        description=
        "Use this tool to simulate a predefined 50/200 SMA crossover strategy on a stock symbol. It returns buy/sell signal counts and latest price. Input should be the stock symbol only (e.g., AAPL)."
    ),
    Tool(
        name="CompareStockStrategies",
        func=lambda query: compare_trade_strategies(*query.split(" vs ")),
        description=
        "Compare SMA crossover strategy for two tickers. Format: 'AAPL vs MSFT'"
    ),
    Tool.from_function(
        func=smart_swing_signal,
        name="SmartSwingSignal",
        description=
        "Use this tool to get a technical trade setup using SMA, RSI, and Inside Bar patterns for swing trading. Format: Use SmartSwingSignal on [TICKER], e.g., 'Use SmartSwingSignal on TSLA'"
    )
]

memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)

# Initialize the agent
agent = initialize_agent(tools=tools,
                         llm=llm,
                         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                         memory=memory,
                         verbose=True,
                         handle_parsing_errors=True)


# Interface function
def ask_agent(query):
    return agent.run(query)
