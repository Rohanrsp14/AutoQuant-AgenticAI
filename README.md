# AutoQuant: Your Financial Research Agent 🧠📈

AutoQuant is your intelligent assistant for **stock research automation**. It helps you quickly analyze public companies, generate backtested trade setups, and even export findings as PDF deliverables — all powered by LLMs + LangChain.

---

## 🚀 Features

- **SmartSwingSignal**  
  Combines 50/200 SMA trend, RSI pullback, and Inside Bar breakout logic to generate execution-ready swing trade setups with stop loss, target, and RRR.

- **SMA Crossover Strategy Tool**  
  Simulates golden cross (50 > 200 SMA) for any stock, showing buy/sell signal counts and price trend.

- **Compare Strategy Tool**  
  Compares performance between two tickers using SMA crossover logic.

- **Stock Summary Tool**  
  Fetches current price, P/E ratio, market cap, and daily stats from Yahoo Finance.

- **PDF Export** *(Optional)*  
  Export trade ideas and summaries for sharing or documentation.

---

## 📊 Demo

👉 Try it Live: [Streamlit App Link](https://your-streamlit-app-url)  
💻 Repo: [AutoQuant on GitHub](https://github.com/Rohanrsp14/AutoQuant)

---

## 🛠️ How It Works

This app is built using:

- **LangChain** agents with tools
- **OpenRouter API** for LLMs
- **Yahoo Finance (yfinance)** for market data
- **Streamlit** for UI
- **Plotly** (optional) for interactive charts

---

## 🧱 File Structure

```bash
├── app.py                  # Streamlit frontend
├── agent.py                # LangChain agent + tools
├── main.py                 # Optional CLI/testing
├── test_strategy.py        # Strategy simulation tests
├── requirements.txt
└── README.md
```

---

## 📦 Setup (Locally)

```bash
git clone https://github.com/Rohanrsp14/AutoQuant.git
cd AutoQuant
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚖️ License

[MIT License](LICENSE)

---

## 👨‍💻 Author

**Rohan**  
Senior Product Manager | Data-Driven | Passionate about Trading, Tech & Automation  
[LinkedIn](https://www.linkedin.com/in/your-link) · [GitHub](https://github.com/Rohanrsp14)
