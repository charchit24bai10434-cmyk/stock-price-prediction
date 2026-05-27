<div align="center">

# 📊 StockSense AI
### ML-Powered Stock Market Intelligence Dashboard

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://stock-price-prediction-2wgat2ju8shkyi7m5xtn7g.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![VIT Bhopal](https://img.shields.io/badge/VIT_Bhopal-Group_142-blue?style=for-the-badge)](https://vitbhopal.ac.in)

**A full-stack AI dashboard for real-time stock price prediction, technical analysis, sentiment scoring, and Buy/Sell signal generation using BiLSTM deep learning.**

[🚀 Live Demo](https://stock-price-prediction-2wgat2ju8shkyi7m5xtn7g.streamlit.app) · [📓 Notebook](notebooks/Stock_Price_Prediction_BiLSTM.ipynb) · [📄 Report](assets/ProjectReport.pdf) · [🐛 Issues](https://github.com/charchit24bai10434-cmyk/stock-price-prediction/issues)

</div>

---

## 🎯 What This Project Does

StockSense AI is an end-to-end machine learning project that:

- **Predicts** next-day stock prices using deep learning (BiLSTM, LSTM, GRU, BiGRU)
- **Analyzes** technical indicators — RSI, MACD, Bollinger Bands, Moving Averages
- **Generates** Buy/Sell/Hold signals with confidence scores
- **Scores** market sentiment from financial news headlines
- **Tracks** a live portfolio watchlist with real-time price changes

---

## 🏆 Model Results — AAPL (2020–2024)

| Model | Accuracy | RMSE | MAE | R² Score |
|-------|----------|------|-----|----------|
| LSTM | 94.32% | 5.82 | 4.35 | 0.9104 |
| GRU | 92.99% | 6.45 | 5.10 | 0.8921 |
| **BiLSTM** ⭐ | **96.13%** | **5.25** | **3.95** | **0.9290** |
| BiGRU | 95.65% | 5.95 | 4.65 | 0.9187 |

> **BiLSTM wins** — bidirectional processing captures both past and future temporal dependencies, giving it a structural advantage over unidirectional models.

---

## 🧠 Why BiLSTM?

Standard LSTMs process sequences in one direction (past → future). **BiLSTM processes both forward and backward**, giving the model richer context about price patterns.

```
Standard LSTM:   t-60 → t-59 → ... → t-1 → t  (forward only)
BiLSTM:          t-60 → t-59 → ... → t-1 → t  (forward)
                 t    → t-1  → ... → t-59 → t-60  (backward)
                 Combined output → Dense → Prediction
```

This is why BiLSTM achieves **96.13% accuracy** vs LSTM's 94.32%.

---

## 🏗️ Architecture

```
Input: 60-day window of normalized Close prices
         ↓
Bidirectional LSTM (128 units) ← captures forward & backward patterns
         ↓
Dropout (0.3) ← prevents overfitting
         ↓
Bidirectional LSTM (64 units)
         ↓
Dropout (0.2)
         ↓
Dense (32, ReLU)
         ↓
Dense (1) → Predicted next-day price
```

**Training config:** Adam (lr=0.0001) · Batch size 32 · 50 epochs · EarlyStopping

---

## 🚀 Features

### 📈 Real-Time Dashboard
- Live stock data via Yahoo Finance API
- 12 stocks supported: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META, NFLX, JPM, Reliance, TCS, Infosys

### 📊 Technical Analysis
- **RSI** (14-day) — Overbought/oversold detection
- **MACD** — Momentum and trend direction
- **Bollinger Bands** — Volatility bands
- **MA20 / MA50** — Short and medium-term trend lines

### 🤖 AI Buy/Sell Signal
- Composite scoring from RSI + MACD + predicted price change
- Confidence percentage displayed per signal
- Color-coded: 🟢 BUY · 🔴 SELL · 🟡 HOLD

### 📰 Sentiment Analysis
- Headline-level sentiment scoring (positive/negative/neutral)
- Overall market sentiment summary per stock
- Stock-specific news feed

### 💼 Portfolio Watchlist
- Track multiple stocks simultaneously
- Live price + daily % change
- Color-coded P&L display

---

## 🗂️ Project Structure

```
stock-price-prediction/
├── app/
│   └── app.py                          # Streamlit web application
├── notebooks/
│   └── Stock_Price_Prediction_BiLSTM.ipynb  # Full training notebook
├── src/
│   ├── models.py                       # Model architectures
│   ├── preprocessing.py                # Data pipeline
│   └── indicators.py                   # Technical indicators
├── results/
│   ├── model_comparison.png            # Performance comparison chart
│   ├── bilstm_prediction.png           # Actual vs predicted
│   └── error_distribution.png          # Error analysis
├── assets/
│   └── ProjectReport.pdf               # Full project report
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚡ Quick Start

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/charchit24bai10434-cmyk/stock-price-prediction.git
cd stock-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app/app.py
```

### Run the Notebook

```bash
cd notebooks
jupyter notebook Stock_Price_Prediction_BiLSTM.ipynb
```

Or open directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d46qzqQl9ddH0FlR95PsrqWE1O5FJ87Z)

---

## 🔬 Methodology

### 1. Data Collection
```python
import yfinance as yf
df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
```

### 2. Preprocessing
```python
# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(close_prices)

# Create 60-day sequences
X, y = create_sequences(scaled, look_back=60)
```

### 3. Model Training
```python
model = build_bilstm(input_shape=(60, 1))
model.fit(X_train, y_train,
          epochs=50, batch_size=32,
          callbacks=[EarlyStopping(patience=10)])
```

### 4. Evaluation
```python
# RMSE: 5.25 | MAE: 3.95 | R²: 0.9290 | MAPE: 3.87%
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Data | Yahoo Finance API (yfinance) |
| Analysis | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git / GitHub |

---

## 🔮 Future Improvements

- [ ] Real-time BiLSTM inference (currently using GB for cloud deployment)
- [ ] Live news scraping via NewsAPI or RSS feeds
- [ ] Attention mechanism / Transformer architecture
- [ ] Walk-forward backtesting with Sharpe ratio
- [ ] SHAP explainability for model decisions
- [ ] Multi-feature input (Volume, Open, High, Low + technicals)
- [ ] FastAPI backend + React frontend for production

---

## 👥 Team

| Name | Registration |
|------|-------------|
| Mohit Kumar Mishra | 24BAI10174 |
| Charchit Bari | 24BAI10434 |
| Anshika Kumari | 24BAI10333 |
| Siddhi Sibangi Kar | 24BAI10280 |
| Anushka Thakur | 24BAI10438 |
| Vishal Dubey | 24BAI10041 |

**Supervisor:** Dr. Anil Kumar Yadav  
**Institution:** VIT Bhopal University, School of Computing Science & Engineering

---

## 📚 References

1. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8).
2. Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder*. arXiv:1406.1078.
3. Graves, A. & Schmidhuber, J. (2005). *Framewise phoneme classification with bidirectional LSTM networks*. IJCNN.
4. Schuster, M. & Paliwal, K. (1997). *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ by Group 142 | VIT Bhopal University | 2026</sub>
</div>
