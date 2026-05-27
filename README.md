# 📊 StockSense AI — Deep Learning Stock Forecasting

> Interactive stock market forecasting using deep learning and real-time financial data.

---

## 🏆 Results at a Glance

| Model  | Accuracy | RMSE | MAE  | R² Score |
|--------|----------|------|------|----------|
| LSTM   | 94.32%   | 5.82 | 4.35 | —        |
| GRU    | 92.99%   | 6.45 | 5.10 | —        |
| **BiLSTM** | **96.13%** | **5.25** | **3.95** | **0.9290** |
| BiGRU  | 95.65%   | 5.95 | 4.65 | —        |

🥇 **Best Performing Model: BiLSTM**

---

## 📌 Project Overview

StockSense AI is a stock forecasting platform that combines deep learning research with an interactive prediction application for stock market analysis.

Using historical Apple (AAPL) stock market data from **January 2020 to January 2024**, this project evaluates LSTM, GRU, BiLSTM, and BiGRU models to identify the most effective forecasting approach.

The project also includes an interactive **Streamlit web application** for real-time stock analysis and visualization.

---

## 🚀 Features

- Real-time stock market data retrieval using Yahoo Finance
- Stock forecasting and predictive analytics
- Comparative deep learning model analysis (LSTM, GRU, BiLSTM, BiGRU)
- Interactive Streamlit dashboard
- Visual stock trend analysis
- Performance evaluation with RMSE, MAE, MAPE, and R²
- Time-series preprocessing with sliding window sequence generation

---

## 🗂️ Repository Structure

```bash
stock-price-prediction/
├── app.py
├── requirements.txt
├── stock_forecasting_experiments.ipynb
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/charchit24bai10434-cmyk/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook stock_forecasting_experiments.ipynb
```

Or open in Google Colab:

[Open in Colab](https://colab.research.google.com/drive/1d46qzqQl9ddH0FlR95PsrqWE1O5FJ87Z)

### 4. Launch Interactive App

```bash
streamlit run app.py
```

---

## 🧠 Methodology

```text
Data Acquisition (Yahoo Finance)
        ↓
Data Preprocessing
  • Close price extraction
  • MinMax normalization
  • 60-day sequence generation
  • Train-test split
        ↓
Model Development
  • LSTM
  • GRU
  • BiLSTM
  • BiGRU
        ↓
Training
  • Adam optimizer
  • EarlyStopping
  • ReduceLROnPlateau
        ↓
Evaluation
  • RMSE
  • MAE
  • MAPE
  • R² Score
        ↓
Deployment
  • Streamlit interactive dashboard
```

---

## 📊 Best Model Architecture (BiLSTM)

```text
Input Sequence (60 timesteps)
        ↓
Bidirectional LSTM (128 units)
        ↓
Dropout (0.3)
        ↓
Bidirectional LSTM (64 units)
        ↓
Dropout (0.2)
        ↓
Dense Layer (32, ReLU)
        ↓
Output Layer (1)
```

---

## 📈 Key Results

- **BiLSTM achieved highest performance**
- **R² Score: 0.9290**
- **MAPE: 3.87%**
- Consistent prediction performance across testing period
- Stable forecasting with low prediction error

---

## 🔮 Future Improvements

- Add technical indicators (RSI, MACD, Bollinger Bands)
- Multi-stock forecasting support
- Multi-day forecasting
- Transformer-based forecasting models
- Sentiment analysis integration
- Explainable AI using SHAP/LIME

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- yfinance

---

## 📄 License

Open for educational and portfolio demonstration purposes.
