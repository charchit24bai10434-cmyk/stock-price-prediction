# 📈 Stock Price Prediction Using Deep Learning

> **A Comparative Analysis of LSTM, GRU, BiLSTM, and BiGRU Models**
>
> VIT Bhopal University | School of Computing Science & Engineering | Group 142

---

## 🏆 Results at a Glance

| Model  | Accuracy | RMSE | MAE  | R² Score |
|--------|----------|------|------|----------|
| LSTM   | 94.32%   | 5.82 | 4.35 | —        |
| GRU    | 92.99%   | 6.45 | 5.10 | —        |
| **BiLSTM** | **96.13%** | **5.25** | **3.95** | **0.9290** |
| BiGRU  | 95.65%   | 5.95 | 4.65 | —        |

🥇 **Winner: BiLSTM** — Highest accuracy, lowest errors, most reliable predictions.

---

## 📌 Project Overview

This project develops and compares four deep learning models for stock price forecasting using historical market data. Using Apple (AAPL) stock data from **January 2020 to January 2024**, we demonstrate that Bidirectional LSTM (BiLSTM) outperforms other RNN variants by capturing both forward and backward temporal dependencies.

### Objectives
- Develop deep learning models for stock price prediction
- Compare LSTM, GRU, BiLSTM, and BiGRU architectures
- Identify the best model for financial time-series forecasting

---

## 🗂️ Repository Structure

```
stock-price-prediction/
├── app/
│   └── app.py                  # Streamlit interactive web app
├── notebooks/
│   └── Stock_Price_Prediction_BiLSTM.ipynb   # Full Jupyter notebook
├── src/
│   └── (utility scripts)
├── results/
│   └── (generated charts & plots)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
cd notebooks
jupyter notebook Stock_Price_Prediction_BiLSTM.ipynb
```

Or open directly in **Google Colab**:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d46qzqQl9ddH0FlR95PsrqWE1O5FJ87Z)

### 4. Launch the Web App
```bash
cd app
streamlit run app.py
```

---

## 🧠 Methodology

```
Data Acquisition (yfinance)
        ↓
Data Preprocessing
  • Close price extraction
  • MinMax normalization [0,1]
  • 60-day sliding window sequences
  • 80/20 train-test split
        ↓
Model Building (TensorFlow/Keras)
  • LSTM  →  GRU  →  BiLSTM  →  BiGRU
        ↓
Training
  • Adam optimizer (lr=0.0001)
  • Batch size: 32 | Epochs: 50
  • EarlyStopping + ReduceLROnPlateau
        ↓
Evaluation
  • RMSE, MAE, MAPE, R², Accuracy
  • Error distribution analysis
  • Actual vs Predicted plots
```

---

## 📊 Model Architecture (BiLSTM — Best Model)

```
Input (60 days × 1 feature)
    ↓
Bidirectional LSTM (128 units) → captures forward & backward patterns
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.2)
    ↓
Dense (32, ReLU)
    ↓
Dense (1) — predicted next-day price
```

---

## 📈 Key Results

- **BiLSTM R² Score: 0.9290** — the model explains 92.9% of price variance
- **MAPE: 3.87%** — predictions within ~4% of actual price on average
- **Stationary errors** — consistent performance throughout the test period
- **Near-zero error bias** — normally distributed prediction errors

---

## 🔮 Future Work

- [ ] Add technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Incorporate sentiment analysis from news/social media
- [ ] Experiment with Attention mechanisms and Transformers
- [ ] Deploy as a real-time prediction API
- [ ] Extend to multi-stock and multi-day forecasting
- [ ] Apply SHAP/LIME for model explainability

---

## 👥 Team

| Name | Registration No. |
|------|-----------------|
| Mohit Kumar Mishra | 24BAI10174 |
| Charchit Bari | 24BAI10434 |
| Anshika Kumari | 24BAI10333 |
| Siddhi Sibangi Kar | 24BAI10280 |
| Anushka Thakur | 24BAI10438 |
| Vishal Dubey | 24BAI10041 |

**Supervisor:** Dr. Anil Kumar Yadav  
**Reviewers:** Dr. Jyoti Chauhan | Dr. Komarasamy G  
**Institution:** VIT Bhopal University, Kothrikalan, Sehore, MP

---

## 📚 References

1. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
2. Graves, A. (2013). *Generating Sequences with Recurrent Neural Networks*. arXiv:1308.0850.
3. Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder*. arXiv:1406.1078.
4. Kaggle Stock Market Dataset — historical price data.
5. TensorFlow/Keras documentation — [keras.io](https://keras.io)

---

## 📄 License

This project is for academic purposes at VIT Bhopal University.

---

<p align="center">Made with ❤️ by Group 142 | VIT Bhopal</p>
