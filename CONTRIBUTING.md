# Contributing to StockSense AI

Thank you for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/charchit24bai10434-cmyk/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```

## Project Structure

- `app/app.py` — Streamlit dashboard
- `src/models.py` — Model architectures (LSTM, GRU, BiLSTM, BiGRU)
- `src/preprocessing.py` — Data pipeline
- `src/indicators.py` — Technical indicators
- `notebooks/` — Jupyter notebooks for experiments

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test thoroughly
5. Submit a Pull Request

## Ideas for Contributions

- Add Transformer/Attention-based model
- Integrate live news API (NewsAPI)
- Add backtesting framework with Sharpe ratio
- Multi-step ahead forecasting
- Additional stocks / indices

## Code Style

- Follow PEP 8
- Add docstrings to all functions
- Keep functions small and focused
