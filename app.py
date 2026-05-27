"""
Stock Price Prediction Web App
BiLSTM Deep Learning Demo — VIT Bhopal Group 142
Uses sklearn for lightweight deployment on Streamlit Cloud
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stock Predictor | BiLSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin-bottom: 0.5rem; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; }
    .metric-card {
        background: #1e2a3a; border-radius: 10px; padding: 1.2rem;
        text-align: center; border-left: 4px solid #e94560;
    }
    .metric-card h3 { color: #e94560; font-size: 1.8rem; margin: 0; }
    .metric-card p  { color: #a8b2d8; margin: 0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>📈 Stock Price Prediction Using Deep Learning</h1>
  <p>Comparative Analysis of LSTM · GRU · BiLSTM · BiGRU</p>
  <p style="color:#64ffda; font-size:0.85rem;">VIT Bhopal University &nbsp;|&nbsp; Group 142</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.selectbox("Stock Ticker", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"], index=0)
    start  = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end    = st.date_input("End Date",   value=pd.to_datetime("2024-01-01"))
    st.markdown("---")
    st.subheader("Model Settings")
    look_back = st.slider("Look-back Window (days)", 10, 60, 30, 5)
    run = st.button("🚀 Load & Predict", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    **👥 Team Members**  
    Mohit Kumar Mishra  
    Charchit Bari  
    Anshika Kumari  
    Siddhi Sibangi Kar  
    Anushka Thakur  
    Vishal Dubey  
    
    **Supervisor:** Dr. Anil Kumar Yadav  
    **VIT Bhopal University**
    """)

# ── Static results (always shown) ─────────────────────────────────────────────
st.subheader("📊 Model Performance Comparison (AAPL 2020–2024)")

static = {
    "LSTM":   {"Accuracy": 94.32, "RMSE": 5.82, "MAE": 4.35, "color": "steelblue"},
    "GRU":    {"Accuracy": 92.99, "RMSE": 6.45, "MAE": 5.10, "color": "green"},
    "BiLSTM": {"Accuracy": 96.13, "RMSE": 5.25, "MAE": 3.95, "color": "#e94560"},
    "BiGRU":  {"Accuracy": 95.65, "RMSE": 5.95, "MAE": 4.65, "color": "orange"},
}

col1, col2, col3, col4 = st.columns(4)
for col, (name, m) in zip([col1, col2, col3, col4], static.items()):
    trophy = "🏆 " if name == "BiLSTM" else ""
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <h3>{trophy}{name}</h3>
          <p>Accuracy: <b>{m['Accuracy']}%</b></p>
          <p>RMSE: {m['RMSE']}</p>
          <p>MAE: {m['MAE']}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.patch.set_facecolor('#0d1117')
names  = list(static.keys())
colors = [static[n]["color"] for n in names]
for ax, metric in zip(axes, ['Accuracy', 'RMSE', 'MAE']):
    vals = [static[n][metric] for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor='none', width=0.5)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9, color='white')
    ax.set_facecolor('#161b22')
    ax.set_title(metric, color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.yaxis.grid(True, color='#30363d', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
plt.suptitle('Model Performance Comparison', color='white', fontsize=13)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Live Data Section ──────────────────────────────────────────────────────────
if not run:
    st.info("👈 Select a stock and click **Load & Predict** to see live price data and predictions.")
    st.stop()

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    df.reset_index(inplace=True)
    return df

with st.spinner(f"Fetching {ticker} data..."):
    df = load_data(ticker, start, end)

if df.empty:
    st.error("No data found. Try a different ticker or date range.")
    st.stop()

st.subheader(f"📉 {ticker} Price History")
fig, ax = plt.subplots(figsize=(12, 3))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
ax.plot(df['Date'], df['Close'], color='#64ffda', linewidth=1.2)
ax.set_title(f"{ticker} Closing Price ({start} → {end})", color='white')
ax.tick_params(colors='white')
ax.spines[['top','right','left','bottom']].set_color('#30363d')
ax.yaxis.grid(True, color='#30363d', linestyle='--', alpha=0.4)
st.pyplot(fig)
plt.close()

# Preprocessing + simple ML prediction
close = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

X, y = [], []
for i in range(look_back, len(scaled)):
    X.append(scaled[i-look_back:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

with st.spinner("Running prediction model..."):
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds_sc = model.predict(X_test).reshape(-1, 1)
    preds  = scaler.inverse_transform(preds_sc)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(actual, preds))
mae  = mean_absolute_error(actual, preds)
r2   = r2_score(actual, preds)
mape = np.mean(np.abs((actual - preds) / actual)) * 100
acc  = 100 - mape

st.subheader("📈 Live Prediction Results")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy",  f"{acc:.2f}%")
c2.metric("RMSE",      f"{rmse:.2f}")
c3.metric("MAE",       f"{mae:.2f}")
c4.metric("R² Score",  f"{r2:.4f}")
c5.metric("MAPE",      f"{mape:.2f}%")

col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot(actual, color='royalblue', label='Actual',    linewidth=1.2)
    ax.plot(preds,  color='#e94560',  label='Predicted', linewidth=1.2, linestyle='--')
    ax.set_title('Actual vs Predicted Price', color='white')
    ax.legend(facecolor='#1e2a3a', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines[['top','right','left','bottom']].set_color('#30363d')
    st.pyplot(fig)
    plt.close()

with col_b:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    errors = actual.flatten() - preds.flatten()
    ax.hist(errors, bins=40, color='#e94560', alpha=0.85, edgecolor='none')
    ax.axvline(0, color='#64ffda', linestyle='--', linewidth=1.5)
    ax.set_title('Prediction Error Distribution', color='white')
    ax.tick_params(colors='white')
    ax.spines[['top','right','left','bottom']].set_color('#30363d')
    st.pyplot(fig)
    plt.close()

latest_pred   = preds[-1][0]
latest_actual = actual[-1][0]
st.success(f"🔮 Latest Prediction: **${latest_pred:.2f}** | Actual: **${latest_actual:.2f}** | Error: **${abs(latest_actual-latest_pred):.2f}**")

st.markdown("---")
st.caption("📚 Full BiLSTM implementation available in the Jupyter notebook. This demo uses Gradient Boosting for lightweight cloud deployment.")
