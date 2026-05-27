"""
Stock Price Prediction Web App
BiLSTM Deep Learning Demo — VIT Bhopal Group 142
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor | BiLSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; }
    .metric-card {
        background: #1e2a3a; border-radius: 10px; padding: 1.2rem;
        text-align: center; border-left: 4px solid #e94560;
    }
    .metric-card h3 { color: #e94560; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #a8b2d8; margin: 0; font-size: 0.85rem; }
    .team-box {
        background: #1e2a3a; border-radius: 10px; padding: 1rem; margin-top: 1rem;
        font-size: 0.8rem; color: #a8b2d8;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📈 Stock Price Prediction</h1>
  <p>Comparative Analysis of LSTM · GRU · BiLSTM · BiGRU using Deep Learning</p>
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
    look_back  = st.slider("Look-back Window (days)", 30, 120, 60, 10)
    epochs     = st.slider("Epochs", 5, 50, 10, 5)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    model_choice = st.selectbox("Select Model", ["BiLSTM", "LSTM", "GRU", "BiGRU"])

    run = st.button("🚀 Train & Predict", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div class="team-box">
    <b>👥 Team Members</b><br>
    Mohit Kumar Mishra<br>
    Charchit Bari<br>
    Anshika Kumari<br>
    Siddhi Sibangi Kar<br>
    Anushka Thakur<br>
    Vishal Dubey<br><br>
    <b>Supervisor:</b> Dr. Anil Kumar Yadav
    </div>
    """, unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_model(name, input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    if name == "LSTM":
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
    elif name == "GRU":
        model.add(GRU(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(GRU(64))
    elif name == "BiLSTM":
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64)))
    elif name == "BiGRU":
        model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Bidirectional(GRU(64)))

    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.0001), loss='mse', metrics=['mae'])
    return model

# ── Default state: show static comparison metrics ─────────────────────────────
if not run:
    st.subheader("📊 Pre-computed Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    static = {
        "LSTM":   {"Accuracy": 94.32, "RMSE": 5.82, "MAE": 4.35},
        "GRU":    {"Accuracy": 92.99, "RMSE": 6.45, "MAE": 5.10},
        "BiLSTM": {"Accuracy": 96.13, "RMSE": 5.25, "MAE": 3.95},
        "BiGRU":  {"Accuracy": 95.65, "RMSE": 5.95, "MAE": 4.65},
    }
    for col, (name, m) in zip([col1, col2, col3, col4], static.items()):
        highlight = "🏆 " if name == "BiLSTM" else ""
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <h3>{highlight}{name}</h3>
              <p>Accuracy: <b>{m['Accuracy']}%</b></p>
              <p>RMSE: {m['RMSE']}</p>
              <p>MAE: {m['MAE']}</p>
            </div>""", unsafe_allow_html=True)

    st.info("👈 Configure your settings in the sidebar and click **Train & Predict** to run the model live.")

    # Static bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#0d1117')
    colors = ['steelblue', 'green', '#e94560', 'orange']
    names  = list(static.keys())
    for ax, metric in zip(axes, ['Accuracy', 'RMSE', 'MAE']):
        vals = [static[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor='none', width=0.5)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9, color='white')
        ax.set_facecolor('#161b22')
        ax.set_title(metric, color='white')
        ax.tick_params(colors='white')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.yaxis.grid(True, color='#30363d', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
    plt.suptitle('Model Performance Comparison (AAPL 2020–2024)', color='white', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.stop()

# ── Live Training ─────────────────────────────────────────────────────────────
st.subheader(f"🔄 Training {model_choice} on {ticker} ({start} → {end})")

with st.spinner("Fetching data from Yahoo Finance..."):
    df = load_data(ticker, str(start), str(end))

if df.empty:
    st.error("No data found. Try a different ticker or date range.")
    st.stop()

# Price chart
fig, ax = plt.subplots(figsize=(12, 3))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
ax.plot(df['Date'], df['Close'], color='#64ffda', linewidth=1.2)
ax.set_title(f"{ticker} Closing Price", color='white')
ax.tick_params(colors='white')
ax.spines[['top','right','left','bottom']].set_color('#30363d')
st.pyplot(fig)
plt.close()

# Preprocessing
data       = df[['Close']].values
scaler     = MinMaxScaler()
scaled     = scaler.fit_transform(data)
train_size = int(len(scaled) * 0.8)
train_d    = scaled[:train_size]
test_d     = scaled[train_size:]

X_train, y_train = create_sequences(train_d, look_back)
X_test,  y_test  = create_sequences(
    np.concatenate([train_d[-look_back:], test_d]), look_back
)
X_train = X_train.reshape(-1, look_back, 1)
X_test  = X_test.reshape(-1, look_back, 1)

# Train
model = build_model(model_choice, (look_back, 1))

progress = st.progress(0, text="Training in progress…")
loss_hist, val_hist = [], []

for ep in range(epochs):
    h = model.fit(X_train, y_train, epochs=1, batch_size=batch_size,
                  validation_split=0.1, verbose=0)
    loss_hist.append(h.history['loss'][0])
    val_hist.append(h.history['val_loss'][0])
    progress.progress((ep + 1) / epochs, text=f"Epoch {ep+1}/{epochs} | Loss: {loss_hist[-1]:.6f}")

st.success(f"✅ {model_choice} training complete!")

# Predictions
preds_sc = model.predict(X_test, verbose=0)
preds    = scaler.inverse_transform(preds_sc)
actual   = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(actual, preds))
mae  = mean_absolute_error(actual, preds)
r2   = r2_score(actual, preds)
mape = np.mean(np.abs((actual - preds) / actual)) * 100
acc  = 100 - mape

# Metrics
st.subheader("📈 Model Performance")
c1, c2, c3, c4, c5 = st.columns(5)
for col, label, val, fmt in zip(
    [c1, c2, c3, c4, c5],
    ["Accuracy", "RMSE", "MAE", "R² Score", "MAPE"],
    [acc, rmse, mae, r2, mape],
    ["{:.2f}%", "{:.2f}", "{:.2f}", "{:.4f}", "{:.2f}%"]
):
    col.metric(label, fmt.format(val))

# Plots
col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot(actual,  color='royalblue', label='Actual',    linewidth=1.2)
    ax.plot(preds,   color='#e94560',   label='Predicted', linewidth=1.2, linestyle='--')
    ax.set_title('Actual vs Predicted', color='white')
    ax.legend(facecolor='#1e2a3a', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines[['top','right','left','bottom']].set_color('#30363d')
    st.pyplot(fig)
    plt.close()

with col_b:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot(loss_hist, color='royalblue', label='Train Loss')
    ax.plot(val_hist,  color='#e94560',   label='Val Loss', linestyle='--')
    ax.set_title('Training Loss Curve', color='white')
    ax.set_xlabel('Epoch', color='white')
    ax.legend(facecolor='#1e2a3a', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines[['top','right','left','bottom']].set_color('#30363d')
    st.pyplot(fig)
    plt.close()

# Error Distribution
fig, ax = plt.subplots(figsize=(10, 3))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
errors = actual.flatten() - preds.flatten()
ax.hist(errors, bins=50, color='#e94560', alpha=0.8, edgecolor='none')
ax.axvline(0, color='#64ffda', linestyle='--', linewidth=1.5, label='Zero Error')
ax.set_title('Distribution of Prediction Errors', color='white')
ax.set_xlabel('Error (Actual − Predicted)', color='white')
ax.tick_params(colors='white')
ax.legend(facecolor='#1e2a3a', labelcolor='white')
ax.spines[['top','right','left','bottom']].set_color('#30363d')
st.pyplot(fig)
plt.close()

latest_pred = preds[-1][0]
latest_actual = actual[-1][0]
st.markdown(f"""
---
**🔮 Latest Prediction:** `${latest_pred:.2f}` &nbsp;|&nbsp;
**Actual:** `${latest_actual:.2f}` &nbsp;|&nbsp;
**Error:** `${abs(latest_actual - latest_pred):.2f}`
""")
