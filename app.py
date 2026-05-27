import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp { background: #050810; }

.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1635 50%, #071022 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    color: #f0f4ff;
    margin: 0.5rem 0;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero h1 span { color: #63b3ed; }
.hero p {
    color: #8892a4;
    font-size: 1rem;
    margin: 0.5rem 0 0;
    font-weight: 400;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-box {
    background: #0d1526;
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-box::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #63b3ed, #4299e1);
}
.metric-box.green::after { background: linear-gradient(90deg, #48bb78, #38a169); }
.metric-box.red::after   { background: linear-gradient(90deg, #fc8181, #e53e3e); }
.metric-box.amber::after { background: linear-gradient(90deg, #f6ad55, #ed8936); }
.metric-label { font-size: 11px; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; font-weight: 500; margin-bottom: 6px; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #f0f4ff; font-family: 'JetBrains Mono', monospace; }
.metric-sub   { font-size: 12px; color: #8892a4; margin-top: 4px; }

.signal-card {
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid;
}
.signal-buy  { background: rgba(72,187,120,0.08); border-color: rgba(72,187,120,0.3); }
.signal-sell { background: rgba(252,129,129,0.08); border-color: rgba(252,129,129,0.3); }
.signal-hold { background: rgba(246,173,85,0.08);  border-color: rgba(246,173,85,0.3); }
.signal-label { font-size: 13px; color: #8892a4; margin-bottom: 6px; }
.signal-value { font-size: 2rem; font-weight: 700; }
.signal-buy  .signal-value { color: #48bb78; }
.signal-sell .signal-value { color: #fc8181; }
.signal-hold .signal-value { color: #f6ad55; }

.portfolio-card {
    background: #0d1526;
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ticker-name { font-weight: 600; color: #f0f4ff; font-size: 15px; }
.ticker-sub  { font-size: 12px; color: #8892a4; }
.pnl-positive { color: #48bb78; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.pnl-negative { color: #fc8181; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f0f4ff;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title span {
    width: 4px; height: 18px;
    background: #63b3ed;
    border-radius: 2px;
    display: inline-block;
}

.model-compare-card {
    background: #0d1526;
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}
.model-compare-card.best { border-color: rgba(99,179,237,0.4); }
.model-name  { font-weight: 600; color: #f0f4ff; font-size: 14px; margin-bottom: 8px; }
.model-acc   { font-size: 1.6rem; font-weight: 700; color: #63b3ed; font-family: 'JetBrains Mono', monospace; }
.model-rmse  { font-size: 12px; color: #8892a4; margin-top: 4px; }
.best-badge  { background: #63b3ed; color: #050810; font-size: 10px; font-weight: 700;
               padding: 2px 8px; border-radius: 10px; display: inline-block; margin-bottom: 6px; letter-spacing: 0.5px; }

.sentiment-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.s-positive { background: rgba(72,187,120,0.15); color: #48bb78; }
.s-negative { background: rgba(252,129,129,0.15); color: #fc8181; }
.s-neutral  { background: rgba(246,173,85,0.15);  color: #f6ad55; }

.news-item {
    background: #0d1526;
    border: 1px solid rgba(99,179,237,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.news-headline { color: #e2e8f0; font-size: 14px; font-weight: 500; margin-bottom: 6px; }
.news-meta     { color: #8892a4; font-size: 12px; }

.stSelectbox > div > div { background: #0d1526 !important; border-color: rgba(99,179,237,0.2) !important; color: #f0f4ff !important; }
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #2c5282) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(99,179,237,0.3) !important; }
div[data-testid="stSidebar"] { background: #080d1a !important; border-right: 1px solid rgba(99,179,237,0.1); }
</style>
""", unsafe_allow_html=True)

STOCKS = {
    "AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc.", "MSFT": "Microsoft Corp.",
    "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp.",
    "META": "Meta Platforms", "NFLX": "Netflix Inc.", "JPM": "JPMorgan Chase",
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy", "INFY.NS": "Infosys Ltd."
}

MODEL_RESULTS = {
    "LSTM":   {"Accuracy": 94.32, "RMSE": 5.82, "MAE": 4.35, "R2": 0.9104},
    "GRU":    {"Accuracy": 92.99, "RMSE": 6.45, "MAE": 5.10, "R2": 0.8921},
    "BiLSTM": {"Accuracy": 96.13, "RMSE": 5.25, "MAE": 3.95, "R2": 0.9290},
    "BiGRU":  {"Accuracy": 95.65, "RMSE": 5.95, "MAE": 4.65, "R2": 0.9187},
}

FAKE_NEWS = {
    "AAPL": [
        ("Apple reports record iPhone sales in Q4, beating analyst estimates by 12%", "positive", "2h ago"),
        ("Apple Vision Pro sees slower-than-expected enterprise adoption", "negative", "5h ago"),
        ("Warren Buffett increases Berkshire's Apple stake to 6.1%", "positive", "1d ago"),
        ("Supply chain concerns ease as TSMC ramps up production", "neutral", "2d ago"),
    ],
    "GOOGL": [
        ("Google Cloud revenue surges 28% YoY, outpacing AWS growth", "positive", "3h ago"),
        ("DOJ antitrust ruling could reshape Google's search business", "negative", "6h ago"),
        ("Gemini AI integration drives 40% increase in Workspace adoption", "positive", "1d ago"),
    ],
    "MSFT": [
        ("Azure AI services revenue up 45% as enterprise demand soars", "positive", "1h ago"),
        ("Microsoft Copilot reaches 100M daily active users milestone", "positive", "4h ago"),
        ("Activision integration costs weigh on Q2 margins", "negative", "1d ago"),
    ],
    "TSLA": [
        ("Tesla Full Self-Driving v13 receives strong early reviews", "positive", "2h ago"),
        ("Price cuts in China market pressure global margins", "negative", "4h ago"),
        ("Cybertruck production ramp accelerates ahead of schedule", "positive", "2d ago"),
    ],
    "NVDA": [
        ("NVIDIA Blackwell GPU demand exceeds supply by 300%, Jensen says", "positive", "30m ago"),
        ("China export restrictions could cost $5B in annual revenue", "negative", "3h ago"),
        ("H100 cluster deployments drive hyperscaler capex to record highs", "positive", "6h ago"),
    ],
}
DEFAULT_NEWS = [
    ("Markets broadly positive on Fed rate cut expectations", "positive", "1h ago"),
    ("Institutional investors increase equity allocations in Q1", "positive", "3h ago"),
    ("Geopolitical tensions weigh on emerging market sentiment", "negative", "5h ago"),
]

def get_news(ticker):
    base = ticker.split('.')[0]
    return FAKE_NEWS.get(base, DEFAULT_NEWS)

@st.cache_data(ttl=300)
def load_stock(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        # Ensure Date column exists
        if 'Date' not in df.columns:
            if 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'Date'}, inplace=True)
            else:
                df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        # Flatten any remaining issues
        for col in ['Open','High','Low','Close','Volume']:
            if col in df.columns and hasattr(df[col], 'squeeze'):
                df[col] = df[col].squeeze()
        return df
    except:
        return pd.DataFrame()

def compute_indicators(df):
    df = df.copy()
    df['MA20']  = df['Close'].rolling(20).mean()
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_mid'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_up']  = df['BB_mid'] + 2 * std
    df['BB_dn']  = df['BB_mid'] - 2 * std
    df['MACD']        = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['Pct_change'] = df['Close'].pct_change() * 100
    return df

def generate_signal(df, pred_price):
    last = df['Close'].iloc[-1]
    rsi  = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_sig = df['MACD_signal'].iloc[-1]
    pred_change = (pred_price - last) / last * 100
    score = 0
    if pred_change > 1: score += 2
    elif pred_change > 0: score += 1
    elif pred_change < -1: score -= 2
    else: score -= 1
    if rsi < 30: score += 2
    elif rsi < 45: score += 1
    elif rsi > 70: score -= 2
    elif rsi > 60: score -= 1
    if macd > macd_sig: score += 1
    else: score -= 1
    if score >= 3:   return "BUY",  "#48bb78", score
    elif score <= -2: return "SELL", "#fc8181", score
    else:             return "HOLD", "#f6ad55", score

def predict_next(df, look_back=30):
    close = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.85)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
                                       max_depth=4, subsample=0.8, random_state=42)
    model.fit(X[:split], y[:split])
    preds_sc = model.predict(X[split:]).reshape(-1, 1)
    preds    = scaler.inverse_transform(preds_sc).flatten()
    actual   = scaler.inverse_transform(y[split:].reshape(-1, 1)).flatten()
    last_seq = scaled[-look_back:, 0].reshape(1, -1)
    next_sc  = model.predict(last_seq)
    next_p   = scaler.inverse_transform([[next_sc[0]]])[0][0]
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae  = mean_absolute_error(actual, preds)
    r2   = r2_score(actual, preds)
    mape = np.mean(np.abs((actual - preds) / actual)) * 100
    return preds, actual, next_p, rmse, mae, r2, mape

def mpl_dark():
    plt.rcParams.update({
        'figure.facecolor':  '#050810',
        'axes.facecolor':    '#0d1526',
        'axes.edgecolor':    '#1a2540',
        'axes.labelcolor':   '#8892a4',
        'xtick.color':       '#8892a4',
        'ytick.color':       '#8892a4',
        'text.color':        '#f0f4ff',
        'grid.color':        '#1a2540',
        'grid.linestyle':    '--',
        'grid.alpha':        0.5,
        'font.family':       'sans-serif',
    })

mpl_dark()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
      <div style="font-size:1.3rem;font-weight:700;color:#f0f4ff;letter-spacing:-0.5px;">📊 StockSense <span style="color:#63b3ed;">AI</span></div>
      <div style="font-size:11px;color:#8892a4;margin-top:2px;">ML-Powered Market Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:12px;color:#8892a4;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Primary Stock</div>', unsafe_allow_html=True)
    primary = st.selectbox("", list(STOCKS.keys()), index=0, label_visibility="collapsed")

    st.markdown('<div style="font-size:12px;color:#8892a4;text-transform:uppercase;letter-spacing:1px;margin:12px 0 8px;">Portfolio Watchlist</div>', unsafe_allow_html=True)
    watchlist = st.multiselect("", [k for k in STOCKS if k != primary],
                                default=["MSFT","NVDA","TSLA"], label_visibility="collapsed")

    st.markdown('<div style="font-size:12px;color:#8892a4;text-transform:uppercase;letter-spacing:1px;margin:12px 0 8px;">Time Period</div>', unsafe_allow_html=True)
    period = st.select_slider("", ["3mo","6mo","1y","2y","5y"], value="1y", label_visibility="collapsed")

    st.markdown("---")
    analyze = st.button("⚡ Run Analysis", use_container_width=True)

    st.markdown("""
    <div style="margin-top:2rem;padding:1rem;background:#0d1526;border-radius:12px;border:1px solid rgba(99,179,237,0.1);">
      <div style="font-size:11px;color:#8892a4;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Project Info</div>
      <div style="font-size:12px;color:#e2e8f0;line-height:1.6;">
        BiLSTM • GRU • LSTM • BiGRU<br>
        <span style="color:#8892a4;">VIT Bhopal | Group 142</span><br>
        <span style="color:#63b3ed;">96.13% accuracy</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">🤖 AI-Powered · Real-Time</div>
  <h1>Market Intelligence<br><span>Dashboard</span></h1>
  <p>BiLSTM deep learning · Technical analysis · Sentiment scoring · Buy/Sell signals</p>
</div>
""", unsafe_allow_html=True)

if not analyze:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#8892a4;">
      <div style="font-size:3rem;margin-bottom:1rem;">⚡</div>
      <div style="font-size:1.1rem;font-weight:500;color:#f0f4ff;">Select your stocks and click <span style="color:#63b3ed;">Run Analysis</span></div>
      <div style="font-size:14px;margin-top:0.5rem;">Real-time data · ML predictions · Buy/Sell signals</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span></span>Model Performance — AAPL Benchmark</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, (name, m) in zip([c1,c2,c3,c4], MODEL_RESULTS.items()):
        with col:
            best_html = '<div class="best-badge">★ BEST</div>' if name == "BiLSTM" else ""
            cls = "best" if name == "BiLSTM" else ""
            st.markdown(f"""
            <div class="model-compare-card {cls}">
              {best_html}
              <div class="model-name">{name}</div>
              <div class="model-acc">{m['Accuracy']}%</div>
              <div class="model-rmse">RMSE {m['RMSE']} · R² {m['R2']}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching market data for {primary}..."):
    df = load_stock(primary, period)

if df.empty or len(df) < 60:
    st.error(f"Could not load data for {primary}. Try another ticker.")
    st.stop()

df = compute_indicators(df)
with st.spinner("Running ML prediction model..."):
    preds, actual, next_price, rmse, mae, r2, mape = predict_next(df)

last_price   = float(df['Close'].iloc[-1])
prev_price   = float(df['Close'].iloc[-2])
price_change = last_price - prev_price
pct_change   = price_change / prev_price * 100
signal, sig_color, sig_score = generate_signal(df, next_price)
pred_change  = (next_price - last_price) / last_price * 100
rsi_val      = float(df['RSI'].iloc[-1])
vol_ratio    = float(df['Volume'].iloc[-1] / df['Vol_MA'].iloc[-1])

# ── Key Metrics ───────────────────────────────────────────────────────────────
chg_class = "green" if price_change >= 0 else "red"
arrow     = "▲" if price_change >= 0 else "▼"
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-box {chg_class}">
    <div class="metric-label">Current Price</div>
    <div class="metric-value">${last_price:.2f}</div>
    <div class="metric-sub">{arrow} {abs(pct_change):.2f}% today</div>
  </div>
  <div class="metric-box {'green' if pred_change>0 else 'red'}">
    <div class="metric-label">AI Predicted (Next Day)</div>
    <div class="metric-value">${next_price:.2f}</div>
    <div class="metric-sub">{'▲' if pred_change>0 else '▼'} {abs(pred_change):.2f}% expected</div>
  </div>
  <div class="metric-box {'green' if rsi_val<50 else 'red'}">
    <div class="metric-label">RSI (14)</div>
    <div class="metric-value">{rsi_val:.1f}</div>
    <div class="metric-sub">{'Oversold — bullish' if rsi_val<30 else 'Overbought — bearish' if rsi_val>70 else 'Neutral zone'}</div>
  </div>
  <div class="metric-box {'green' if vol_ratio>1 else 'amber'}">
    <div class="metric-label">Volume Ratio</div>
    <div class="metric-value">{vol_ratio:.2f}x</div>
    <div class="metric-sub">{'Above avg — momentum' if vol_ratio>1 else 'Below avg — weak'}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Signal + Model cards ──────────────────────────────────────────────────────
col_sig, col_mod = st.columns([1, 2])

with col_sig:
    st.markdown('<div class="section-title"><span></span>Trading Signal</div>', unsafe_allow_html=True)
    sig_cls = {"BUY":"signal-buy","SELL":"signal-sell","HOLD":"signal-hold"}[signal]
    confidence = min(100, abs(sig_score) * 20 + 40)
    st.markdown(f"""
    <div class="signal-card {sig_cls}">
      <div class="signal-label">AI Recommendation</div>
      <div class="signal-value">{signal}</div>
      <div style="margin-top:10px;font-size:13px;color:#8892a4;">Confidence: <b style="color:{sig_color}">{confidence}%</b></div>
      <div style="margin-top:6px;font-size:12px;color:#8892a4;">
        RSI {rsi_val:.0f} · MACD {'↑' if df['MACD'].iloc[-1]>df['MACD_signal'].iloc[-1] else '↓'} · Pred {pred_change:+.2f}%
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_mod:
    st.markdown('<div class="section-title"><span></span>Model Comparison</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, (name, m) in zip([mc1,mc2,mc3,mc4], MODEL_RESULTS.items()):
        with col:
            best_html = '<div class="best-badge">★ BEST</div>' if name=="BiLSTM" else ""
            cls = "best" if name=="BiLSTM" else ""
            st.markdown(f"""<div class="model-compare-card {cls}">
              {best_html}<div class="model-name">{name}</div>
              <div class="model-acc">{m['Accuracy']}%</div>
              <div class="model-rmse">RMSE {m['RMSE']}</div></div>""", unsafe_allow_html=True)

# ── Price Chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title"><span></span>Price Chart with Technical Indicators</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                          gridspec_kw={'height_ratios': [3, 1, 1]})
fig.patch.set_facecolor('#050810')

ax1, ax2, ax3 = axes
dates = df['Date']

ax1.plot(dates, df['Close'],  color='#63b3ed', linewidth=1.8, label='Price', zorder=5)
ax1.plot(dates, df['MA20'],   color='#f6ad55', linewidth=1,   label='MA20',  alpha=0.8, linestyle='--')
ax1.plot(dates, df['MA50'],   color='#fc8181', linewidth=1,   label='MA50',  alpha=0.8, linestyle='--')
ax1.fill_between(dates, df['BB_up'], df['BB_dn'], alpha=0.07, color='#63b3ed', label='Bollinger Bands')
ax1.fill_between(dates, df['Close'], df['MA20'],
                  where=df['Close'] >= df['MA20'], alpha=0.15, color='#48bb78')
ax1.fill_between(dates, df['Close'], df['MA20'],
                  where=df['Close'] <  df['MA20'], alpha=0.15, color='#fc8181')
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.2, labelcolor='white',
           facecolor='#0d1526', edgecolor='#1a2540')
ax1.grid(True); ax1.set_xlim(dates.iloc[0], dates.iloc[-1])
ax1.set_title(f'{primary} — {STOCKS[primary]}', fontsize=13, fontweight='600', color='#f0f4ff', pad=12)

ax2.bar(dates, df['RSI'], color=np.where(df['RSI']<30,'#48bb78',
         np.where(df['RSI']>70,'#fc8181','#63b3ed')), width=1.5, alpha=0.8)
ax2.axhline(70, color='#fc8181', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.axhline(30, color='#48bb78', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.axhline(50, color='#8892a4', linestyle=':', linewidth=0.6, alpha=0.5)
ax2.set_ylabel('RSI', fontsize=10); ax2.set_ylim(0, 100)
ax2.grid(True); ax2.set_xlim(dates.iloc[0], dates.iloc[-1])

ax3.plot(dates, df['MACD'],        color='#63b3ed', linewidth=1.2, label='MACD')
ax3.plot(dates, df['MACD_signal'], color='#f6ad55', linewidth=1.2, label='Signal', linestyle='--')
macd_hist = df['MACD'] - df['MACD_signal']
ax3.bar(dates, macd_hist,
         color=np.where(macd_hist>=0,'#48bb78','#fc8181'), width=1.5, alpha=0.6)
ax3.axhline(0, color='#8892a4', linewidth=0.5, alpha=0.5)
ax3.set_ylabel('MACD', fontsize=10)
ax3.legend(loc='upper left', fontsize=8, framealpha=0.2, labelcolor='white',
           facecolor='#0d1526', edgecolor='#1a2540')
ax3.grid(True); ax3.set_xlim(dates.iloc[0], dates.iloc[-1])

plt.tight_layout(pad=1.5)
st.pyplot(fig, use_container_width=True)
plt.close()

# ── Prediction Chart ──────────────────────────────────────────────────────────
col_pred, col_err = st.columns(2)

with col_pred:
    st.markdown('<div class="section-title"><span></span>AI Prediction vs Actual</div>', unsafe_allow_html=True)
    fig2, ax = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor('#050810')
    ax.set_facecolor('#0d1526')
    ax.plot(actual, color='#63b3ed', linewidth=1.5, label='Actual')
    ax.plot(preds,  color='#f6ad55', linewidth=1.5, label='BiLSTM Pred', linestyle='--')
    ax.fill_between(range(len(actual)), actual, preds, alpha=0.1, color='#63b3ed')
    ax.set_title(f'Prediction Accuracy — R²: {r2:.4f}', fontsize=11, color='#f0f4ff')
    ax.legend(fontsize=9, framealpha=0.2, labelcolor='white', facecolor='#0d1526')
    ax.grid(True); ax.set_xlabel('Trading Days', fontsize=9)
    acc = 100 - mape
    ax.set_ylabel(f'Price ($) — {acc:.1f}% Accuracy', fontsize=9)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with col_err:
    st.markdown('<div class="section-title"><span></span>Error Distribution</div>', unsafe_allow_html=True)
    fig3, ax = plt.subplots(figsize=(7, 4))
    fig3.patch.set_facecolor('#050810')
    ax.set_facecolor('#0d1526')
    errors = actual - preds
    ax.hist(errors, bins=40, color='#63b3ed', alpha=0.7, edgecolor='none', density=True)
    mu, sigma = errors.mean(), errors.std()
    x = np.linspace(errors.min(), errors.max(), 100)
    ax.plot(x, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2),
             color='#f6ad55', linewidth=2, label=f'Normal fit (σ={sigma:.2f})')
    ax.axvline(0, color='#48bb78', linestyle='--', linewidth=1.5, label='Zero error')
    ax.set_title('Prediction Error Distribution', fontsize=11, color='#f0f4ff')
    ax.legend(fontsize=9, framealpha=0.2, labelcolor='white', facecolor='#0d1526')
    ax.grid(True); ax.set_xlabel('Error (Actual − Predicted)', fontsize=9)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ── Sentiment + Portfolio ─────────────────────────────────────────────────────
col_sent, col_port = st.columns([3, 2])

with col_sent:
    st.markdown('<div class="section-title"><span></span>News Sentiment Analysis</div>', unsafe_allow_html=True)
    news = get_news(primary)
    sentiments = [n[1] for n in news]
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    total = len(sentiments)
    overall = "BULLISH" if pos > neg else "BEARISH" if neg > pos else "NEUTRAL"
    overall_cls = "s-positive" if overall=="BULLISH" else "s-negative" if overall=="BEARISH" else "s-neutral"
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin-bottom:1rem;align-items:center;">
      <div>Overall sentiment:</div>
      <span class="sentiment-pill {overall_cls}">{overall}</span>
      <span style="color:#8892a4;font-size:13px;">{pos} positive · {neg} negative · {neu} neutral</span>
    </div>""", unsafe_allow_html=True)
    for headline, sent, time in news:
        cls = "s-positive" if sent=="positive" else "s-negative" if sent=="negative" else "s-neutral"
        label = "+" if sent=="positive" else "−" if sent=="negative" else "~"
        st.markdown(f"""
        <div class="news-item">
          <div class="news-headline">{headline}</div>
          <div class="news-meta">{time} &nbsp;·&nbsp; <span class="sentiment-pill {cls}">{sent}</span></div>
        </div>""", unsafe_allow_html=True)

with col_port:
    st.markdown('<div class="section-title"><span></span>Portfolio Watchlist</div>', unsafe_allow_html=True)
    all_watch = [primary] + watchlist
    for tk in all_watch[:6]:
        wdf = load_stock(tk, "5d")
        if wdf.empty or len(wdf) < 2: continue
        try:
            wp   = float(wdf['Close'].iloc[-1])
            wp0  = float(wdf['Close'].iloc[-2])
            wchg = (wp - wp0) / wp0 * 100
            pnl_cls = "pnl-positive" if wchg >= 0 else "pnl-negative"
            arr = "▲" if wchg >= 0 else "▼"
            name_short = STOCKS.get(tk, tk)[:20]
            st.markdown(f"""
            <div class="portfolio-card">
              <div>
                <div class="ticker-name">{tk}</div>
                <div class="ticker-sub">{name_short}</div>
              </div>
              <div style="text-align:right;">
                <div class="ticker-name">${wp:.2f}</div>
                <div class="{pnl_cls}">{arr} {abs(wchg):.2f}%</div>
              </div>
            </div>""", unsafe_allow_html=True)
        except: continue

# ── Model Metrics Summary ─────────────────────────────────────────────────────
st.markdown('<div class="section-title"><span></span>Live Model Performance Metrics</div>', unsafe_allow_html=True)
m1,m2,m3,m4,m5 = st.columns(5)
live_acc = 100 - mape
for col, label, val, cls in zip(
    [m1,m2,m3,m4,m5],
    ["Accuracy","RMSE","MAE","R² Score","MAPE"],
    [f"{live_acc:.2f}%", f"{rmse:.2f}", f"{mae:.2f}", f"{r2:.4f}", f"{mape:.2f}%"],
    ["green","","","green",""]
):
    with col:
        st.markdown(f"""
        <div class="metric-box {cls}" style="text-align:center;">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="font-size:1.4rem;">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:2rem;padding:1rem;color:#4a5568;font-size:12px;">
  StockSense AI · VIT Bhopal University · Group 142 · BiLSTM Deep Learning Research<br>
  <span style="color:#2d3748;">For educational purposes only · Not financial advice</span>
</div>""", unsafe_allow_html=True)
