import streamlit as st
import feedparser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
import pytz

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="F&O Pro Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Configuration & Constants
# --------------------------
IST = pytz.timezone('Asia/Kolkata')

# Combined Stock List (Restricted to 5 specific stocks)
FNO_STOCKS = [
    "Reliance", "HDFC Bank", "ICICI Bank", "HCL Tech", "M&M"
]

# Ticker Mapping
STOCK_TICKER_MAP = {
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "HCL Tech": "HCLTECH.NS",
    "M&M": "M&M.NS"
}

FINANCIAL_RSS_FEEDS = [
    ("https://feeds.feedburner.com/ndtvprofit-latest", "NDTV Profit"),
    ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "ET Markets"),
    ("https://www.moneycontrol.com/rss/latestnews.xml", "Moneycontrol"),
]

POSITIVE_WORDS = ['surge', 'rally', 'gain', 'profit', 'growth', 'high', 'rise', 'up', 'bullish', 
                  'strong', 'beats', 'outperform', 'success', 'jumps', 'soars', 'positive']
NEGATIVE_WORDS = ['fall', 'drop', 'loss', 'decline', 'weak', 'down', 'crash', 'bearish',
                  'concern', 'worry', 'risk', 'plunge', 'slump', 'miss', 'negative']

# --------------------------
# Global State
# --------------------------
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "All Stocks"

# --------------------------
# Helper Functions - News
# --------------------------
def is_recent(published_time, hours_limit=48):
    try:
        if not published_time:
            return True
        pub_time = None
        if hasattr(published_time, 'tm_year'):
            pub_time = datetime(*published_time[:6])
        elif isinstance(published_time, str):
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%S%z']:
                try:
                    pub_time = datetime.strptime(published_time, fmt)
                    break
                except:
                    continue
        if pub_time:
            if pub_time.tzinfo:
                pub_time = pub_time.replace(tzinfo=None)
            cutoff_time = datetime.now() - timedelta(hours=hours_limit)
            return pub_time >= cutoff_time
        return True
    except:
        return True

def analyze_sentiment(text):
    text_lower = text.lower()
    positive_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
    negative_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = "positive"
        score = min(0.6 + (positive_count * 0.1), 0.95)
    elif negative_count > positive_count:
        sentiment = "negative"
        score = min(0.6 + (negative_count * 0.1), 0.95)
    else:
        sentiment = "neutral"
        score = 0.5
    return sentiment, round(score, 2)

def check_fno_mention(text):
    text_upper = text.upper()
    for stock in FNO_STOCKS:
        if stock.upper() in text_upper:
            return True
    return False

def get_mentioned_stocks(text):
    text_upper = text.upper()
    mentioned = []
    for stock in FNO_STOCKS:
        if stock.upper() in text_upper:
            if stock not in mentioned:
                mentioned.append(stock)
    return mentioned if mentioned else ["Other"]

def fetch_news(num_articles=15, specific_stock=None, force_new=False):
    all_articles = []
    seen_titles = set() if force_new or (specific_stock and specific_stock != "All Stocks") else {a['Title'] for a in st.session_state.news_articles}
    
    # 1. Google News
    if specific_stock and specific_stock != "All Stocks":
        priority_stocks = [specific_stock]
    else:
        priority_stocks = FNO_STOCKS[:10]  # Just check a few top stocks for general news to save time
    
    for stock in priority_stocks:
        if len(all_articles) >= num_articles: break
        try:
            url = f"https://news.google.com/rss/search?q={stock}+stock+india+when:2d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            limit = 10 if specific_stock == stock else 2
            
            for entry in feed.entries[:limit]:
                title = entry.title
                if title in seen_titles: continue
                if not is_recent(getattr(entry, 'published_parsed', None)): continue
                
                all_articles.append(entry)
                seen_titles.add(title)
        except:
            continue

    # 2. General feeds
    if len(all_articles) < num_articles:
        for feed_url, source_name in FINANCIAL_RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    title = entry.title
                    if title in seen_titles: continue
                    full_text = title + " " + getattr(entry, 'summary', '')
                    
                    if specific_stock and specific_stock != "All Stocks":
                        if specific_stock.upper() not in full_text.upper(): continue
                    elif not check_fno_mention(full_text):
                        continue
                        
                    if not is_recent(getattr(entry, 'published_parsed', None)): continue
                    all_articles.append(entry)
                    seen_titles.add(title)
                    if len(all_articles) >= num_articles: break
            except:
                continue
            if len(all_articles) >= num_articles: break

    # Process
    records = []
    for art in all_articles:
        title = art.title
        source = getattr(art, "source", {}).get("title", "Unknown") if hasattr(art, "source") else "Unknown"
        sentiment, score = analyze_sentiment(title)
        records.append({
            "Title": title,
            "Source": source,
            "Sentiment": sentiment,
            "Score": score,
            "Link": art.link,
            "Published": getattr(art, 'published', 'Recent'),
            "Stocks": get_mentioned_stocks(title + " " + getattr(art, 'summary', ''))
        })
    return records

# --------------------------
# Helper Functions - Technical Indicators
# --------------------------
def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    return sma + (std * std_dev), sma, sma - (std * std_dev)

def calculate_supertrend(df, period=10, multiplier=3):
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    
    # Simple iterative calculation
    st_val = lower_band.iloc[period]
    direc = 1
    
    supertrend_vals = [np.nan] * len(df)
    direction_vals = [np.nan] * len(df)
    
    for i in range(period, len(df)):
        curr_close = close.iloc[i]
        prev_close = close.iloc[i-1]
        prev_st = st_val
        
        if direc == 1:
            if curr_close > prev_st:
                st_val = max(lower_band.iloc[i], prev_st)
            else:
                st_val = upper_band.iloc[i]
                direc = -1
        else:
            if curr_close < prev_st:
                st_val = min(upper_band.iloc[i], prev_st)
            else:
                st_val = lower_band.iloc[i]
                direc = 1
        
        supertrend_vals[i] = st_val
        direction_vals[i] = direc
        
    return pd.Series(supertrend_vals, index=df.index), pd.Series(direction_vals, index=df.index)

def calculate_ichimoku(df):
    high9 = df['High'].rolling(window=9).max()
    low9 = df['Low'].rolling(window=9).min()
    tenkan_sen = (high9 + low9) / 2
    
    high26 = df['High'].rolling(window=26).max()
    low26 = df['Low'].rolling(window=26).min()
    kijun_sen = (high26 + low26) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    high52 = df['High'].rolling(window=52).max()
    low52 = df['Low'].rolling(window=52).min()
    senkou_span_b = ((high52 + low52) / 2).shift(26)
    
    chikou_span = df['Close'].shift(-26)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_obv(df):
    obv = [0] * len(df)
    obv[0] = df['Volume'].iloc[0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['Volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=df.index)

def calculate_fibonacci(df, lookback=50):
    recent = df.tail(lookback)
    high = recent['High'].max()
    low = recent['Low'].min()
    diff = high - low
    levels = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }
    return levels

def fetch_stock_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        # Handle MultiIndex columns if present (new yfinance update)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --------------------------
# Main UI
# --------------------------
tab1, tab2 = st.tabs(["📰 News Dashboard", "📈 Advanced Charts"])

# --- TAB 1: NEWS ---
with tab1:
    st.header("F&O News Monitor")
    
    col1, col2, col3 = st.columns([2, 5, 2])
    
    with col1:
        stock_filter = st.selectbox(
            "Filter by Stock", 
            ["All Stocks"] + sorted(FNO_STOCKS), 
            index=0 if st.session_state.selected_stock == "All Stocks" else sorted(FNO_STOCKS).index(st.session_state.selected_stock) + 1 if st.session_state.selected_stock in FNO_STOCKS else 0
        )
        if stock_filter != st.session_state.selected_stock:
            st.session_state.selected_stock = stock_filter
            st.rerun()

    with col3:
        if st.button("🔄 Refresh News", use_container_width=True):
            with st.spinner("Fetching latest news..."):
                new_items = fetch_news(15, st.session_state.selected_stock, force_new=True)
                if new_items:
                    st.session_state.news_articles = new_items + st.session_state.news_articles
                    # Deduplicate
                    seen = set()
                    unique = []
                    for x in st.session_state.news_articles:
                        if x['Title'] not in seen:
                            unique.append(x)
                            seen.add(x['Title'])
                    st.session_state.news_articles = unique[:100]
            st.rerun()

    # Initial load
    if not st.session_state.news_articles:
        with st.spinner("Loading initial news..."):
            st.session_state.news_articles = fetch_news(15, st.session_state.selected_stock)

    # Filter display
    filtered_news = st.session_state.news_articles
    if st.session_state.selected_stock != "All Stocks":
        filtered_news = [n for n in st.session_state.news_articles if st.session_state.selected_stock in n['Stocks']]

    # Metrics
    if filtered_news:
        df_news = pd.DataFrame(filtered_news)
        sentiment_counts = df_news['Sentiment'].value_counts()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Articles", len(filtered_news))
        m2.metric("🟢 Positive", sentiment_counts.get("positive", 0))
        m3.metric("⚪ Neutral", sentiment_counts.get("neutral", 0))
        m4.metric("🔴 Negative", sentiment_counts.get("negative", 0))
        
        st.divider()
        
        for art in filtered_news:
            with st.container():
                cols = st.columns([1, 15])
                with cols[0]:
                    emoji = "🟢" if art['Sentiment'] == "positive" else "🔴" if art['Sentiment'] == "negative" else "⚪"
                    st.write(f"## {emoji}")
                with cols[1]:
                    st.markdown(f"[{art['Title']}]({art['Link']})")
                    st.caption(f"**Source:** {art['Source']} • **Confidence:** {art['Score']} • **Mentions:** {', '.join(art['Stocks'])}")
                    st.divider()
    else:
        st.info("No news found for the selected criteria.")

# --- TAB 2: ADVANCED CHARTS ---
with tab2:
    st.header("Advanced Technical Charts")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        chart_stock = st.selectbox("Select Stock", sorted(FNO_STOCKS), key="chart_stock")
    with c2:
        chart_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    with c3:
        chart_interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    with c4:
        chart_type = st.selectbox("Chart Type", ["Candlestick + MA", "Ichimoku Cloud", "Fibonacci", "Volume + OBV"], index=0)

    # Fetch Data
    ticker = STOCK_TICKER_MAP.get(chart_stock)
    if ticker:
        with st.spinner("Loading data..."):
            df = fetch_stock_data(ticker, chart_period, chart_interval)
        
        if df is not None and not df.empty:
            # Calculations
            df['SMA_20'] = calculate_sma(df['Close'], 20)
            df['SMA_50'] = calculate_sma(df['Close'], 50)
            df['EMA_12'] = calculate_ema(df['Close'], 12)
            df['EMA_26'] = calculate_ema(df['Close'], 26)
            df['BB_upper'], df['BB_mid'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])
            df['Supertrend'], df['ST_Dir'] = calculate_supertrend(df)
            
            # --- PLOTTING ---
            fig = go.Figure()
            
            # Base Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price'
            ))
            
            if chart_type == "Candlestick + MA":
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)))
                
                # Supertrend overlay
                st_green = df[df['ST_Dir'] == 1]
                st_red = df[df['ST_Dir'] == -1]
                fig.add_trace(go.Scatter(x=st_green.index, y=st_green['Supertrend'], mode='markers', marker=dict(color='green', size=2), name='Support'))
                fig.add_trace(go.Scatter(x=st_red.index, y=st_red['Supertrend'], mode='markers', marker=dict(color='red', size=2), name='Resistance'))

            elif chart_type == "Ichimoku Cloud":
                tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)
                fig.add_trace(go.Scatter(x=df.index, y=tenkan, name='Tenkan', line=dict(color='red', width=1)))
                fig.add_trace(go.Scatter(x=df.index, y=kijun, name='Kijun', line=dict(color='blue', width=1)))
                fig.add_trace(go.Scatter(x=df.index, y=span_a, name='Span A', line=dict(color='lightgreen', width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=df.index, y=span_b, name='Span B', line=dict(color='lightcoral', width=0), fill='tonexty', fillcolor='rgba(0,250,0,0.1)', showlegend=False))

            elif chart_type == "Fibonacci":
                levels = calculate_fibonacci(df)
                colors = ['gray', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']
                for i, (k, v) in enumerate(levels.items()):
                    fig.add_hline(y=v, line_dash="dash", line_color=colors[i%len(colors)], annotation_text=f"Fib {k}")

            elif chart_type == "Volume + OBV":
                # OBV in a separate subplot would be better but keeping simple for now
                obv = calculate_obv(df)
                # Rescale OBV to price for visibility or just show simpler volume
                # Let's show Bollinger + Volume bars scaled
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dot')))
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dot')))

            fig.update_layout(
                title=f"{chart_stock} Technical Analysis",
                height=600,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                yaxis_title="Price (₹)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics Row
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            chg = curr - prev
            pct = (chg/prev)*100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"₹{curr:.2f}", f"{chg:.2f} ({pct:.2f}%)")
            m2.metric("RSI (14)", f"{calculate_rsi(df['Close']).iloc[-1]:.2f}")
            
            macd, sig, hist = calculate_macd(df['Close'])
            m3.metric("MACD", f"{macd.iloc[-1]:.2f}")
            
            vol = df['Volume'].iloc[-1]
            m4.metric("Volume", f"{vol:,.0f}")
            
        else:
            st.error("No data found for this ticker.")
    else:
        st.error("Ticker mapping not found.")
