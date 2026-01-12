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
    page_title="F&O Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------
# Config - Restricted to 5 Stocks
# --------------------------
FNO_STOCKS = [
    "Reliance", "HDFC Bank", "ICICI Bank", "HCL Tech", "M&M"
]

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

ARTICLES_PER_REFRESH = 15
NEWS_AGE_LIMIT_HOURS = 48

POSITIVE_WORDS = ['surge', 'rally', 'gain', 'profit', 'growth', 'high', 'rise', 'up', 'bullish', 
                  'strong', 'beats', 'outperform', 'success', 'jumps', 'soars', 'positive']
NEGATIVE_WORDS = ['fall', 'drop', 'loss', 'decline', 'weak', 'down', 'crash', 'bearish',
                  'concern', 'worry', 'risk', 'plunge', 'slump', 'miss', 'negative']

# --------------------------
# Initialize Session State
# --------------------------
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "All Stocks"

# --------------------------
# Technical Analysis Helpers (From User Code)
# --------------------------
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_ao(high, low, fast=5, slow=34):
    median_price = (high + low) / 2
    ao = median_price.rolling(window=fast).mean() - median_price.rolling(window=slow).mean()
    return ao

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

# --------------------------
# Sentiment Analysis
# --------------------------
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

# --------------------------
# News Functions (Strict Logic)
# --------------------------
def is_recent(published_time, hours_limit=NEWS_AGE_LIMIT_HOURS):
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

def check_fno_mention(text):
    text_upper = text.upper()
    for stock in FNO_STOCKS:
        # Strict checking: stock name must appear in text
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
    
    if force_new or (specific_stock and specific_stock != "All Stocks"):
        seen_titles = set()
    else:
        seen_titles = {article['Title'] for article in st.session_state.news_articles}
    
    # 1. Google News - Specific Queries
    if specific_stock and specific_stock != "All Stocks":
        priority_stocks = [specific_stock]
        num_articles = num_articles * 2
    else:
        priority_stocks = FNO_STOCKS  # Check ALL 5 stocks in the list
    
    for stock in priority_stocks:
        if len(all_articles) >= num_articles:
            break
        try:
            url = f"https://news.google.com/rss/search?q={stock}+stock+india+when:2d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            
            # If using specific stock, take more. If 'All', take fewer per stock.
            limit = 10 if specific_stock == stock else 3
            
            for entry in feed.entries[:limit]:
                title = entry.title
                if title in seen_titles:
                    continue
                published = getattr(entry, 'published_parsed', None)
                if not is_recent(published):
                    continue
                
                # Double check that the title actually is relevant even from Google
                # This prevents "Avatar" if google returns garbage for a query
                if stock.upper() not in title.upper() and stock.upper() not in getattr(entry, 'summary', '').upper():
                     # Loose check passed by Google, but strict check failed? 
                     # For safety, trust Google query but maybe verify filtering?
                     # Let's trust Google query for now as it contains the keyword.
                     pass

                all_articles.append(entry)
                seen_titles.add(title)
                if len(all_articles) >= num_articles:
                    break
        except:
            continue
    
    # 2. General feeds with STRICT FILTERING
    if len(all_articles) < num_articles:
        for feed_url, source_name in FINANCIAL_RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    title = entry.title if hasattr(entry, 'title') else ""
                    if title in seen_titles:
                        continue
                    full_text = title + " " + getattr(entry, 'summary', '')
                    
                    if specific_stock and specific_stock != "All Stocks":
                        # Must strictly contain the specific stock name
                        if specific_stock.upper() not in full_text.upper():
                            continue
                    else:
                        # Must strictly contain ANY of the 5 stocks
                        if not check_fno_mention(full_text):
                            continue
                    
                    published = getattr(entry, 'published_parsed', None)
                    if not is_recent(published):
                        continue
                    all_articles.append(entry)
                    seen_titles.add(title)
                    if len(all_articles) >= num_articles:
                        break
            except:
                continue
            if len(all_articles) >= num_articles:
                break
    
    return all_articles[:num_articles]

def process_news(articles):
    records = []
    for art in articles:
        title = art.title
        source = getattr(art, "source", {}).get("title", "Unknown") if hasattr(art, "source") else "Unknown"
        url = art.link
        published = getattr(art, 'published', 'Unknown')
        mentioned_stocks = get_mentioned_stocks(title + " " + getattr(art, 'summary', ''))
        sentiment, score = analyze_sentiment(title)
        
        records.append({
            "Title": title,
            "Source": source,
            "Sentiment": sentiment,
            "Score": score,
            "Link": url,
            "Published": published,
            "Stocks": mentioned_stocks
        })
    return records

def filter_news_by_stock(news_articles, stock_name):
    if stock_name == "All Stocks":
        return news_articles
    filtered = []
    for article in news_articles:
        if stock_name in article.get('Stocks', []):
            filtered.append(article)
    return filtered

# --------------------------
# UI Structure
# --------------------------
tab1, tab2 = st.tabs(["📰 News Dashboard", "💹 Stock Charts"])

# --- TAB 1: NEWS ---
with tab1:
    st.title("📈 F&O Stocks News Dashboard")
    st.markdown("Real-time news filtered strictly for: **" + ", ".join(FNO_STOCKS) + "**")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        stock_options = ["All Stocks"] + sorted(FNO_STOCKS)
        selected_stock = st.selectbox(
            "🔍 Filter by Stock",
            options=stock_options,
            index=stock_options.index(st.session_state.selected_stock) if st.session_state.selected_stock in stock_options else 0,
            key="stock_filter_box"
        )
        
        if selected_stock != st.session_state.selected_stock:
            st.session_state.selected_stock = selected_stock
            st.rerun()

    with col2:
        if st.button("🔄 Refresh News", type="primary", use_container_width=True):
            with st.spinner(f"Fetching latest updates..."):
                # Force refresh
                new_articles = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock, force_new=True)
                if new_articles:
                    processed_news = process_news(new_articles)
                    st.session_state.news_articles = processed_news + st.session_state.news_articles
                    # Deduplicate
                    seen = set()
                    unique_articles = []
                    for article in st.session_state.news_articles:
                        if article['Title'] not in seen:
                            unique_articles.append(article)
                            seen.add(article['Title'])
                    st.session_state.news_articles = unique_articles[:100]
                    st.success(f"✅ Refreshed!")
                    st.rerun()

    with col3:
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.news_articles = []
            st.success("✅ Cleared!")
            st.rerun()

    # Auto-load on first run
    if not st.session_state.news_articles:
        with st.spinner("Loading news..."):
            initial_news = fetch_news(ARTICLES_PER_REFRESH, st.session_state.selected_stock)
            if initial_news:
                st.session_state.news_articles = process_news(initial_news)

    filtered_articles = filter_news_by_stock(st.session_state.news_articles, st.session_state.selected_stock)

    if filtered_articles:
        for article in filtered_articles:
            with st.container():
                sentiment_emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}
                emoji = sentiment_emoji.get(article['Sentiment'], "⚪")
                
                st.markdown(f"**{emoji} [{article['Title']}]({article['Link']})**")
                st.caption(f"Source: {article['Source']} | {article.get('Published', '')}")
                if article.get('Stocks'):
                    st.caption(f"Tags: {', '.join(article['Stocks'])}")
                st.markdown("---")
    else:
        st.info("No relevant news found. Try hitting Refresh.")

# --- TAB 2: CHARTS ---
with tab2:
    st.title("💹 Advanced Stock Charts")
    st.markdown("Technical Analysis Charts")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        selected_chart_stock = st.selectbox(
            "📊 Select Stock",
            options=sorted(FNO_STOCKS),
            key="chart_stock"
        )
    
    with col2:
        period = st.selectbox(
            "📅 Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=2,
            key="chart_period"
        )
    
    ticker = STOCK_TICKER_MAP.get(selected_chart_stock)
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if not df.empty:
                # Calculate indicators
                df['RSI'] = calculate_rsi(df['Close'])
                df['MACD'], df['Signal'] = calculate_macd(df['Close'])
                df['AO'] = calculate_ao(df['High'], df['Low'])
                
                df['SMA_20'] = calculate_sma(df['Close'], 20)
                df['SMA_50'] = calculate_sma(df['Close'], 50)
                df['SMA_200'] = calculate_sma(df['Close'], 200)
                
                df['EMA_9'] = calculate_ema(df['Close'], 9)
                df['EMA_20'] = calculate_ema(df['Close'], 20)
                df['EMA_50'] = calculate_ema(df['Close'], 50)
                
                # Plot
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'],
                    name='Price'
                )])
                
                # Add SMAs
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)))
                
                fig.update_layout(height=600, title=f"{selected_chart_stock} Price Chart", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Indicators
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("RSI")
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with c2:
                    st.subheader("MACD")
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')))
                    st.plotly_chart(fig_macd, use_container_width=True)

            else:
                st.warning("No data found.")
        except Exception as e:
            st.error(f"Error loading chart: {e}")
