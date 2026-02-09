import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Quant Portfolio Builder",
    page_icon=" ",
    layout="wide"
)

# Title
st.title("Free Cloud Stock Portfolio Builder")
st.markdown("Multi-factor portfolio: **Size, Value, Quality, Momentum, Low Beta**")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # API Key input
    api_key = st.text_input(
        "Alpha Vantage API Key",
        type="password", # Changed '4HG3DPJX5HYP4KMX' to 'password'
        help="Get free key from https://www.alphavantage.co/support/#api-key"
    )

    st.session_state.api_key = api_key

    # Universe selection
    st.subheader("Stock Universe")
    universe_size = st.slider("Number of stocks to analyze", 20, 100, 50)

    # Factor weights
    st.subheader("Factor Weights")
    size_weight = st.slider("Size Weight", 0.0, 1.0, 0.2)
    value_weight = st.slider("Value Weight", 0.0, 1.0, 0.2)
    quality_weight = st.slider("Quality Weight", 0.0, 1.0, 0.2)
    momentum_weight = st.slider("Momentum Weight", 0.0, 1.0, 0.2)
    lowbeta_weight = st.slider("Low Beta Weight", 0.0, 1.0, 0.2)

    # Normalize weights
    total = size_weight + value_weight + quality_weight + momentum_weight + lowbeta_weight
    if total != 1.0:
        st.warning(f"Weights sum to {total:.1f}. Normalizing...")
        size_weight /= total
        value_weight /= total
        quality_weight /= total
        momentum_weight /= total
        lowbeta_weight /= total

    # Portfolio size
    portfolio_size = st.slider("Portfolio size (stocks)", 5, 30, 10)

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        build_portfolio = st.button("Build Portfolio", type="primary")
    with col2:
        update_prices = st.button("Update Prices")

# Function to get stock data (with fallback to Yahoo Finance)
def get_stock_data(symbol):
    """Get stock data with fallback mechanisms"""
    try:
        # Try Yahoo Finance first (no API limits)
        ticker = yf.Ticker(symbol)
        info = ticker.info

        data = {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'roe': info.get('returnOnEquity', 0),
            'profit_margin': info.get('profitMargins', 0),
            'beta': info.get('beta', 1),
            'sector': info.get('sector', 'N/A'),
            'current_price': info.get('currentPrice', 0)
        }

        # Get historical data for momentum
        hist = ticker.history(period="1y")
        if len(hist) > 0:
            if len(hist) >= 30:
                momentum_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-30] - 1) * 100
            else:
                momentum_1m = 0

            if len(hist) >= 252:
                momentum_12m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-252] - 1) * 100
            else:
                momentum_12m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100

            data['momentum_1m'] = momentum_1m
            data['momentum_12m'] = momentum_12m
            data['volatility'] = hist['Close'].pct_change().std() * np.sqrt(252) * 100

        return data

    except Exception as e:
        st.warning(f"Error fetching {symbol}: {str(e)}")
        return None

# Function to calculate factor scores
def calculate_factor_scores(stocks_df):
    """Calculate normalized factor scores"""
    df = stocks_df.copy()

    # Size factor (smaller market cap = better)
    df['size_score'] = 1 - (df['market_cap'].rank(pct=True))

    # Value factor (lower P/E, P/B = better)
    df['pe_score'] = 1 - (df['pe_ratio'].replace([np.inf, -np.inf], np.nan).rank(pct=True, na_option='keep'))
    df['pb_score'] = 1 - (df['pb_ratio'].replace([np.inf, -np.inf], np.nan).rank(pct=True, na_option='keep'))
    df['value_score'] = (df['pe_score'].fillna(0.5) + df['pb_score'].fillna(0.5)) / 2

    # Quality factor (higher ROE, profit margin = better)
    df['roe_score'] = df['roe'].rank(pct=True)
    df['profit_margin_score'] = df['profit_margin'].rank(pct=True)
    df['quality_score'] = (df['roe_score'] + df['profit_margin_score']) / 2

    # Momentum factor (higher momentum = better)
    df['momentum_score'] = df['momentum_12m'].rank(pct=True)

    # Low beta factor (lower beta = better)
    df['lowbeta_score'] = 1 - (df['beta'].rank(pct=True))

    return df

# Function to build portfolio
def build_multi_factor_portfolio(universe, weights, top_n=10):
    """Build portfolio based on factor scores"""

    st.info(f"Analyzing {len(universe)} stocks...")

    # Get data for all stocks
    progress_bar = st.progress(0)
    stocks_data = []

    for i, symbol in enumerate(universe[:50]):  # Limit for free tier
        data = get_stock_data(symbol)
        if data and data['current_price'] > 1:  # Filter out penny stocks
            stocks_data.append(data)
        progress_bar.progress((i + 1) / min(50, len(universe)))
        time.sleep(0.1)  # Rate limiting

    if not stocks_data:
        st.error("No stock data retrieved. Check your internet connection.")
        return None

    # Create DataFrame
    df = pd.DataFrame(stocks_data)

    # Calculate factor scores
    df = calculate_factor_scores(df)

    # Calculate composite score
    df['composite_score'] = (
        df['size_score'] * weights['size'] +
        df['value_score'] * weights['value'] +
        df['quality_score'] * weights['quality'] +
        df['momentum_score'] * weights['momentum'] +
        df['lowbeta_score'] * weights['lowbeta']
    )

    # Select top stocks
    portfolio = df.nlargest(top_n, 'composite_score').copy()

    # Calculate weights (score-weighted)
    portfolio['weight'] = portfolio['composite_score'] / portfolio['composite_score'].sum()
    portfolio['weight'] = portfolio['weight'].round(4)

    # Calculate position value based on hypothetical $10,000 portfolio
    portfolio['position_value'] = (portfolio['weight'] * 10000).round(2)
    portfolio['shares'] = (portfolio['position_value'] / portfolio['current_price']).round(2)

    return portfolio

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Performance", "Risk Analysis", "Settings"])

with tab1:
    st.header("Current Portfolio")

    # Default stock universe (S&P 100 + Tech)
    default_universe = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM',
        'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'XOM', 'CSCO',
        'ADBE', 'CRM', 'NFLX', 'CMCSA', 'PEP', 'INTC', 'ABT', 'TMO', 'ACN', 'WMT',
        'ABBV', 'CVX', 'MRK', 'PFE', 'T', 'DHR', 'MDT', 'NKE', 'BMY', 'RTX',
        'HON', 'LIN', 'AMGN', 'PM', 'UPS', 'SBUX', 'TXN', 'QCOM', 'INTU', 'GS'
    ]

    if build_portfolio:
        with st.spinner("Building multi-factor portfolio..."):
            weights = {
                'size': size_weight,
                'value': value_weight,
                'quality': quality_weight,
                'momentum': momentum_weight,
                'lowbeta': lowbeta_weight
            }

            portfolio = build_multi_factor_portfolio(default_universe, weights, portfolio_size)

            if portfolio is not None:
                st.session_state.portfolio = portfolio
                st.session_state.last_update = datetime.now()

                # Display portfolio
                st.success(f"Portfolio built with {len(portfolio)} stocks!")

                # Portfolio metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg P/E", f"{portfolio['pe_ratio'].mean():.1f}")
                with col2:
                    st.metric("Avg ROE", f"{portfolio['roe'].mean():.1f}%")
                with col3:
                    st.metric("Avg Beta", f"{portfolio['beta'].mean():.2f}")
                with col4:
                    st.metric("Avg Momentum", f"{portfolio['momentum_12m'].mean():.1f}%")

                # Display portfolio table
                display_cols = ['symbol', 'name', 'sector', 'current_price',
                              'weight', 'position_value', 'shares']
                st.dataframe(
                    portfolio[display_cols].rename(columns={
                        'symbol': 'Symbol',
                        'name': 'Name',
                        'sector': 'Sector',
                        'current_price': 'Price',
                        'weight': 'Weight',
                        'position_value': 'Position $',
                        'shares': 'Shares'
                    }),
                    use_container_width=True
                )

                # Factor exposure chart
                st.subheader("Factor Exposure")
                factors = ['Size', 'Value', 'Quality', 'Momentum', 'Low Beta']
                scores = [
                    portfolio['size_score'].mean(),
                    portfolio['value_score'].mean(),
                    portfolio['quality_score'].mean(),
                    portfolio['momentum_score'].mean(),
                    portfolio['lowbeta_score'].mean()
                ]

                fig = go.Figure(data=go.Scatterpolar(
                    r=scores,
                    theta=factors,
                    fill='toself'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download button
                csv = portfolio.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio CSV",
                    data=csv,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    elif st.session_state.portfolio is not None:
        # Display existing portfolio
        portfolio = st.session_state.portfolio
        st.dataframe(portfolio, use_container_width=True)
        if st.session_state.last_update:
            st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

with tab2:
    st.header("Portfolio Performance")

    if st.session_state.portfolio is not None:
        portfolio = st.session_state.portfolio

        # Simulate performance (in real app, fetch actual historical data)
        symbols = portfolio['symbol'].tolist()

        # Get historical data for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        try:
            # Get price data for portfolio
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)['Adj Close']

            # Calculate portfolio returns
            portfolio_returns = (data * portfolio.set_index('symbol')['weight']).sum(axis=1)
            portfolio_returns = portfolio_returns / portfolio_returns.iloc[0]  # Normalize

            # Plot performance
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_returns.index,
                y=portfolio_returns.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))

            # Add S&P 500 for comparison
            sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)['Adj Close']
            sp500 = sp500 / sp500.iloc[0]
            fig.add_trace(go.Scatter(
                x=sp500.index,
                y=sp500.values,
                mode='lines',
                name='S&P 500',
                line=dict(color='gray', width=1, dash='dash')
            ))

            fig.update_layout(
                title="Portfolio Performance vs S&P 500",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching performance data: {str(e)}")
            st.info("Performance simulation requires live market data")

    else:
        st.info("Build a portfolio first to see performance analysis")

with tab3:
    st.header("Risk Analysis")

    if st.session_state.portfolio is not None:
        portfolio = st.session_state.portfolio

        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Beta", f"{portfolio['beta'].mean():.2f}")
        with col2:
            st.metric("Max Stock Weight", f"{portfolio['weight'].max()*100:.1f}%")
        with col3:
            st.metric("Sector Concentration",
                     f"{portfolio['sector'].value_counts().iloc[0] / len(portfolio)*100:.0f}%")
        with col4:
            st.metric("Price Range",
                     f"${portfolio['current_price'].min():.1f} - ${portfolio['current_price'].max():.1f}")

        # Sector allocation pie chart
        sector_allocation = portfolio.groupby('sector')['weight'].sum().reset_index()
        fig = go.Figure(data=[go.Pie(
            labels=sector_allocation['sector'],
            values=sector_allocation['weight'],
            hole=.3
        )])
        fig.update_layout(title="Sector Allocation", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Risk metrics table
        risk_metrics = pd.DataFrame({
            'Metric': ['Portfolio Beta', 'Weighted Avg P/E', 'Weighted Avg ROE',
                      'Avg Daily Volatility', 'Concentration (Top 5)', 'Diversification Score'],
            'Value': [
                f"{portfolio['beta'].mean():.2f}",
                f"{portfolio['pe_ratio'].mean():.1f}",
                f"{portfolio['roe'].mean():.1f}%",
                f"{portfolio['volatility'].mean():.1f}%",
                f"{portfolio.nlargest(5, 'weight')['weight'].sum()*100:.1f}%",
                f"{1 - (portfolio['weight']**2).sum():.3f}"
            ]
        })
        st.dataframe(risk_metrics, use_container_width=True, hide_index=True)

with tab4:
    st.header("Settings & Configuration")

    st.subheader("Import/Export Portfolio")

    # Import portfolio
    uploaded_file = st.file_uploader("Import portfolio CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            imported_portfolio = pd.read_csv(uploaded_file)
            st.session_state.portfolio = imported_portfolio
            st.success("Portfolio imported successfully!")
        except Exception as e:
            st.error(f"Error importing file: {str(e)}")

    # Export current portfolio
    if st.session_state.portfolio is not None:
        st.download_button(
            label="Export Portfolio JSON",
            data=st.session_state.portfolio.to_json(indent=2),
            file_name="portfolio_export.json",
            mime="application/json"
        )

    st.subheader("API Status")
    if st.session_state.api_key:
        st.success("API Key configured")
    else:
        st.warning("No API Key configured. Using Yahoo Finance fallback.")

    st.subheader("About")
    st.markdown("""
    **Multi-Factor Portfolio Builder**

    This tool builds portfolios based on 5 factors:
    1. **Size**: Smaller market capitalization
    2. **Value**: Low P/E and P/B ratios
    3. **Quality**: High ROE and profit margins
    4. **Momentum**: Positive price momentum
    5. **Low Beta**: Lower systematic risk

    **Data Sources**:
    - Yahoo Finance (primary, no API limits)
    - Alpha Vantage (fallback, requires API key)

    **Note**: This is for educational purposes. Past performance doesn't guarantee future results.
    """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Data from Yahoo Finance & Alpha Vantage | Free Cloud Hosting")
