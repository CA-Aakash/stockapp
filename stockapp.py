import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(
    page_title="Stock Price Data Fetcher",
    page_icon="📈",
    layout="wide"
)

def fetch_stock_data(df, years_back=5):
    """
    Fetch historical stock data for tickers from uploaded DataFrame
    """
    # Extract symbols from the DataFrame
    if 'Symbol' in df.columns:
        symbols = df['Symbol'].tolist()
        companies = df['Company Name'].tolist() if 'Company Name' in df.columns else df.iloc[:, 0].tolist()
    else:
        # Use second column for symbols, first for company names
        symbols = df.iloc[:, 1].tolist()
        companies = df.iloc[:, 0].tolist()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Dictionary to store all stock data
    all_stock_data = {}
    successful_fetches = 0
    
    # Fetch data for each stock
    for i, (symbol, company) in enumerate(zip(symbols, companies)):
        try:
            status_text.text(f"Fetching data for {symbol} ({i+1}/{len(symbols)})...")
            progress_bar.progress((i + 1) / len(symbols))
            
            # Add .NS for NSE stocks (Indian market)
            ticker_symbol = f"{symbol}.NS"
            
            # Download stock data
            stock = yf.Ticker(ticker_symbol)
            hist_data = stock.history(start=start_date, end=end_date)
            
            if not hist_data.empty:
                # Add stock symbol and company name columns
                hist_data['Stock_Symbol'] = symbol
                hist_data['Company_Name'] = company
                
                # Reset index to make Date a column
                hist_data = hist_data.reset_index()
                
                # Fix timezone issue - convert to timezone-naive datetime
                if 'Date' in hist_data.columns:
                    hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.tz_localize(None)
                
                # Store in dictionary
                all_stock_data[symbol] = hist_data
                successful_fetches += 1
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
            
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {str(e)}")
            continue
    
    status_text.text(f"✅ Completed! Successfully fetched data for {successful_fetches}/{len(symbols)} stocks")
    
    return all_stock_data

def create_excel_download(all_stock_data):
    """
    Create Excel file with multiple sheets for download
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Create summary sheet
        summary_data = []
        for symbol, data in all_stock_data.items():
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                summary_data.append({
                    'Symbol': symbol,
                    'Company_Name': data['Company_Name'].iloc[0],
                    'Latest_Price': round(latest_price, 2),
                    'Data_Points': len(data),
                    'Start_Date': data['Date'].min().strftime('%Y-%m-%d'),
                    'End_Date': data['Date'].max().strftime('%Y-%m-%d')
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Combine all data into one sheet
        if all_stock_data:
            combined_data = pd.concat(all_stock_data.values(), ignore_index=True)
            # Ensure Date column is timezone-naive
            if 'Date' in combined_data.columns:
                combined_data['Date'] = pd.to_datetime(combined_data['Date']).dt.tz_localize(None)
            combined_data.to_excel(writer, sheet_name='All_Stock_Data', index=False)
    
    return output.getvalue(), summary_df

def filter_data_by_date(data, start_date, end_date):
    """Filter data based on date range"""
    mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
    return data[mask]

def create_price_chart(data, symbol):
    """Create interactive price chart with volume"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} - Stock Price', 'Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.6)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} - Price and Volume Analysis',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        height=600,
        showlegend=False
    )
    
    return fig

def create_moving_average_chart(data, symbol):
    """Create moving average chart"""
    # Calculate moving averages
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'],
        mode='lines', name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA_20'],
        mode='lines', name='20-Day MA',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA_50'],
        mode='lines', name='50-Day MA',
        line=dict(color='green', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA_200'],
        mode='lines', name='200-Day MA',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title=f'{symbol} - Moving Averages Analysis',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        height=500
    )
    
    return fig

def create_returns_chart(data, symbol):
    """Create returns analysis chart"""
    data['Daily_Return'] = data['Close'].pct_change() * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} - Daily Returns', 'Returns Distribution'),
        vertical_spacing=0.15
    )
    
    # Daily returns line chart
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Daily_Return'],
            mode='lines',
            name='Daily Returns (%)',
            line=dict(color='purple')
        ),
        row=1, col=1
    )
    
    # Returns histogram
    fig.add_trace(
        go.Histogram(
            x=data['Daily_Return'].dropna(),
            nbinsx=50,
            name='Returns Distribution',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} - Returns Analysis',
        height=600,
        showlegend=False
    )
    
    return fig

def create_volatility_chart(data, symbol):
    """Create volatility analysis chart"""
    # Calculate rolling volatility (30-day)
    data['Volatility'] = data['Close'].pct_change().rolling(window=30).std() * np.sqrt(252) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Volatility'],
        mode='lines',
        name='30-Day Volatility (%)',
        line=dict(color='red', width=2),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title=f'{symbol} - Volatility Analysis (Annualized)',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        height=400
    )
    
    return fig

def create_performance_comparison(all_stock_data, selected_stocks):
    """Create performance comparison chart"""
    fig = go.Figure()
    
    for symbol in selected_stocks:
        if symbol in all_stock_data:
            data = all_stock_data[symbol].copy()
            # Normalize to 100 for comparison
            data['Normalized'] = (data['Close'] / data['Close'].iloc[0]) * 100
            
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Normalized'],
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title='Stock Performance Comparison (Normalized to 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        height=500
    )
    
    return fig

# Main Streamlit App
def main():
    st.title("📈 Advanced Stock Price Data Fetcher")
    st.markdown("Upload your Excel file with stock symbols and get comprehensive historical analysis!")
    
    # Initialize session state
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** your Excel file with stock symbols
        2. **Preview** the data to ensure correct format
        3. **Fetch** historical stock data
        4. **Analyze** with interactive charts and filters
        5. **Download** the complete dataset
        
        **Expected Format:**
        - Column A: Company Name
        - Column B: Stock Symbol
        """)
        
        st.header("⚙️ Settings")
        years_back = st.slider("Years of historical data", 1, 10, 5)
        
        st.header("ℹ️ Note")
        st.info("This app fetches data from Yahoo Finance for NSE stocks (Indian market).")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel file", 
        type=['xlsx', 'xls'],
        help="Upload your Excel file containing company names and stock symbols"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)
            
            # Display preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Stocks", len(df))
            with col2:
                st.metric("Years of Data", years_back)
            
            # Fetch data button
            if st.button("🚀 Fetch Stock Data", type="primary"):
                if len(df) > 0:
                    with st.spinner("Fetching historical stock data..."):
                        all_stock_data = fetch_stock_data(df, years_back)
                        st.session_state.stock_data = all_stock_data
                    
                    if all_stock_data:
                        st.success(f"✅ Successfully fetched data for {len(all_stock_data)} stocks!")
                        st.rerun()
                else:
                    st.error("❌ No data found in the uploaded file.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Display analysis if data is available
    if st.session_state.stock_data:
        all_stock_data = st.session_state.stock_data
        
        st.header("📊 Data Analysis & Visualization")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Individual Analysis", "🔍 Filtered Data", "📋 Summary", "💾 Download"])
        
        with tab1:
            st.subheader("Individual Stock Analysis")
            
            # Stock selection
            stock_symbols = list(all_stock_data.keys())
            selected_stock = st.selectbox("Select a stock for detailed analysis:", stock_symbols)
            
            if selected_stock:
                stock_data = all_stock_data[selected_stock].copy()
                
                # Date range filter
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=stock_data['Date'].min())
                with col2:
                    end_date = st.date_input("End Date", value=stock_data['Date'].max())
                
                # Filter data
                filtered_data = filter_data_by_date(stock_data, start_date, end_date)
                
                if not filtered_data.empty:
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{filtered_data['Close'].iloc[-1]:.2f}")
                    with col2:
                        daily_change = filtered_data['Close'].iloc[-1] - filtered_data['Close'].iloc[-2]
                        st.metric("Daily Change", f"₹{daily_change:.2f}")
                    with col3:
                        total_return = ((filtered_data['Close'].iloc[-1] / filtered_data['Close'].iloc[0]) - 1) * 100
                        st.metric("Total Return", f"{total_return:.2f}%")
                    with col4:
                        volatility = filtered_data['Close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Volatility", f"{volatility:.2f}%")
                    
                    # Charts
                    st.plotly_chart(create_price_chart(filtered_data, selected_stock), use_container_width=True)
                    st.plotly_chart(create_moving_average_chart(filtered_data, selected_stock), use_container_width=True)
                    st.plotly_chart(create_returns_chart(filtered_data, selected_stock), use_container_width=True)
                    st.plotly_chart(create_volatility_chart(filtered_data, selected_stock), use_container_width=True)
        
        with tab2:
            st.subheader("Multi-Stock Comparison")
            
            # Multi-select for stocks
            selected_stocks = st.multiselect(
                "Select stocks to compare:",
                stock_symbols,
                default=stock_symbols[:5] if len(stock_symbols) >= 5 else stock_symbols
            )
            
            if selected_stocks:
                # Performance comparison
                st.plotly_chart(create_performance_comparison(all_stock_data, selected_stocks), use_container_width=True)
                
                # Comparison table
                comparison_data = []
                for symbol in selected_stocks:
                    data = all_stock_data[symbol]
                    total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                    max_price = data['Close'].max()
                    min_price = data['Close'].min()
                    
                    comparison_data.append({
                        'Symbol': symbol,
                        'Current Price': f"₹{data['Close'].iloc[-1]:.2f}",
                        'Total Return (%)': f"{total_return:.2f}%",
                        'Volatility (%)': f"{volatility:.2f}%",
                        'Max Price': f"₹{max_price:.2f}",
                        'Min Price': f"₹{min_price:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(comparison_data))
        
        with tab3:
            st.subheader("Summary Statistics")
            
            # Create summary
            excel_data, summary_df = create_excel_download(all_stock_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Overall market stats
            all_returns = []
            for symbol, data in all_stock_data.items():
                returns = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                all_returns.append(returns)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Return", f"{np.mean(all_returns):.2f}%")
            with col2:
                st.metric("Best Performer", f"{max(all_returns):.2f}%")
            with col3:
                st.metric("Worst Performer", f"{min(all_returns):.2f}%")
        
        with tab4:
            st.subheader("Download Data")
            
            excel_data, summary_df = create_excel_download(all_stock_data)
            
            st.download_button(
                label="📥 Download Complete Excel File",
                data=excel_data,
                file_name="stock_historical_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.info("The Excel file contains:\n- Summary sheet with key metrics\n- Complete historical data for all stocks")
    
    else:
        st.info("👆 Please upload an Excel file and fetch data to start analysis!")
        
        # Show sample format
        st.subheader("📝 Sample Format")
        sample_data = {
            'Company Name': ['Adani Enterprises', 'Adani Ports & SEZ', 'Apollo Hospitals'],
            'Symbol': ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP']
        }
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()