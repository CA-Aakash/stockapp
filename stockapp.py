import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import io

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

# Main Streamlit App
def main():
    st.title("📈 Stock Price Data Fetcher")
    st.markdown("Upload your Excel file with stock symbols and get 5 years of historical data!")
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** your Excel file with stock symbols
        2. **Preview** the data to ensure correct format
        3. **Fetch** historical stock data (5 years)
        4. **Download** the complete dataset
        
        **Expected Format:**
        - Column A: Company Name
        - Column B: Stock Symbol
        """)
        
        st.header("⚙️ Settings")
        years_back = st.slider("Years of historical data", 1, 10, 5)
        
        st.header("ℹ️ Note")
        st.info("This app fetches data from Yahoo Finance for NSE stocks (Indian market). Processing time depends on the number of stocks.")
    
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
                    
                    if all_stock_data:
                        st.success(f"✅ Successfully fetched data for {len(all_stock_data)} stocks!")
                        
                        # Create Excel file for download
                        excel_data, summary_df = create_excel_download(all_stock_data)
                        
                        # Display summary
                        st.subheader("📈 Summary")
                        st.dataframe(summary_df)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Excel File",
                            data=excel_data,
                            file_name="stock_historical_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Display sample data
                        if len(all_stock_data) > 0:
                            st.subheader("📊 Sample Data")
                            sample_stock = list(all_stock_data.keys())[0]
                            sample_data = all_stock_data[sample_stock]
                            st.dataframe(sample_data.head())
                    else:
                        st.error("❌ No data could be fetched. Please check your stock symbols.")
                else:
                    st.error("❌ No data found in the uploaded file.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your file is in the correct format with Company Name in column A and Symbol in column B.")
    
    else:
        st.info("👆 Please upload an Excel file to get started!")
        
        # Show sample format
        st.subheader("📝 Sample Format")
        sample_data = {
            'Company Name': ['Adani Enterprises', 'Adani Ports & SEZ', 'Apollo Hospitals'],
            'Symbol': ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP']
        }
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()