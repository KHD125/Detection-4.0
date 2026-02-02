import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import io

# =========================================================
# 1. APP CONFIGURATION & STYLING
# =========================================================
st.set_page_config(
    page_title="Wave Detection 4.0 | Pro Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "Hedge Fund" look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stDataFrame { border-radius: 10px; }
    h1 { color: #0e1117; }
    h3 { color: #262730; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. CORE LOGIC: DATA MERGING & CLEANING
# =========================================================
@st.cache_data(ttl=600)  # Cache results to make it fast
def process_files(uploaded_files):
    if not uploaded_files:
        return None
    
    master_df = pd.DataFrame()
    
    # PROGRESS BAR
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip() # Clean headers
            
            if master_df.empty:
                master_df = df
            else:
                # Intelligent Merge Key Detection
                key = 'companyId' if 'companyId' in master_df.columns and 'companyId' in df.columns else 'Name'
                
                # Merge only new columns to preserve data integrity
                new_cols = [c for c in df.columns if c not in master_df.columns or c == key]
                master_df = pd.merge(master_df, df[new_cols], on=key, how='outer')
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    status_text.text("✅ Merging Complete. Cleaning Data...")
    
    # DATA CLEANING
    # List of non-numeric metadata columns to ignore
    non_numeric = ['companyId', 'Name', 'Industry', 'Sector', 'Fundamentals Source', 'Verdict']
    numeric_cols = [c for c in master_df.columns if c not in non_numeric]
    
    for col in numeric_cols:
        # Force numeric, turn errors to NaN
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
        
        # Fill NaNs smartly
        if "Growth" in col or "Change" in col or "Returns" in col:
             master_df[col] = master_df[col].fillna(0) # Neutral for growth
        else:
             master_df[col] = master_df[col].fillna(master_df[col].median()) # Median for ratios
             
    progress_bar.empty()
    return master_df

# =========================================================
# 3. THE BRAIN: REGIME DETECTION & SCORING
# =========================================================
def analyze_market_regime(df):
    """Detects if we are in a Bull, Bear, or Sideways market based on median returns."""
    median_3m = df['Returns 3M'].median() if 'Returns 3M' in df.columns else 0
    median_1y = df['Returns 1Y'].median() if 'Returns 1Y' in df.columns else 0
    
    if median_3m > 8 and median_1y > 15:
        return "🚀 BULL RUN", {'Quality': 0.20, 'Growth': 0.30, 'Valuation': 0.10, 'Safety': 0.10, 'Momentum': 0.30}
    elif median_3m < -5:
        return "🐻 BEAR / CORRECTION", {'Quality': 0.25, 'Growth': 0.10, 'Valuation': 0.30, 'Safety': 0.25, 'Momentum': 0.10}
    else:
        return "⚖️ SIDEWAYS / CHOPPY", {'Quality': 0.25, 'Growth': 0.20, 'Valuation': 0.20, 'Safety': 0.20, 'Momentum': 0.15}

def run_scoring_engine(df, weights):
    scaler = MinMaxScaler()
    
    # 1. QUALITY SCORE (Profitability)
    qual_cols = ['ROE', 'ROCE', 'NPM', 'OPM']
    valid_qual = [c for c in qual_cols if c in df.columns]
    if valid_qual:
        df['Score_Quality'] = scaler.fit_transform(df[valid_qual].mean(axis=1).values.reshape(-1,1))
    else:
        df['Score_Quality'] = 0

    # 2. GROWTH SCORE (Expansion)
    growth_cols = ['PAT Growth TTM', 'Revenue Growth TTM', 'Returns 3Y']
    valid_growth = [c for c in growth_cols if c in df.columns]
    if valid_growth:
        df['Score_Growth'] = scaler.fit_transform(df[valid_growth].mean(axis=1).values.reshape(-1,1))
    else:
        df['Score_Growth'] = 0

    # 3. VALUATION SCORE (Cheapness - Lower is better)
    if 'Price To Earnings' in df.columns:
        pe_inv = 1 / df['Price To Earnings'].clip(lower=0.1)
        df['Score_Valuation'] = scaler.fit_transform(pe_inv.values.reshape(-1,1))
    else:
        df['Score_Valuation'] = 0

    # 4. SAFETY SCORE (Resilience)
    if 'Debt To Equity' in df.columns:
        de_inv = 1 / (df['Debt To Equity'].clip(lower=0.01) + 1)
        promoter = df.get('Promoter Holdings', 0) / 100
        df['Score_Safety'] = scaler.fit_transform((de_inv + promoter).values.reshape(-1,1))
    else:
        df['Score_Safety'] = 0

    # 5. MOMENTUM SCORE (Trend Strength)
    rsi = df.get('RSI 14W', pd.Series([50]*len(df)))
    adx = df.get('ADX 14W', pd.Series([20]*len(df)))
    fii = df.get('Change In FII Holdings Latest Quarter', 0)
    
    raw_momentum = (rsi * 0.4) + (adx * 0.3) + (fii * 10)
    df['Score_Momentum'] = scaler.fit_transform(raw_momentum.values.reshape(-1,1))

    # FINAL WEIGHTED SCORE
    df['Final_Score'] = (
        df['Score_Quality'] * weights['Quality'] +
        df['Score_Growth'] * weights['Growth'] +
        df['Score_Valuation'] * weights['Valuation'] +
        df['Score_Safety'] * weights['Safety'] +
        df['Score_Momentum'] * weights['Momentum']
    )
    
    return df

def get_smart_verdict(row):
    """Assigns a 'Human Readable' tag to the stock."""
    score = row['Final_Score']
    pe = row.get('Price To Earnings', 99)
    growth = row.get('PAT Growth TTM', 0)
    adx = row.get('ADX 14W', 0)
    
    if score > 0.85: return "💎 STRONG BUY (Top Pick)"
    if score > 0.70 and growth > 30: return "🚀 HIGH GROWTH"
    if score > 0.65 and pe < 15: return "💰 VALUE BUY"
    if adx > 35 and row.get('RSI 14W', 50) > 60: return "⚡ MOMENTUM ROCKET"
    if score < 0.3: return "❌ SELL / AVOID"
    return "👀 WATCHLIST"

# =========================================================
# 4. MAIN UI LAYOUT
# =========================================================
def main():
    st.title("🌊 Wave Detection 4.0 | Financial Intelligence")
    st.markdown("### Upload your raw CSVs below to activate the algorithm.")
    
    # SIDEBAR
    with st.sidebar:
        st.header("📂 Data Import")
        uploaded_files = st.file_uploader("Select CSV Files", accept_multiple_files=True, type=['csv'])
        st.info("💡 Tip: Upload multiple sector files at once for a full market scan.")

    if uploaded_files:
        # PROCESS
        df = process_files(uploaded_files)
        
        if df is not None:
            # ANALYZE REGIME
            regime, weights = analyze_market_regime(df)
            
            # DASHBOARD HEADER
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-card'><h4>📡 Market Regime</h4><h2>{regime}</h2></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><h4>📊 Total Stocks</h4><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
            with c3:
                top_stock = df.sort_values(by='Returns 1Y', ascending=False).iloc[0]['Name']
                st.markdown(f"<div class='metric-card'><h4>🚀 Top Performer (1Y)</h4><h2>{top_stock}</h2></div>", unsafe_allow_html=True)

            # RUN ALGORITHM
            df = run_scoring_engine(df, weights)
            df['Verdict'] = df.apply(get_smart_verdict, axis=1)
            
            # SORT & RANK
            df = df.sort_values(by='Final_Score', ascending=False)
            df.insert(0, 'Rank', range(1, len(df) + 1))

            # --- TABS FOR DIFFERENT VIEWS ---
            tab1, tab2, tab3 = st.tabs(["🏆 Top Rankings", "📈 Visual Analysis", "🔍 Deep Dive Data"])
            
            with tab1:
                st.subheader(f"Top Picks for {regime} Environment")
                
                # Show key columns
                display_cols = ['Rank', 'Name', 'Verdict', 'Final_Score', 'Price To Earnings', 'ROE', 'PAT Growth TTM', 'ADX 14W']
                # Filter to ensure columns exist
                display_cols = [c for c in display_cols if c in df.columns]
                
                # Highlight logic
                st.dataframe(
                    df[display_cols].head(20).style.background_gradient(subset=['Final_Score'], cmap='Greens'),
                    use_container_width=True,
                    height=800
                )
                
            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Growth vs. Valuation (Hunt for Hidden Gems)")
                    if 'PAT Growth TTM' in df.columns and 'Price To Earnings' in df.columns:
                        fig = px.scatter(
                            df, x='Price To Earnings', y='PAT Growth TTM', 
                            color='Verdict', hover_name='Name', size='Final_Score',
                            log_x=True, title="PE Ratio vs Growth (Top Left is Best)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.subheader("Momentum Strength (RSI vs ADX)")
                    if 'RSI 14W' in df.columns and 'ADX 14W' in df.columns:
                        fig = px.scatter(
                            df, x='RSI 14W', y='ADX 14W',
                            color='Final_Score', hover_name='Name',
                            title="Trend Strength (Top Right is Strongest)"
                        )
                        # Add 'Safe Zone' box
                        fig.add_shape(type="rect", x0=60, y0=25, x1=100, y1=100,
                            line=dict(color="Green"), fillcolor="Green", opacity=0.1)
                        st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Full Dataset Export")
                st.dataframe(df)
                
                # CSV Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Analysis as CSV",
                    data=csv,
                    file_name=f"Wave_Detection_Analysis_{regime.split()[1]}.csv",
                    mime='text/csv',
                )

if __name__ == "__main__":
    main()
