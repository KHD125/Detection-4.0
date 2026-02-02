import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 1. APP CONFIGURATION & STYLING
# =========================================================
st.set_page_config(
    page_title="Wave Detection Final | Pro Stock Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional "Hedge Fund" UI Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #2962ff;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stDataFrame { border-radius: 10px; }
    h1, h2, h3 { color: #0e1117; font-family: 'Helvetica', sans-serif; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. INTELLIGENT DATA PROCESSING ENGINE
# =========================================================
@st.cache_data(ttl=600) 
def process_files(uploaded_files):
    if not uploaded_files: return None
    
    master_df = pd.DataFrame()
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip() # Remove spaces
            
            if master_df.empty:
                master_df = df
            else:
                # Auto-Detect Merge Key
                key = 'companyId' if 'companyId' in master_df.columns and 'companyId' in df.columns else 'Name'
                
                # Merge only new columns (Prevent Duplicates)
                new_cols = [c for c in df.columns if c not in master_df.columns or c == key]
                master_df = pd.merge(master_df, df[new_cols], on=key, how='outer')
        except Exception as e:
            st.error(f"Skipped {file.name}: {e}")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    # --- DATA CLEANING ---
    non_numeric = ['companyId', 'Name', 'Industry', 'Sector', 'Fundamentals Source', 'Verdict']
    numeric_cols = [c for c in master_df.columns if c not in non_numeric]
    
    for col in numeric_cols:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
        
        # Smart Fill: 0 for Growth/Returns, Median for Ratios
        if any(x in col for x in ["Growth", "Change", "Returns", "Flow"]):
             master_df[col] = master_df[col].fillna(0)
        else:
             master_df[col] = master_df[col].fillna(master_df[col].median())
             
    progress_bar.empty()
    return master_df

# =========================================================
# 3. THE BRAIN: REGIME DETECTION
# =========================================================
def analyze_market_regime(df):
    """
    Scans the uploaded list to determine if the market is Bull, Bear, or Sideways.
    Adjusts the strategy weights automatically.
    """
    m3 = df['Returns 3M'].median() if 'Returns 3M' in df.columns else 0
    m1y = df['Returns 1Y'].median() if 'Returns 1Y' in df.columns else 0
    
    if m3 > 8 and m1y > 15:
        # BULL MODE: Aggressive (High Weight on Momentum & Growth)
        return "🚀 BULL RUN", {'Quality': 0.20, 'Growth': 0.30, 'Valuation': 0.10, 'Safety': 0.10, 'Momentum': 0.30}
    elif m3 < -5:
        # BEAR MODE: Defensive (High Weight on Valuation & Safety)
        return "🐻 BEAR MARKET", {'Quality': 0.25, 'Growth': 0.05, 'Valuation': 0.30, 'Safety': 0.30, 'Momentum': 0.10}
    else:
        # CHOPPY MODE: Balanced
        return "⚖️ SIDEWAYS", {'Quality': 0.25, 'Growth': 0.20, 'Valuation': 0.20, 'Safety': 0.20, 'Momentum': 0.15}

# =========================================================
# 4. SCORING ENGINE (V4 LOGIC)
# =========================================================
def run_scoring_engine(df, weights):
    scaler = MinMaxScaler()
    
    # A. QUALITY (Profitability)
    q_cols = [c for c in ['ROE', 'ROCE', 'NPM', 'OPM'] if c in df.columns]
    if q_cols:
        df['Score_Quality'] = scaler.fit_transform(df[q_cols].mean(axis=1).values.reshape(-1,1))
    else: df['Score_Quality'] = 0

    # B. GROWTH (Expansion)
    g_cols = [c for c in ['PAT Growth TTM', 'Revenue Growth TTM', 'Returns 3Y'] if c in df.columns]
    if g_cols:
        df['Score_Growth'] = scaler.fit_transform(df[g_cols].mean(axis=1).values.reshape(-1,1))
    else: df['Score_Growth'] = 0

    # C. VALUATION (Value - Inverted)
    if 'Price To Earnings' in df.columns:
        pe_inv = 1 / df['Price To Earnings'].clip(lower=0.1)
        df['Score_Valuation'] = scaler.fit_transform(pe_inv.values.reshape(-1,1))
    else: df['Score_Valuation'] = 0

    # D. SAFETY (Risk - Inverted)
    if 'Debt To Equity' in df.columns:
        de_inv = 1 / (df['Debt To Equity'].clip(lower=0.01) + 1)
        prom = df.get('Promoter Holdings', 0) / 100
        df['Score_Safety'] = scaler.fit_transform((de_inv + prom).values.reshape(-1,1))
    else: df['Score_Safety'] = 0

    # E. MOMENTUM (Trend Strength)
    rsi = df.get('RSI 14W', pd.Series([50]*len(df)))
    adx = df.get('ADX 14W', pd.Series([20]*len(df)))
    fii = df.get('Change In FII Holdings Latest Quarter', 0)
    
    # Power Trend Formula
    raw_mom = (rsi * 0.4) + (adx * 0.3) + (fii * 10)
    df['Score_Momentum'] = scaler.fit_transform(raw_mom.values.reshape(-1,1))

    # FINAL WEIGHTED CALCULATION
    df['Final_Score'] = (
        df['Score_Quality'] * weights['Quality'] +
        df['Score_Growth'] * weights['Growth'] +
        df['Score_Valuation'] * weights['Valuation'] +
        df['Score_Safety'] * weights['Safety'] +
        df['Score_Momentum'] * weights['Momentum']
    )
    return df

# =========================================================
# 5. SMART VERDICT & TRAP DETECTION
# =========================================================
def get_smart_verdict(row):
    """
    Generates the final Buy/Sell signal.
    INCLUDES SAFETY CHECK: Flags high-scoring stocks with negative cash flow.
    """
    score = row['Final_Score']
    pe = row.get('Price To Earnings', 99)
    growth = row.get('PAT Growth TTM', 0)
    adx = row.get('ADX 14W', 0)
    
    # --- 🛡️ THE TRAP CHECK ---
    # If a stock has a high score (>0.7) but Negative Free Cash Flow, flag it.
    fcf = row.get('Free Cash Flow', 1) 
    if fcf < 0 and score > 0.7:
        return "⚠️ TRAP (Neg Cash)"

    # --- STANDARD RANKING ---
    if score > 0.82: return "💎 STRONG BUY"
    if score > 0.70 and growth > 30: return "🚀 HIGH GROWTH"
    if score > 0.65 and pe < 15: return "💰 VALUE BUY"
    if adx > 30 and row.get('RSI 14W', 50) > 60: return "⚡ MOMENTUM PLAY"
    if score < 0.35: return "❌ AVOID"
    return "👀 WATCHLIST"

# =========================================================
# 6. MAIN DASHBOARD UI
# =========================================================
def main():
    st.title("🌊 Wave Detection Final | Institutional Grade Analysis")
    st.markdown("### ⚡ AI-Powered Stock Ranking System")
    
    with st.sidebar:
        st.header("📂 Data Center")
        uploaded_files = st.file_uploader("Upload Sector CSVs", accept_multiple_files=True, type=['csv'])
        st.info("💡 Pro Tip: Drag & Drop all your watchlist files here at once.")

    if uploaded_files:
        # 1. Process
        df = process_files(uploaded_files)
        
        if df is not None:
            # 2. Analyze Regime
            regime, weights = analyze_market_regime(df)
            
            # 3. Dashboard Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-card'><h4>📡 Market Regime</h4><h3>{regime}</h3></div>", unsafe_allow_html=True)
            with col2:
                best_stock = df.sort_values(by='Returns 1Y', ascending=False).iloc[0]['Name']
                st.markdown(f"<div class='metric-card'><h4>🚀 1Y Leader</h4><h3>{best_stock}</h3></div>", unsafe_allow_html=True)
            with col3:
                 st.markdown(f"<div class='metric-card'><h4>📊 Stocks Analyzed</h4><h3>{len(df)}</h3></div>", unsafe_allow_html=True)

            # 4. Run Core Engine
            df = run_scoring_engine(df, weights)
            df['Verdict'] = df.apply(get_smart_verdict, axis=1)
            
            # Sort & Rank
            df = df.sort_values(by='Final_Score', ascending=False)
            df.insert(0, 'Rank', range(1, len(df) + 1))

            # 5. Visuals & Tables
            tab1, tab2, tab3 = st.tabs(["🏆 Top Ranked Stocks", "📈 Strategy Map", "📥 Export Data"])
            
            with tab1:
                st.subheader("Top Picks (Adjusted for Market Regime)")
                
                # Dynamic Column Selection (Only show what exists)
                target_cols = ['Rank', 'Name', 'Verdict', 'Final_Score', 'Close Price', 'Price To Earnings', 'ROE', 'PAT Growth TTM', 'RSI 14W', 'Free Cash Flow']
                final_cols = [c for c in target_cols if c in df.columns]
                
                # Highlight "Strong Buy" in Green and "Trap" in Red
                def highlight_verdict(val):
                    if 'STRONG BUY' in val: return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    if 'TRAP' in val: return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                    if 'AVOID' in val: return 'color: red'
                    return ''

                st.dataframe(
                    df[final_cols].head(30).style.applymap(highlight_verdict, subset=['Verdict']),
                    height=800,
                    use_container_width=True
                )
                
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    if 'Price To Earnings' in df.columns and 'PAT Growth TTM' in df.columns:
                        fig = px.scatter(df, x='Price To Earnings', y='PAT Growth TTM', color='Verdict', 
                                       hover_name='Name', size='Final_Score', log_x=True, 
                                       title="Growth at a Reasonable Price (GARP)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                with col2:
                    if 'RSI 14W' in df.columns and 'ADX 14W' in df.columns:
                        fig = px.scatter(df, x='RSI 14W', y='ADX 14W', color='Final_Score', 
                                       hover_name='Name', title="Momentum Strength Matrix")
                        fig.add_shape(type="rect", x0=60, y0=25, x1=100, y1=100, line=dict(color="Green"), opacity=0.1)
                        st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Download Full Analysis")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "Final_Wave_Analysis.csv", "text/csv")

if __name__ == "__main__":
    main()
