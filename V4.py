import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 1. APP CONFIGURATION
# =========================================================
st.set_page_config(page_title="Wave Detection V5 | God Mode", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; border-left: 5px solid #00c853; }
    h1 { color: #0e1117; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. 100% DATA INGESTION & CLEANING
# =========================================================
@st.cache_data(ttl=600)
def process_files(uploaded_files):
    if not uploaded_files: return None
    
    master_df = pd.DataFrame()
    progress = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            
            if master_df.empty:
                master_df = df
            else:
                key = 'companyId' if 'companyId' in master_df.columns and 'companyId' in df.columns else 'Name'
                new_cols = [c for c in df.columns if c not in master_df.columns or c == key]
                master_df = pd.merge(master_df, df[new_cols], on=key, how='outer')
        except Exception:
            pass
        progress.progress((i + 1) / len(uploaded_files))
    
    # --- INTELLIGENT CLEANING ---
    non_numeric = ['companyId', 'Name', 'Industry', 'Sector', 'Fundamentals Source', 'Verdict']
    numeric_cols = [c for c in master_df.columns if c not in non_numeric]
    
    for col in numeric_cols:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
        # Fill logic: Growth/Returns = 0 (Neutral), Ratios = Median (Sector Avg)
        if any(x in col for x in ["Growth", "Change", "Returns", "Flow"]):
             master_df[col] = master_df[col].fillna(0)
        else:
             master_df[col] = master_df[col].fillna(master_df[col].median())
             
    progress.empty()
    return master_df

# =========================================================
# 3. REGIME DETECTION (THE BRAIN)
# =========================================================
def analyze_market_regime(df):
    m3 = df['Returns 3M'].median() if 'Returns 3M' in df.columns else 0
    m1y = df['Returns 1Y'].median() if 'Returns 1Y' in df.columns else 0
    
    if m3 > 8 and m1y > 15:
        # BULL: Aggressive on Growth & Momentum, but verified by Cash
        return "🚀 BULL RUN", {'Quality':0.15, 'Growth':0.25, 'Valuation':0.10, 'Safety':0.10, 'Momentum':0.25, 'Cash':0.15}
    elif m3 < -5:
        # BEAR: Safety, Valuation and Cash Flow are King
        return "🐻 BEAR MARKET", {'Quality':0.20, 'Growth':0.05, 'Valuation':0.25, 'Safety':0.20, 'Momentum':0.05, 'Cash':0.25}
    else:
        # SIDEWAYS: Balanced approach
        return "⚖️ SIDEWAYS", {'Quality':0.20, 'Growth':0.15, 'Valuation':0.15, 'Safety':0.15, 'Momentum':0.15, 'Cash':0.20}

# =========================================================
# 4. GOD MODE SCORING ENGINE (100% DATA USAGE)
# =========================================================
def run_god_mode_scoring(df, weights):
    scaler = MinMaxScaler()
    
    # 1. QUALITY (Efficiency)
    # Uses: ROE, ROCE, NPM, OPM
    q_cols = [c for c in ['ROE', 'ROCE', 'NPM', 'OPM'] if c in df.columns]
    if q_cols:
        df['Score_Quality'] = scaler.fit_transform(df[q_cols].mean(axis=1).values.reshape(-1,1))
    else: df['Score_Quality'] = 0

    # 2. GROWTH (Expansion + Acceleration)
    # Uses: TTM Growth + QoQ Growth (Acceleration)
    g_cols = [c for c in ['PAT Growth TTM', 'Revenue Growth TTM', 'PAT Growth QoQ', 'Revenue Growth QoQ'] if c in df.columns]
    if g_cols:
        df['Score_Growth'] = scaler.fit_transform(df[g_cols].mean(axis=1).values.reshape(-1,1))
    else: df['Score_Growth'] = 0

    # 3. VALUATION (Cheapness)
    if 'Price To Earnings' in df.columns:
        pe_inv = 1 / df['Price To Earnings'].clip(lower=0.1)
        ps_inv = 1 / df['Price To Sales'].clip(lower=0.1) if 'Price To Sales' in df.columns else 0
        df['Score_Valuation'] = scaler.fit_transform((pe_inv + ps_inv).values.reshape(-1,1))
    else: df['Score_Valuation'] = 0

    # 4. SAFETY (Balance Sheet Strength)
    if 'Debt To Equity' in df.columns:
        de_inv = 1 / (df['Debt To Equity'].clip(lower=0.01) + 1)
        prom = df.get('Promoter Holdings', 0) / 100
        df['Score_Safety'] = scaler.fit_transform((de_inv + prom).values.reshape(-1,1))
    else: df['Score_Safety'] = 0

    # 5. CASH FLOW KING (The "Truth" Indicator) - NEW!
    # Uses: Operating Cash Flow + Free Cash Flow
    c_cols = [c for c in ['Operating Cash Flow', 'Free Cash Flow'] if c in df.columns]
    if c_cols:
        df['Score_Cash'] = scaler.fit_transform(df[c_cols].mean(axis=1).values.reshape(-1,1))
    else: df['Score_Cash'] = 0.5

    # 6. POWER MOMENTUM (Technicals + Institutional Flow) - UPGRADED!
    # Uses: RSI Daily + Weekly + ADX + FII + DII
    rsi_w = df.get('RSI 14W', 50)
    rsi_d = df.get('RSI 14D', 50) 
    adx = df.get('ADX 14W', 20)
    fii = df.get('Change In FII Holdings Latest Quarter', 0)
    dii = df.get('Change In DII Holdings Latest Quarter', 0) 
    
    # Momentum Formula: 
    mom_score = (rsi_w * 0.25) + (rsi_d * 0.1) + (adx * 0.2) + ((fii + dii) * 15)
    df['Score_Momentum'] = scaler.fit_transform(mom_score.values.reshape(-1,1))

    # FINAL FORMULA
    df['Final_Score'] = (
        df['Score_Quality'] * weights['Quality'] +
        df['Score_Growth'] * weights['Growth'] +
        df['Score_Valuation'] * weights['Valuation'] +
        df['Score_Safety'] * weights['Safety'] +
        df['Score_Momentum'] * weights['Momentum'] +
        df['Score_Cash'] * weights['Cash']
    )
    return df

# =========================================================
# 5. UI & REPORTING
# =========================================================
def main():
    st.title("⚡ Wave Detection V5 | God Mode Analytics")
    
    with st.sidebar:
        st.header("📂 Upload Data")
        files = st.file_uploader("Drop CSVs Here", accept_multiple_files=True, type=['csv'])

    if files:
        df = process_files(files)
        if df is not None:
            regime, w = analyze_market_regime(df)
            
            # Header Metrics
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-card'><h4>Market Regime</h4><h3>{regime}</h3></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><h4>Stocks Analyzed</h4><h3>{len(df)}</h3></div>", unsafe_allow_html=True)
            
            # Run Engine
            df = run_god_mode_scoring(df, w)
            
            # Generate Verdicts
            def get_verdict(row):
                s = row['Final_Score']
                fcf = row.get('Free Cash Flow', 0)
                dii = row.get('Change In DII Holdings Latest Quarter', 0)
                
                if s > 0.85: return "💎 PERFECT 10"
                if s > 0.75 and fcf > 0: return "💰 CASH MACHINE"
                if dii > 1 and row.get('Change In FII Holdings Latest Quarter', 0) > 1: return "🏦 BIG MONEY BUY"
                if row.get('RSI 14W', 0) > 65: return "🚀 MOMENTUM"
                if s < 0.3: return "❌ WEAK"
                return "⚠️ WATCH"
                
            df['Verdict'] = df.apply(get_verdict, axis=1)
            df = df.sort_values(by='Final_Score', ascending=False)
            df.insert(0, 'Rank', range(1, len(df)+1))

            # Display
            st.subheader("🏆 The Ultimate Rankings")
            
            # Select columns dynamically (only if they exist)
            show_cols = ['Rank','Name','Verdict','Final_Score','Price To Earnings','PAT Growth QoQ','Free Cash Flow','RSI 14W']
            show_cols = [c for c in show_cols if c in df.columns]
            
            st.dataframe(df[show_cols].head(50).style.background_gradient(subset=['Final_Score'], cmap='Greens'), height=800)
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download God Mode Report", csv, "God_Mode_Analysis.csv", "text/csv")

if __name__ == "__main__":
    main()
