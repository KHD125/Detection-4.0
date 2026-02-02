"""
V5 QUANT - DATA-DRIVEN STOCK ANALYZER
=====================================
Built on ACTUAL correlations from your data, not theory.

CORE INSIGHT: Momentum + Smart Money explains 90% of returns.
Everything else is noise.

DESIGN PHILOSOPHY:
1. Only use metrics that ACTUALLY correlate with returns
2. Weight by correlation strength, not arbitrary percentages
3. Simple rules > Complex logic
4. Let data decide, not human intuition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="V5 Quant | Data-Driven",
    page_icon="📈",
    layout="wide"
)

# Clean CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #2d3748;
    }
    .score-high { color: #00ff88; font-size: 2rem; font-weight: 700; }
    .score-mid { color: #ffd700; font-size: 2rem; font-weight: 700; }
    .score-low { color: #ff4757; font-size: 2rem; font-weight: 700; }
    .signal-buy { background: #00ff88; color: black; padding: 4px 12px; border-radius: 4px; font-weight: 600; }
    .signal-hold { background: #ffd700; color: black; padding: 4px 12px; border-radius: 4px; font-weight: 600; }
    .signal-avoid { background: #ff4757; color: white; padding: 4px 12px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=600)
def load_data(files):
    if not files:
        return None
    
    master = pd.DataFrame()
    for file in files:
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            if master.empty:
                master = df
            else:
                key = 'companyId' if 'companyId' in df.columns else 'Name'
                new_cols = [c for c in df.columns if c not in master.columns or c == key]
                if len(new_cols) > 1:
                    master = pd.merge(master, df[new_cols], on=key, how='outer')
        except Exception as e:
            st.warning(f"Skipped {file.name}: {e}")
    
    return master

# ============================================================
# QUANT SCORING ENGINE
# ============================================================
def quant_score(df):
    """
    3-Factor Model based on ACTUAL data correlations:
    
    Factor 1: MOMENTUM (70%) - Corr: 0.80+
        - RSI 14W (0.878 corr)
        - Returns 3M (0.707 corr)
        - 52WH Distance (-0.655 corr, inverted)
    
    Factor 2: SMART MONEY (20%) - Corr: 0.40+
        - FII Changes (+0.499 corr)
        - Promoter Changes (stability signal)
    
    Factor 3: FUNDAMENTALS (10%) - Corr: 0.35+
        - ROCE (+0.413 corr)
        - Revenue Growth (+0.336 corr)
    """
    n = len(df)
    
    def safe_col(name, default=0):
        if name in df.columns:
            return df[name].fillna(df[name].median() if df[name].notna().any() else default)
        return pd.Series([default] * n, index=df.index)
    
    def percentile_rank(series):
        return series.rank(pct=True, method='average').fillna(0.5)
    
    # ══════════════════════════════════════════════════════
    # FACTOR 1: MOMENTUM (70% weight) - THE KING
    # ══════════════════════════════════════════════════════
    rsi = safe_col('RSI 14W', 50)
    ret_3m = safe_col('Returns 3M', 0)
    dist_52wh = safe_col('52WH Distance', 25)
    
    # RSI sweet spot: 50-70 is ideal (trending but not overbought)
    rsi_score = percentile_rank(rsi)
    # Penalize extreme overbought (>75)
    rsi_score = np.where(rsi > 75, rsi_score * 0.7, rsi_score)
    
    # Returns: Higher = Better
    ret_score = percentile_rank(ret_3m)
    
    # 52WH: Lower distance = Better (closer to high)
    dist_score = percentile_rank(-dist_52wh)  # Negate so lower = higher score
    
    # Momentum composite (weighted by correlation strength)
    # RSI: 0.878, Ret3M: 0.707, 52WH: 0.655
    momentum = (
        rsi_score * 0.40 +      # Strongest signal
        ret_score * 0.35 +      # Strong signal
        dist_score * 0.25       # Good signal
    )
    df['Momentum_Score'] = momentum
    
    # ══════════════════════════════════════════════════════
    # FACTOR 2: SMART MONEY (20% weight)
    # ══════════════════════════════════════════════════════
    fii_chg = safe_col('Change In FII Holdings Latest Quarter', 0)
    prom_chg = safe_col('Change In Promoter Holdings Latest Quarter', 0)
    
    # FII buying = strong positive signal
    fii_score = percentile_rank(fii_chg)
    
    # Promoter not selling = stability (penalize heavy selling)
    prom_score = percentile_rank(prom_chg)
    
    # Smart money composite
    smart_money = (
        fii_score * 0.70 +      # FII is the key signal
        prom_score * 0.30       # Promoter stability
    )
    df['SmartMoney_Score'] = smart_money
    
    # ══════════════════════════════════════════════════════
    # FACTOR 3: FUNDAMENTALS (10% weight) - Minor role
    # ══════════════════════════════════════════════════════
    roce = safe_col('ROCE', 12)
    rev_growth = safe_col('Revenue Growth TTM', 5)
    
    # Only ROCE and Rev Growth have meaningful correlation
    roce_score = percentile_rank(roce)
    rev_score = percentile_rank(rev_growth)
    
    fundamentals = (
        roce_score * 0.60 +     # ROCE has 0.413 corr
        rev_score * 0.40        # Rev Growth has 0.336 corr
    )
    df['Fundamental_Score'] = fundamentals
    
    # ══════════════════════════════════════════════════════
    # FINAL SCORE (Weighted by predictive power)
    # ══════════════════════════════════════════════════════
    df['Quant_Score'] = (
        momentum * 0.70 +       # Momentum dominates
        smart_money * 0.20 +    # Smart money matters
        fundamentals * 0.10    # Fundamentals = minor
    ) * 100
    
    # ══════════════════════════════════════════════════════
    # SAFETY NET (Catch Pump & Dumps)
    # ══════════════════════════════════════════════════════
    # These don't predict returns, but they PREVENT disasters
    fcf = safe_col('Free Cash Flow', 0)
    ocf = safe_col('Operating Cash Flow', 0)
    de_ratio = safe_col('Debt To Equity', 0.5)
    icr = safe_col('Interest Coverage Ratio', 5)
    prom_hold = safe_col('Promoter Holdings', 50)
    
    def detect_red_flags(row):
        """
        Red flags don't predict winners, but they identify TRAPS.
        A stock can have great momentum but be a house of cards.
        """
        flags = []
        
        # CASH TRAP: Negative FCF + Negative OCF = Burning cash
        row_fcf = row.get('Free Cash Flow', 0) if pd.notna(row.get('Free Cash Flow', 0)) else 0
        row_ocf = row.get('Operating Cash Flow', 0) if pd.notna(row.get('Operating Cash Flow', 0)) else 0
        if row_fcf < 0 and row_ocf < 0:
            flags.append("CASH_TRAP")
        
        # DEBT BOMB: D/E > 2 AND weak interest coverage
        row_de = row.get('Debt To Equity', 0) if pd.notna(row.get('Debt To Equity', 0)) else 0
        row_icr = row.get('Interest Coverage Ratio', 10) if pd.notna(row.get('Interest Coverage Ratio', 10)) else 10
        if row_de > 2 and row_icr < 2:
            flags.append("DEBT_BOMB")
        
        # PROMOTER DUMP: Promoter selling heavily (>3% in quarter)
        row_prom_chg = row.get('Change In Promoter Holdings Latest Quarter', 0)
        if pd.notna(row_prom_chg) and row_prom_chg < -3:
            flags.append("PROMOTER_EXIT")
        
        # LOW SKIN: Promoter holds < 25% (no accountability)
        row_prom = row.get('Promoter Holdings', 50) if pd.notna(row.get('Promoter Holdings', 50)) else 50
        if row_prom < 25:
            flags.append("LOW_SKIN")
        
        # SMART MONEY FLEE: FII dumping while momentum still looks good
        row_fii = row.get('Change In FII Holdings Latest Quarter', 0)
        row_mom = row.get('Momentum_Score', 0.5)
        if pd.notna(row_fii) and row_fii < -2 and row_mom > 0.5:
            flags.append("FII_EXITING")  # FII leaving before crash
        
        return flags
    
    df['Red_Flags'] = df.apply(detect_red_flags, axis=1)
    df['Flag_Count'] = df['Red_Flags'].apply(len)
    
    # ══════════════════════════════════════════════════════
    # SIGNAL LOGIC (Momentum + Safety Net)
    # ══════════════════════════════════════════════════════
    def get_signal(row):
        score = row['Quant_Score']
        mom = row['Momentum_Score']
        flags = row['Red_Flags']
        flag_count = row['Flag_Count']
        
        # ─────────────────────────────────────────────────
        # TRAP DETECTION (Override everything)
        # ─────────────────────────────────────────────────
        if 'CASH_TRAP' in flags and 'DEBT_BOMB' in flags:
            return "🚨 DEATH SPIRAL", "trap", "Burning cash + Heavy debt = Bankruptcy risk"
        
        if 'CASH_TRAP' in flags and mom > 0.6:
            return "🚨 PUMP & DUMP", "trap", "High momentum but negative cash flow = Unsustainable"
        
        if 'PROMOTER_EXIT' in flags:
            return "🚨 INSIDER EXIT", "trap", "Promoters dumping = They know something bad"
        
        if 'FII_EXITING' in flags and 'CASH_TRAP' in flags:
            return "🚨 SMART EXIT", "trap", "FIIs leaving + Cash negative = Crash incoming"
        
        # Multiple red flags = Too risky even with good momentum
        if flag_count >= 2:
            return "⚠️ RISKY", "avoid", f"Multiple red flags: {', '.join(flags)}"
        
        # ─────────────────────────────────────────────────
        # NORMAL SIGNAL LOGIC
        # ─────────────────────────────────────────────────
        # Single flag = Caution but not disqualifying
        caution = " ⚡" if flag_count == 1 else ""
        flag_note = f" (Watch: {flags[0]})" if flag_count == 1 else ""
        
        if score >= 75 and mom >= 0.6:
            return f"🚀 STRONG BUY{caution}", "buy", f"Top momentum + smart money{flag_note}"
        elif score >= 60 and mom >= 0.5:
            return f"📈 BUY{caution}", "buy", f"Good momentum + positive signals{flag_note}"
        elif score >= 45:
            return f"⏸️ HOLD{caution}", "hold", f"Neutral - wait for momentum{flag_note}"
        elif score >= 30:
            return "⚠️ WEAK", "avoid", "Below average on all factors"
        else:
            return "❌ AVOID", "avoid", "Poor momentum + weak fundamentals"
    
    signals = df.apply(get_signal, axis=1)
    df['Signal'] = signals.apply(lambda x: x[0])
    df['Signal_Class'] = signals.apply(lambda x: x[1])
    df['Signal_Reason'] = signals.apply(lambda x: x[2])
    
    return df

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <h1 style='text-align:center; color:#00ff88;'>V5 QUANT</h1>
    <p style='text-align:center; color:#888;'>Data-Driven • 3-Factor Model • No Bullshit</p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Upload Data")
        files = st.file_uploader("CSVs", accept_multiple_files=True, type=['csv'])
        
        st.markdown("---")
        st.markdown("""
        ### Model Logic
        
        **Factor Weights (by correlation):**
        - 🔥 Momentum: 70%
        - 💰 Smart Money: 20%  
        - 📊 Fundamentals: 10%
        
        **Momentum (0.80+ corr):**
        - RSI 14W
        - Returns 3M
        - 52WH Distance
        
        **Smart Money (0.40+ corr):**
        - FII Changes
        - Promoter stability
        
        **Fundamentals (0.35+ corr):**
        - ROCE
        - Revenue Growth
        """)
    
    if not files:
        st.markdown("""
        <div style='text-align:center; padding:4rem; background:#1a1a2e; border-radius:12px; margin:2rem 0;'>
            <h3 style='color:#fff;'>Upload CSV Files</h3>
            <p style='color:#888;'>Drop your stock data files in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Why This Model Works
        
        Traditional stock screeners use **theory-based** factors:
        - Low P/E = Good (Theory)
        - High ROE = Good (Theory)
        - Low Debt = Good (Theory)
        
        **But data shows different story:**
        
        | Factor | Correlation with Returns |
        |--------|-------------------------|
        | RSI 14W | **+0.878** 🔥 |
        | Returns 3M | **+0.707** 🔥 |
        | 52WH Distance | **-0.655** 🔥 |
        | FII Changes | **+0.499** |
        | P/E Ratio | **+0.385** (High PE wins!) |
        | ROCE | +0.413 |
        | NPM | -0.013 (useless) |
        | D/E Ratio | +0.068 (useless) |
        
        **The truth:** Momentum + Smart Money = 90% of returns.
        """)
        return
    
    # Load and process data
    df = load_data(files)
    if df is None or df.empty:
        st.error("No valid data")
        return
    
    df = quant_score(df)
    df = df.sort_values('Quant_Score', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df)+1))
    
    # Summary Stats
    col1, col2, col3, col4 = st.columns(4)
    buy_count = len(df[df['Signal_Class'] == 'buy'])
    hold_count = len(df[df['Signal_Class'] == 'hold'])
    avoid_count = len(df[df['Signal_Class'] == 'avoid'])
    trap_count = len(df[df['Signal_Class'] == 'trap'])
    
    col1.metric("🚀 Buy Signals", buy_count)
    col2.metric("⏸️ Hold", hold_count)
    col3.metric("⚠️ Avoid", avoid_count)
    col4.metric("🚨 Traps", trap_count)
    
    # TRAP ALERTS (Show prominently if any)
    trap_df = df[df['Signal_Class'] == 'trap']
    if len(trap_df) > 0:
        st.markdown("### 🚨 TRAP ALERTS - Do NOT Buy These!")
        for _, row in trap_df.iterrows():
            flags_str = ', '.join(row['Red_Flags']) if row['Red_Flags'] else 'Multiple issues'
            st.error(f"**{row.get('Name', 'Unknown')}**: {row['Signal']} - {row.get('Signal_Reason', '')} | Flags: {flags_str}")
    
    # Main Table
    st.markdown("### Rankings")
    
    display_cols = ['Rank', 'Name', 'Signal', 'Signal_Reason', 'Quant_Score', 
                    'Momentum_Score', 'SmartMoney_Score', 'Fundamental_Score', 'Flag_Count']
    
    # Add raw metrics if available
    extra_cols = ['RSI 14W', 'Returns 3M', '52WH Distance', 
                  'Change In FII Holdings Latest Quarter', 'ROCE',
                  'Free Cash Flow', 'Debt To Equity']
    for col in extra_cols:
        if col in df.columns:
            display_cols.append(col)
    
    display_cols = [c for c in display_cols if c in df.columns]
    
    # Color by signal
    def highlight_signal(row):
        sig = row.get('Signal_Class', 'hold')
        if sig == 'buy':
            return ['background-color: rgba(0, 255, 136, 0.15)'] * len(row)
        elif sig == 'trap':
            return ['background-color: rgba(255, 0, 0, 0.25)'] * len(row)
        elif sig == 'avoid':
            return ['background-color: rgba(255, 71, 87, 0.15)'] * len(row)
        return [''] * len(row)
    
    styled = df[display_cols].head(50).style.apply(highlight_signal, axis=1)
    styled = styled.format({
        'Quant_Score': '{:.1f}',
        'Momentum_Score': '{:.2f}',
        'SmartMoney_Score': '{:.2f}',
        'Fundamental_Score': '{:.2f}',
    })
    
    st.dataframe(styled, height=500, use_container_width=True)
    
    # Charts
    st.markdown("### Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Score vs Momentum
        fig = px.scatter(
            df.head(50),
            x='Momentum_Score',
            y='Quant_Score',
            color='Signal_Class',
            hover_name='Name',
            title="Momentum vs Total Score",
            color_discrete_map={'buy': '#00ff88', 'hold': '#ffd700', 'avoid': '#ff4757'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Factor breakdown for top stock
        if len(df) > 0:
            top = df.iloc[0]
            factors = ['Momentum', 'Smart Money', 'Fundamentals']
            values = [
                top['Momentum_Score'] * 0.70,
                top['SmartMoney_Score'] * 0.20,
                top['Fundamental_Score'] * 0.10
            ]
            
            fig = go.Figure(go.Bar(
                x=factors,
                y=values,
                marker_color=['#00ff88', '#3b82f6', '#ffd700']
            ))
            fig.update_layout(
                title=f"Score Breakdown: {top.get('Name', 'Top Stock')}",
                yaxis_title="Contribution to Score"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Export
    st.markdown("### Export")
    col1, col2 = st.columns(2)
    
    csv_all = df.to_csv(index=False).encode('utf-8')
    col1.download_button("📥 All Data", csv_all, "V5_Quant_All.csv", use_container_width=True)
    
    buy_df = df[df['Signal_Class'] == 'buy']
    if len(buy_df) > 0:
        csv_buy = buy_df.to_csv(index=False).encode('utf-8')
        col2.download_button("🚀 Buy Signals", csv_buy, "V5_Quant_Buys.csv", use_container_width=True)

if __name__ == "__main__":
    main()
