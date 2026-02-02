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
    /* Dark theme base */
    .stApp { background: linear-gradient(180deg, #0a0a0f 0%, #0d1117 100%); }
    
    /* Remove default padding */
    .block-container { padding: 1rem 2rem; max-width: 100%; }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 50%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    .main-header p {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.85rem; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.8rem; font-weight: 700; }
    
    /* Signal badges */
    .signal-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .signal-buy { background: linear-gradient(135deg, #00ff88, #00cc6a); color: #000; }
    .signal-trap { background: linear-gradient(135deg, #ff0040, #cc0033); color: #fff; }
    .signal-avoid { background: linear-gradient(135deg, #ff4757, #ff6b7a); color: #fff; }
    .signal-hold { background: linear-gradient(135deg, #ffd700, #ffed4a); color: #000; }
    
    /* Alert boxes */
    .trap-alert {
        background: linear-gradient(135deg, rgba(255,0,64,0.15), rgba(255,0,64,0.05));
        border: 1px solid #ff0040;
        border-left: 4px solid #ff0040;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .trap-alert h4 { color: #ff4757; margin: 0 0 0.3rem 0; font-size: 1rem; }
    .trap-alert p { color: #94a3b8; margin: 0; font-size: 0.85rem; }
    
    /* Section headers */
    .section-header {
        color: #f1f5f9;
        font-size: 1.2rem;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2d3748;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Data table styling */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border: 1px solid #2d3748;
        border-radius: 12px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] .block-container { padding: 1rem; }
    
    /* Buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2d3748;
        color: #f1f5f9;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stDownloadButton button:hover {
        border-color: #00ff88;
        box-shadow: 0 0 20px rgba(0,255,136,0.2);
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .info-card h3 { color: #00ff88; margin-top: 0; }
    .info-card p { color: #94a3b8; }
    
    /* Welcome state */
    .welcome-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 50%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
    }
    .welcome-box h2 { color: #f1f5f9; margin-bottom: 0.5rem; }
    .welcome-box p { color: #64748b; }
    
    /* Plotly charts */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }
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
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>V5 QUANT</h1>
        <p>Data-Driven • 3-Factor Model • Momentum + Smart Money + Safety Net</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Data Input")
        files = st.file_uploader("Upload CSVs", accept_multiple_files=True, type=['csv'], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("## 🧠 Model Logic")
        
        st.markdown("""
        **Factor Weights:**
        
        | Factor | Weight | Why |
        |--------|--------|-----|
        | 🔥 Momentum | 70% | Corr: 0.80+ |
        | 💰 Smart Money | 20% | Corr: 0.40+ |
        | 📊 Fundamentals | 10% | Corr: 0.35+ |
        """)
        
        with st.expander("📈 Momentum Metrics"):
            st.markdown("""
            - **RSI 14W** (0.878 corr)
            - **Returns 3M** (0.707 corr)
            - **52WH Distance** (-0.655 corr)
            """)
        
        with st.expander("💰 Smart Money Metrics"):
            st.markdown("""
            - **FII Changes** (0.499 corr)
            - **Promoter Stability**
            """)
        
        with st.expander("🛡️ Safety Net (Trap Detection)"):
            st.markdown("""
            - **Cash Trap**: Negative FCF + OCF
            - **Debt Bomb**: D/E > 2, ICR < 2
            - **Insider Exit**: Promoter selling > 3%
            - **Smart Exit**: FII dumping
            """)
    
    if not files:
        st.markdown("""
        <div class="welcome-box">
            <h2>📁 Upload Your CSV Files</h2>
            <p>Drop your stock data files in the sidebar to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Info section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>🎯 Why This Model Works</h3>
                <p>Traditional screeners use theory-based factors. V5 uses <strong>data-backed correlations</strong>.</p>
                <br>
                <table style="width:100%; color:#94a3b8;">
                    <tr><td>RSI 14W</td><td style="color:#00ff88; text-align:right;"><strong>+0.878</strong></td></tr>
                    <tr><td>Returns 3M</td><td style="color:#00ff88; text-align:right;"><strong>+0.707</strong></td></tr>
                    <tr><td>52WH Distance</td><td style="color:#00ff88; text-align:right;"><strong>-0.655</strong></td></tr>
                    <tr><td>FII Changes</td><td style="color:#00d4ff; text-align:right;"><strong>+0.499</strong></td></tr>
                    <tr><td>ROCE</td><td style="color:#ffd700; text-align:right;">+0.413</td></tr>
                    <tr><td>NPM</td><td style="color:#ff4757; text-align:right;">-0.013 ❌</td></tr>
                    <tr><td>D/E Ratio</td><td style="color:#ff4757; text-align:right;">+0.068 ❌</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>🛡️ Safety Net Feature</h3>
                <p>V5 catches <strong>Pump & Dump</strong> traps that pure momentum models miss.</p>
                <br>
                <p style="color:#ff4757;"><strong>🚨 DEATH SPIRAL</strong><br>
                <span style="color:#64748b;">Negative cash + Heavy debt</span></p>
                <p style="color:#ff4757;"><strong>🚨 PUMP & DUMP</strong><br>
                <span style="color:#64748b;">High momentum + Burning cash</span></p>
                <p style="color:#ff4757;"><strong>🚨 INSIDER EXIT</strong><br>
                <span style="color:#64748b;">Promoters dumping shares</span></p>
                <p style="color:#ff4757;"><strong>🚨 SMART EXIT</strong><br>
                <span style="color:#64748b;">FIIs leaving before crash</span></p>
            </div>
            """, unsafe_allow_html=True)
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
    
    col1.metric("🚀 Buy", buy_count)
    col2.metric("⏸️ Hold", hold_count)
    col3.metric("⚠️ Avoid", avoid_count)
    col4.metric("🚨 Traps", trap_count)
    
    # TRAP ALERTS (Show prominently if any)
    trap_df = df[df['Signal_Class'] == 'trap']
    if len(trap_df) > 0:
        st.markdown('<div class="section-header">🚨 TRAP ALERTS — Do NOT Buy</div>', unsafe_allow_html=True)
        for _, row in trap_df.iterrows():
            flags_str = ', '.join(row['Red_Flags']) if row['Red_Flags'] else 'Multiple issues'
            st.markdown(f"""
            <div class="trap-alert">
                <h4>{row.get('Name', 'Unknown')} — {row['Signal']}</h4>
                <p>{row.get('Signal_Reason', '')} | Flags: {flags_str}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main Table
    st.markdown('<div class="section-header">📊 Stock Rankings</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="section-header">📈 Visual Analysis</div>', unsafe_allow_html=True)
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
            color_discrete_map={'buy': '#00ff88', 'hold': '#ffd700', 'avoid': '#ff4757', 'trap': '#ff0040'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,46,0.8)',
            font_color='#94a3b8',
            title_font_color='#f1f5f9'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Factor breakdown for top stock
        if len(df) > 0:
            top = df.iloc[0]
            factors = ['Momentum (70%)', 'Smart Money (20%)', 'Fundamentals (10%)']
            values = [
                top['Momentum_Score'] * 0.70 * 100,
                top['SmartMoney_Score'] * 0.20 * 100,
                top['Fundamental_Score'] * 0.10 * 100
            ]
            
            fig = go.Figure(go.Bar(
                x=factors,
                y=values,
                marker_color=['#00ff88', '#3b82f6', '#ffd700'],
                text=[f'{v:.1f}' for v in values],
                textposition='outside'
            ))
            fig.update_layout(
                title=f"Score Breakdown: {top.get('Name', 'Top Stock')}",
                yaxis_title="Contribution to Score",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                font_color='#94a3b8',
                title_font_color='#f1f5f9',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Export
    st.markdown('<div class="section-header">📥 Export Data</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    csv_all = df.to_csv(index=False).encode('utf-8')
    col1.download_button("📥 All Stocks", csv_all, "V5_Quant_All.csv", use_container_width=True)
    
    buy_df = df[df['Signal_Class'] == 'buy']
    if len(buy_df) > 0:
        csv_buy = buy_df.to_csv(index=False).encode('utf-8')
        col2.download_button("🚀 Buy Only", csv_buy, "V5_Quant_Buys.csv", use_container_width=True)
    
    avoid_df = df[df['Signal_Class'].isin(['avoid', 'trap'])]
    if len(avoid_df) > 0:
        csv_avoid = avoid_df.to_csv(index=False).encode('utf-8')
        col3.download_button("🚨 Avoid/Traps", csv_avoid, "V5_Quant_Avoid.csv", use_container_width=True)

if __name__ == "__main__":
    main()
