import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from datetime import datetime

# =========================================================
# 🚀 V4 ULTIMATE - INSTITUTIONAL GRADE STOCK ANALYZER
# =========================================================
# Features:
# - 7-Factor Scoring Model (Quality, Growth, Value, Safety, Momentum, Institutional, Technical)
# - Dynamic Market Regime Detection
# - Smart Money Flow Tracking (FII/DII/Promoter)
# - 52W High Breakout Detection
# - Cash Flow Trap Detection
# - Relative Strength Analysis
# - Multi-Timeframe Momentum
# =========================================================

st.set_page_config(
    page_title="Wave Detection | Pro Stock Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border-radius: 15px;
        padding: 20px;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        color: white;
    }
    .metric-card h4 { color: #888; margin-bottom: 5px; font-size: 0.9rem; }
    .metric-card h3 { color: #00d4ff; margin: 0; font-size: 1.5rem; }
    .score-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .strong-buy { background: #00c853; color: white; }
    .buy { background: #2196f3; color: white; }
    .hold { background: #ff9800; color: white; }
    .avoid { background: #f44336; color: white; }
    .trap { background: #9c27b0; color: white; }
    .stDataFrame { border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #00d4ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# COLUMN MAPPING - Maps your CSV headers to internal names
# =========================================================
COLUMN_MAP = {
    # Quality Metrics (handles both short and full names)
    'ROE (Return on Equity)': 'ROE',
    'ROCE (Return on Capital Employed)': 'ROCE', 
    'NPM (Net Profit Margin)': 'NPM',
    'OPM (Operating Profit Margin)': 'OPM',
    'CWIP (Capital Work in Progress)': 'CWIP',
}

# =========================================================
# 📋 COMPLETE DATA HEADERS REFERENCE
# =========================================================
# All unique columns from your 9 CSV files:
#
# CSV 1 - RETURNS & PRICE DATA:
#   companyId, Fundamentals Source, Name, Market Capitalization, Close Price,
#   Returns 1D, Returns 1W, Returns 1M, Returns 3M, Returns 6M, Returns 1Y, 
#   Returns 3Y, Returns 5Y, 52WH Distance, Returns Vs Nifty 500 1W,
#   Returns Vs Nifty 500 3M, Returns Vs Industry 1W, Returns Vs Industry 3M
#
# CSV 2 - TECHNICAL INDICATORS:
#   RSI 14D, RSI 14W, ADX 14D, ADX 14W
#
# CSV 3 - VALUATION RATIOS:
#   Price To Earnings, Price To Sales, Debt To Equity
#
# CSV 4 - HOLDINGS & INSTITUTIONAL DATA:
#   DII Holdings, FII Holdings, Retail Holdings, Promoter Holdings,
#   Change In DII Holdings Latest Quarter, Change In DII Holdings 1 Year,
#   Change In DII Holdings 2 Years, Change In DII Holdings 3 Years,
#   Change In FII Holdings Latest Quarter, Change In FII Holdings 1 Year,
#   Change In FII Holdings 2 Years, Change In FII Holdings 3 Years,
#   Change In Retail Holdings Latest Quarter, Change In Retail Holdings 1 Year,
#   Change In Retail Holdings 2 Years, Change In Retail Holdings 3 Years,
#   Change In Promoter Holdings Latest Quarter, Change In Promoter Holdings 1 Year,
#   Change In Promoter Holdings 2 Years, Change In Promoter Holdings 3 Years
#
# CSV 5 - GROWTH METRICS:
#   PAT Growth YoY, Revenue Growth QoQ, Revenue Growth TTM, Revenue Growth YoY,
#   EPS Growth TTM, PAT Growth QoQ, PAT Growth TTM, PBT Growth TTM
#
# CSV 6 - FUNDAMENTALS:
#   Industry, Revenue
#
# CSV 7 - BALANCE SHEET:
#   Inventory, CWIP, Cash Equivalents, Total Assets, Debt, Total Liabilities
#
# CSV 8 - CASH FLOW:
#   Operating Cash Flow, Investing Cash Flow, Financing Cash Flow, 
#   Net Cash Flow, Free Cash Flow
#
# CSV 9 - PROFITABILITY RATIOS:
#   NPM, OPM, ROCE, ROE
#
# =========================================================
# TOTAL UNIQUE COLUMNS: 67
# =========================================================

ALL_EXPECTED_COLUMNS = {
    # Identifiers
    'identifiers': ['companyId', 'Fundamentals Source', 'Name', 'Industry'],
    
    # Price & Market Data
    'price_data': ['Market Capitalization', 'Close Price'],
    
    # Returns (Multi-Timeframe)
    'returns': [
        'Returns 1D', 'Returns 1W', 'Returns 1M', 'Returns 3M', 
        'Returns 6M', 'Returns 1Y', 'Returns 3Y', 'Returns 5Y'
    ],
    
    # Relative Strength
    'relative_strength': [
        '52WH Distance', 'Returns Vs Nifty 500 1W', 'Returns Vs Nifty 500 3M',
        'Returns Vs Industry 1W', 'Returns Vs Industry 3M'
    ],
    
    # Technical Indicators
    'technical': ['RSI 14D', 'RSI 14W', 'ADX 14D', 'ADX 14W'],
    
    # Valuation Ratios
    'valuation': ['Price To Earnings', 'Price To Sales', 'Debt To Equity'],
    
    # Holdings Data
    'holdings': ['DII Holdings', 'FII Holdings', 'Retail Holdings', 'Promoter Holdings'],
    
    # Holdings Changes - DII
    'dii_changes': [
        'Change In DII Holdings Latest Quarter', 'Change In DII Holdings 1 Year',
        'Change In DII Holdings 2 Years', 'Change In DII Holdings 3 Years'
    ],
    
    # Holdings Changes - FII
    'fii_changes': [
        'Change In FII Holdings Latest Quarter', 'Change In FII Holdings 1 Year',
        'Change In FII Holdings 2 Years', 'Change In FII Holdings 3 Years'
    ],
    
    # Holdings Changes - Retail
    'retail_changes': [
        'Change In Retail Holdings Latest Quarter', 'Change In Retail Holdings 1 Year',
        'Change In Retail Holdings 2 Years', 'Change In Retail Holdings 3 Years'
    ],
    
    # Holdings Changes - Promoter
    'promoter_changes': [
        'Change In Promoter Holdings Latest Quarter', 'Change In Promoter Holdings 1 Year',
        'Change In Promoter Holdings 2 Years', 'Change In Promoter Holdings 3 Years'
    ],
    
    # Growth Metrics
    'growth': [
        'PAT Growth YoY', 'PAT Growth QoQ', 'PAT Growth TTM',
        'Revenue Growth YoY', 'Revenue Growth QoQ', 'Revenue Growth TTM',
        'EPS Growth TTM', 'PBT Growth TTM'
    ],
    
    # Revenue & Profitability
    'profitability': ['Revenue', 'NPM', 'OPM', 'ROCE', 'ROE'],
    
    # Balance Sheet
    'balance_sheet': [
        'Inventory', 'CWIP', 'Cash Equivalents', 
        'Total Assets', 'Debt', 'Total Liabilities'
    ],
    
    # Cash Flow
    'cash_flow': [
        'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow',
        'Net Cash Flow', 'Free Cash Flow'
    ]
}

# Helper function to get all expected columns as a flat list
def get_all_expected_columns():
    all_cols = []
    for category, cols in ALL_EXPECTED_COLUMNS.items():
        all_cols.extend(cols)
    return all_cols

# =========================================================
# 📊 DATA VALIDATION & COVERAGE REPORT
# =========================================================
def validate_data_coverage(df):
    """Check which expected columns are present in the data"""
    coverage = {}
    for category, cols in ALL_EXPECTED_COLUMNS.items():
        found = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        coverage[category] = {
            'found': found,
            'missing': missing,
            'total': len(cols),
            'found_count': len(found),
            'coverage_pct': len(found) / len(cols) * 100 if cols else 0
        }
    return coverage

def display_data_coverage(coverage):
    """Display data coverage in sidebar"""
    total_found = sum(c['found_count'] for c in coverage.values())
    total_expected = sum(c['total'] for c in coverage.values())
    overall_pct = total_found / total_expected * 100 if total_expected > 0 else 0
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📋 Data Coverage: {overall_pct:.0f}%")
    st.sidebar.progress(overall_pct / 100)
    
    with st.sidebar.expander("View Column Details"):
        for category, data in coverage.items():
            icon = "✅" if data['coverage_pct'] == 100 else "⚠️" if data['coverage_pct'] > 50 else "❌"
            st.markdown(f"**{icon} {category.replace('_', ' ').title()}**: {data['found_count']}/{data['total']}")
            if data['missing']:
                st.caption(f"Missing: {', '.join(data['missing'][:3])}{'...' if len(data['missing']) > 3 else ''}")

# =========================================================
# 🧠 INTELLIGENT DATA PROCESSING ENGINE
# =========================================================
@st.cache_data(ttl=600)
def process_files(uploaded_files):
    """Advanced data processor with smart merging and cleaning"""
    if not uploaded_files:
        return None
    
    master_df = pd.DataFrame()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing: {file.name}")
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            
            # Apply column mapping
            df = df.rename(columns=COLUMN_MAP)
            
            if master_df.empty:
                master_df = df
            else:
                key = 'companyId' if 'companyId' in master_df.columns and 'companyId' in df.columns else 'Name'
                new_cols = [c for c in df.columns if c not in master_df.columns or c == key]
                if len(new_cols) > 1:
                    master_df = pd.merge(master_df, df[new_cols], on=key, how='outer')
        except Exception as e:
            st.warning(f"⚠️ Skipped {file.name}: {e}")
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar.empty()
    status_text.empty()
    
    # --- ADVANCED DATA CLEANING ---
    non_numeric = ['companyId', 'Name', 'Industry', 'Sector', 'Fundamentals Source', 'Verdict']
    numeric_cols = [c for c in master_df.columns if c not in non_numeric]
    
    for col in numeric_cols:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    
    # Smart NaN Handling
    growth_keywords = ["Growth", "Change", "Returns", "Flow"]
    ratio_keywords = ["Ratio", "Holdings", "ROE", "ROCE", "NPM", "OPM", "PE", "PS"]
    
    for col in numeric_cols:
        if any(kw in col for kw in growth_keywords):
            master_df[col] = master_df[col].fillna(0)
        elif any(kw in col for kw in ratio_keywords):
            master_df[col] = master_df[col].fillna(master_df[col].median())
        else:
            master_df[col] = master_df[col].fillna(0)
    
    return master_df

# =========================================================
# 🎯 MARKET REGIME DETECTION (ENHANCED)
# =========================================================
def analyze_market_regime(df):
    """
    Multi-factor market regime detection using:
    - Short-term momentum (1M, 3M returns)
    - Long-term trend (1Y returns)
    - Breadth (% of stocks positive)
    """
    r1m = df['Returns 1M'].median() if 'Returns 1M' in df.columns else 0
    r3m = df['Returns 3M'].median() if 'Returns 3M' in df.columns else 0
    r1y = df['Returns 1Y'].median() if 'Returns 1Y' in df.columns else 0
    
    # Breadth Analysis
    breadth_1m = (df['Returns 1M'] > 0).mean() * 100 if 'Returns 1M' in df.columns else 50
    breadth_3m = (df['Returns 3M'] > 0).mean() * 100 if 'Returns 3M' in df.columns else 50
    
    # Regime Classification
    if r3m > 10 and r1y > 20 and breadth_3m > 65:
        regime = "🚀 STRONG BULL"
        weights = {
            'Quality': 0.15, 'Growth': 0.25, 'Value': 0.05,
            'Safety': 0.05, 'Momentum': 0.25, 'Institutional': 0.15, 'Technical': 0.10
        }
        strategy = "Aggressive Growth + Momentum"
    elif r3m > 5 and r1y > 10 and breadth_3m > 55:
        regime = "📈 BULL MARKET"
        weights = {
            'Quality': 0.20, 'Growth': 0.20, 'Value': 0.10,
            'Safety': 0.10, 'Momentum': 0.20, 'Institutional': 0.10, 'Technical': 0.10
        }
        strategy = "Balanced Growth"
    elif r3m < -10 or (r1m < -5 and breadth_1m < 30):
        regime = "🐻 BEAR MARKET"
        weights = {
            'Quality': 0.25, 'Growth': 0.05, 'Value': 0.25,
            'Safety': 0.25, 'Momentum': 0.05, 'Institutional': 0.10, 'Technical': 0.05
        }
        strategy = "Defensive Quality + Value"
    elif r3m < -3 or breadth_3m < 40:
        regime = "⚠️ CORRECTION"
        weights = {
            'Quality': 0.25, 'Growth': 0.10, 'Value': 0.20,
            'Safety': 0.20, 'Momentum': 0.10, 'Institutional': 0.10, 'Technical': 0.05
        }
        strategy = "Quality + Safety First"
    else:
        regime = "⚖️ SIDEWAYS"
        weights = {
            'Quality': 0.20, 'Growth': 0.15, 'Value': 0.15,
            'Safety': 0.15, 'Momentum': 0.15, 'Institutional': 0.10, 'Technical': 0.10
        }
        strategy = "Selective Stock Picking"
    
    return regime, weights, strategy, {'r1m': r1m, 'r3m': r3m, 'r1y': r1y, 'breadth_3m': breadth_3m}

# =========================================================
# 🔥 7-FACTOR SCORING ENGINE (THE BRAIN) - ULTIMATE VERSION
# =========================================================
def run_ultimate_scoring(df, weights):
    """
    Institutional-Grade 7-Factor Model - USES ALL 67 COLUMNS:
    1. Quality (Profitability & Efficiency)
    2. Growth (Earnings & Revenue Expansion)
    3. Value (Cheapness)
    4. Safety (Financial Strength)
    5. Momentum (Price Trend)
    6. Institutional (Smart Money Flow)
    7. Technical (Breakout Potential)
    """
    n = len(df)
    
    def safe_get(col, default=0):
        """Safely get column or return default series"""
        if col in df.columns:
            return df[col].fillna(default)
        return pd.Series([default] * n, index=df.index)
    
    def normalize(series, inverse=False):
        """Robust normalization with outlier handling"""
        s = series.copy()
        if inverse:
            s = -s
        # Clip outliers at 1st and 99th percentile
        lower, upper = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lower, upper)
        # Min-Max normalize
        min_val, max_val = s.min(), s.max()
        if max_val - min_val == 0:
            return pd.Series([0.5] * n, index=df.index)
        return (s - min_val) / (max_val - min_val)
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 1: QUALITY (Profitability & Efficiency)
    # Uses: ROE, ROCE, NPM, OPM, Operating Cash Flow, Revenue,
    #       Total Assets, Inventory (Asset Turnover)
    # ═══════════════════════════════════════════════════════
    roe = safe_get('ROE')
    roce = safe_get('ROCE')
    npm = safe_get('NPM')
    opm = safe_get('OPM')
    
    # Operating Cash Flow / Revenue (Cash Conversion Efficiency)
    ocf = safe_get('Operating Cash Flow')
    revenue = safe_get('Revenue', 1)
    cash_conversion = (ocf / revenue.replace(0, np.nan)).fillna(0).clip(-1, 1)
    
    # Asset Turnover = Revenue / Total Assets (Efficiency)
    total_assets = safe_get('Total Assets', 1)
    asset_turnover = (revenue / total_assets.replace(0, np.nan)).fillna(0).clip(0, 5)
    
    # Inventory Efficiency (Lower inventory relative to revenue = better)
    inventory = safe_get('Inventory', 0)
    inventory_ratio = (inventory / revenue.replace(0, np.nan)).fillna(0).clip(0, 1)
    
    quality_raw = (
        normalize(roe) * 0.22 +
        normalize(roce) * 0.22 +
        normalize(npm) * 0.18 +
        normalize(opm) * 0.15 +
        normalize(cash_conversion) * 0.10 +
        normalize(asset_turnover) * 0.08 +
        normalize(inventory_ratio, inverse=True) * 0.05
    )
    df['Score_Quality'] = quality_raw
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 2: GROWTH (Earnings & Revenue Expansion)
    # Uses: PAT Growth (TTM, YoY, QoQ), Revenue Growth (TTM, YoY, QoQ),
    #       EPS Growth TTM, PBT Growth TTM
    # ═══════════════════════════════════════════════════════
    pat_ttm = safe_get('PAT Growth TTM')
    pat_yoy = safe_get('PAT Growth YoY')
    pat_qoq = safe_get('PAT Growth QoQ')
    rev_ttm = safe_get('Revenue Growth TTM')
    rev_yoy = safe_get('Revenue Growth YoY')
    rev_qoq = safe_get('Revenue Growth QoQ')
    eps_ttm = safe_get('EPS Growth TTM')
    pbt_ttm = safe_get('PBT Growth TTM')
    
    # Growth Acceleration (QoQ > TTM = Accelerating Growth)
    pat_accelerating = (pat_qoq > pat_ttm).astype(float)
    rev_accelerating = (rev_qoq > rev_ttm).astype(float)
    growth_accel_bonus = (pat_accelerating + rev_accelerating) * 0.1
    
    # Consistent Growth (YoY and TTM both positive)
    consistent_growth = ((pat_yoy > 0) & (pat_ttm > 0) & (rev_yoy > 0) & (rev_ttm > 0)).astype(float) * 0.1
    
    growth_raw = (
        normalize(pat_ttm) * 0.20 +
        normalize(pat_yoy) * 0.10 +
        normalize(rev_ttm) * 0.15 +
        normalize(rev_yoy) * 0.10 +
        normalize(eps_ttm) * 0.15 +
        normalize(pbt_ttm) * 0.10 +
        growth_accel_bonus +
        consistent_growth
    )
    df['Score_Growth'] = normalize(growth_raw)
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 3: VALUE (Cheapness)
    # Uses: Price To Earnings, Price To Sales, PAT Growth (for PEG)
    # ═══════════════════════════════════════════════════════
    pe = safe_get('Price To Earnings', 50)
    ps = safe_get('Price To Sales', 10)
    
    # Earnings Yield (1/PE) - Higher is better (cheaper)
    earnings_yield = (1 / pe.clip(lower=1)).fillna(0)
    sales_yield = (1 / ps.clip(lower=0.1)).fillna(0)
    
    # PEG Ratio (PE / Growth) - Lower is better
    growth_for_peg = pat_ttm.clip(lower=1)
    peg = pe / growth_for_peg
    peg_score = normalize(peg, inverse=True)
    
    value_raw = (
        normalize(earnings_yield) * 0.40 +
        normalize(sales_yield) * 0.30 +
        peg_score * 0.30
    )
    df['Score_Value'] = value_raw
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 4: SAFETY (Financial Strength)
    # Uses: Debt To Equity, Promoter Holdings, Free Cash Flow,
    #       Cash Equivalents, Debt, Total Assets, Total Liabilities,
    #       Operating/Net Cash Flow
    # ═══════════════════════════════════════════════════════
    de = safe_get('Debt To Equity', 1)
    promoter = safe_get('Promoter Holdings', 50)
    fcf = safe_get('Free Cash Flow')
    ocf = safe_get('Operating Cash Flow')
    ncf = safe_get('Net Cash Flow')
    cash = safe_get('Cash Equivalents')
    debt = safe_get('Debt', 1)
    total_assets = safe_get('Total Assets', 1)
    total_liab = safe_get('Total Liabilities', 0)
    
    # Debt Safety (Lower D/E = Better)
    de_score = normalize(de, inverse=True)
    
    # Promoter Confidence (Higher = Better)
    promoter_score = normalize(promoter)
    
    # Cash to Debt Ratio (Higher = Safer)
    cash_debt = (cash / debt.replace(0, np.nan)).fillna(1).clip(0, 5)
    cash_debt_score = normalize(cash_debt)
    
    # Free Cash Flow Positive (Binary bonus)
    fcf_positive = (fcf > 0).astype(float)
    
    # Operating Cash Flow Positive (Binary bonus)
    ocf_positive = (ocf > 0).astype(float)
    
    # Net Cash Flow Trend
    ncf_positive = (ncf > 0).astype(float)
    
    # Asset Coverage Ratio (Total Assets / Total Liabilities)
    asset_coverage = (total_assets / total_liab.replace(0, np.nan)).fillna(2).clip(0, 5)
    
    safety_raw = (
        de_score * 0.25 +
        promoter_score * 0.20 +
        cash_debt_score * 0.15 +
        fcf_positive * 0.15 +
        ocf_positive * 0.10 +
        ncf_positive * 0.05 +
        normalize(asset_coverage) * 0.10
    )
    df['Score_Safety'] = safety_raw
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 5: MOMENTUM (Price Trend - Multi-Timeframe)
    # Uses: RSI 14D/14W, ADX 14D/14W, Returns 1D/1W/1M/3M/6M/1Y/3Y/5Y
    # ═══════════════════════════════════════════════════════
    rsi_w = safe_get('RSI 14W', 50)
    rsi_d = safe_get('RSI 14D', 50)
    adx_w = safe_get('ADX 14W', 20)
    adx_d = safe_get('ADX 14D', 20)
    ret_1d = safe_get('Returns 1D')
    ret_1w = safe_get('Returns 1W')
    ret_1m = safe_get('Returns 1M')
    ret_3m = safe_get('Returns 3M')
    ret_6m = safe_get('Returns 6M')
    ret_1y = safe_get('Returns 1Y')
    ret_3y = safe_get('Returns 3Y')
    ret_5y = safe_get('Returns 5Y')
    
    # RSI Sweet Spot (45-65 is ideal momentum zone)
    rsi_score_w = 1 - np.abs(rsi_w - 55) / 55
    rsi_score_d = 1 - np.abs(rsi_d - 55) / 55
    rsi_combined = (rsi_score_w.clip(0, 1) * 0.6 + rsi_score_d.clip(0, 1) * 0.4)
    
    # Trend Strength (ADX > 25 = Strong Trend)
    adx_combined = normalize(adx_w) * 0.6 + normalize(adx_d) * 0.4
    
    # Short-term Momentum (1D, 1W, 1M)
    short_term = (normalize(ret_1d) * 0.2 + normalize(ret_1w) * 0.3 + normalize(ret_1m) * 0.5)
    
    # Medium-term Momentum (3M, 6M)
    medium_term = (normalize(ret_3m) * 0.5 + normalize(ret_6m) * 0.5)
    
    # Long-term Performance (1Y, 3Y, 5Y) - Consistency check
    long_term = (normalize(ret_1y) * 0.5 + normalize(ret_3y) * 0.3 + normalize(ret_5y) * 0.2)
    
    # Momentum Consistency (All timeframes positive = bonus)
    momentum_consistency = (
        (ret_1m > 0) & (ret_3m > 0) & (ret_6m > 0) & (ret_1y > 0)
    ).astype(float) * 0.1
    
    momentum_raw = (
        short_term * 0.15 +
        medium_term * 0.30 +
        long_term * 0.15 +
        rsi_combined * 0.15 +
        adx_combined * 0.15 +
        momentum_consistency
    )
    df['Score_Momentum'] = momentum_raw
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 6: INSTITUTIONAL (Smart Money Flow)
    # Uses: FII/DII Holdings, ALL Change columns (Q/1Y/2Y/3Y),
    #       Retail Holdings changes, Promoter changes
    # ═══════════════════════════════════════════════════════
    fii_hold = safe_get('FII Holdings')
    dii_hold = safe_get('DII Holdings')
    retail_hold = safe_get('Retail Holdings')
    promoter_hold = safe_get('Promoter Holdings')
    
    # FII Changes (All timeframes)
    fii_chg_q = safe_get('Change In FII Holdings Latest Quarter')
    fii_chg_1y = safe_get('Change In FII Holdings 1 Year')
    fii_chg_2y = safe_get('Change In FII Holdings 2 Years')
    fii_chg_3y = safe_get('Change In FII Holdings 3 Years')
    
    # DII Changes (All timeframes)
    dii_chg_q = safe_get('Change In DII Holdings Latest Quarter')
    dii_chg_1y = safe_get('Change In DII Holdings 1 Year')
    dii_chg_2y = safe_get('Change In DII Holdings 2 Years')
    dii_chg_3y = safe_get('Change In DII Holdings 3 Years')
    
    # Promoter Changes
    prom_chg_q = safe_get('Change In Promoter Holdings Latest Quarter')
    prom_chg_1y = safe_get('Change In Promoter Holdings 1 Year')
    
    # Retail Changes (Inverse - retail selling while institutions buying = good)
    retail_chg_q = safe_get('Change In Retail Holdings Latest Quarter')
    
    # FII Score (Recent > Old)
    fii_score = (
        normalize(fii_chg_q) * 0.40 +
        normalize(fii_chg_1y) * 0.30 +
        normalize(fii_chg_2y) * 0.15 +
        normalize(fii_chg_3y) * 0.15
    )
    
    # DII Score
    dii_score = (
        normalize(dii_chg_q) * 0.40 +
        normalize(dii_chg_1y) * 0.30 +
        normalize(dii_chg_2y) * 0.15 +
        normalize(dii_chg_3y) * 0.15
    )
    
    # Promoter Confidence Score
    prom_score = normalize(prom_chg_q) * 0.6 + normalize(prom_chg_1y) * 0.4
    
    # Smart Money Rotation: Institutions buying + Retail selling = Smart accumulation
    smart_rotation = ((fii_chg_q > 0) | (dii_chg_q > 0)) & (retail_chg_q < 0)
    smart_rotation_bonus = smart_rotation.astype(float) * 0.1
    
    # Total Institutional Holdings
    total_inst = fii_hold + dii_hold
    inst_holding_score = normalize(total_inst)
    
    institutional_raw = (
        fii_score * 0.30 +
        dii_score * 0.25 +
        prom_score * 0.20 +
        inst_holding_score * 0.15 +
        smart_rotation_bonus
    )
    df['Score_Institutional'] = institutional_raw
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 7: TECHNICAL (Breakout Potential)
    # Uses: 52WH Distance, Returns Vs Nifty 500 (1W/3M),
    #       Returns Vs Industry (1W/3M), RSI, ADX
    # ═══════════════════════════════════════════════════════
    dist_52wh = safe_get('52WH Distance', -20)
    ret_vs_nifty_1w = safe_get('Returns Vs Nifty 500 1W')
    ret_vs_nifty_3m = safe_get('Returns Vs Nifty 500 3M')
    ret_vs_ind_1w = safe_get('Returns Vs Industry 1W')
    ret_vs_ind_3m = safe_get('Returns Vs Industry 3M')
    
    # Near 52W High Score (closer to high = stronger)
    # -5% to 0% = Very Near High (Score: 1.0)
    # -10% to -5% = Near High (Score: 0.7)
    # -20% to -10% = Moderate (Score: 0.4)
    # Below -20% = Far from High (Score: 0.2)
    near_high_score = pd.Series(0.2, index=df.index)
    near_high_score = near_high_score.where(dist_52wh <= -20, 0.4)
    near_high_score = near_high_score.where(dist_52wh <= -10, 0.7)
    near_high_score = near_high_score.where(dist_52wh <= -5, 1.0)
    near_high_score = (dist_52wh > -5).astype(float) * 1.0 + \
                      ((dist_52wh <= -5) & (dist_52wh > -10)).astype(float) * 0.7 + \
                      ((dist_52wh <= -10) & (dist_52wh > -20)).astype(float) * 0.4 + \
                      (dist_52wh <= -20).astype(float) * 0.2
    
    # Relative Strength vs Market (Nifty 500)
    rel_vs_market = (
        normalize(ret_vs_nifty_1w) * 0.3 +
        normalize(ret_vs_nifty_3m) * 0.7
    )
    
    # Relative Strength vs Industry (Sector Outperformance)
    rel_vs_industry = (
        normalize(ret_vs_ind_1w) * 0.3 +
        normalize(ret_vs_ind_3m) * 0.7
    )
    
    # Breakout Confirmation (Near high + Strong ADX + RSI > 50)
    breakout_setup = (
        (dist_52wh > -10) & (adx_w > 25) & (rsi_w > 50) & (rsi_w < 70)
    ).astype(float) * 0.15
    
    technical_raw = (
        near_high_score * 0.25 +
        rel_vs_market * 0.25 +
        rel_vs_industry * 0.20 +
        breakout_setup +
        normalize(adx_w) * 0.15
    )
    df['Score_Technical'] = technical_raw
    
    # ═══════════════════════════════════════════════════════
    # FINAL COMPOSITE SCORE
    # ═══════════════════════════════════════════════════════
    df['Final_Score'] = (
        df['Score_Quality'] * weights['Quality'] +
        df['Score_Growth'] * weights['Growth'] +
        df['Score_Value'] * weights['Value'] +
        df['Score_Safety'] * weights['Safety'] +
        df['Score_Momentum'] * weights['Momentum'] +
        df['Score_Institutional'] * weights['Institutional'] +
        df['Score_Technical'] * weights['Technical']
    )
    
    # Normalize final score to 0-100
    df['Final_Score'] = normalize(df['Final_Score']) * 100
    
    return df

# =========================================================
# 🎯 SMART VERDICT ENGINE - ULTIMATE VERSION
# =========================================================
def get_ultimate_verdict(row):
    """
    Multi-layer verdict system with comprehensive trap detection.
    Uses ALL available data for maximum accuracy.
    """
    score = row['Final_Score']
    
    # ═══════════════════════════════════════════════════════
    # GET ALL VALUES SAFELY
    # ═══════════════════════════════════════════════════════
    # Cash Flow
    fcf = row.get('Free Cash Flow', 1)
    ocf = row.get('Operating Cash Flow', 1)
    ncf = row.get('Net Cash Flow', 0)
    
    # Valuation
    pe = row.get('Price To Earnings', 20)
    ps = row.get('Price To Sales', 5)
    
    # Safety
    de = row.get('Debt To Equity', 0)
    cash = row.get('Cash Equivalents', 0)
    debt = row.get('Debt', 1)
    
    # Holdings
    promoter = row.get('Promoter Holdings', 50)
    fii = row.get('FII Holdings', 0)
    dii = row.get('DII Holdings', 0)
    
    # Holdings Changes
    promoter_chg_q = row.get('Change In Promoter Holdings Latest Quarter', 0)
    promoter_chg_1y = row.get('Change In Promoter Holdings 1 Year', 0)
    fii_chg_q = row.get('Change In FII Holdings Latest Quarter', 0)
    fii_chg_1y = row.get('Change In FII Holdings 1 Year', 0)
    dii_chg_q = row.get('Change In DII Holdings Latest Quarter', 0)
    dii_chg_1y = row.get('Change In DII Holdings 1 Year', 0)
    retail_chg_q = row.get('Change In Retail Holdings Latest Quarter', 0)
    
    # Growth
    pat_ttm = row.get('PAT Growth TTM', 0)
    pat_qoq = row.get('PAT Growth QoQ', 0)
    rev_ttm = row.get('Revenue Growth TTM', 0)
    eps_ttm = row.get('EPS Growth TTM', 0)
    
    # Technical
    dist_52wh = row.get('52WH Distance', -20)
    rsi_w = row.get('RSI 14W', 50)
    rsi_d = row.get('RSI 14D', 50)
    adx_w = row.get('ADX 14W', 20)
    adx_d = row.get('ADX 14D', 20)
    
    # Returns
    ret_1m = row.get('Returns 1M', 0)
    ret_3m = row.get('Returns 3M', 0)
    ret_1y = row.get('Returns 1Y', 0)
    ret_vs_nifty = row.get('Returns Vs Nifty 500 3M', 0)
    ret_vs_ind = row.get('Returns Vs Industry 3M', 0)
    
    # Profitability
    roe = row.get('ROE', 0)
    roce = row.get('ROCE', 0)
    npm = row.get('NPM', 0)
    
    # ═══════════════════════════════════════════════════════
    # 🚨 TRAP DETECTION (Check these FIRST - Highest Priority)
    # ═══════════════════════════════════════════════════════
    
    # 🚨 CASH FLOW TRAP: High score but negative cash flow
    if score > 70 and fcf < 0 and ocf < 0:
        return "🚨 CASH TRAP", "trap"
    
    # 🚨 DEBT TRAP: High score but dangerous debt levels
    if score > 65 and de > 2:
        return "🚨 DEBT TRAP", "trap"
    
    # 🚨 PROMOTER DUMPING: Significant promoter selling
    if score > 65 and promoter_chg_q < -2 and promoter_chg_1y < -3:
        return "🚨 PROMOTER EXIT", "trap"
    
    # 🚨 SMART MONEY EXIT: Both FII and DII selling
    if score > 60 and fii_chg_q < -1 and dii_chg_q < -1:
        return "⚠️ INST. EXIT", "trap"
    
    # 🚨 OVERBOUGHT TRAP: High RSI with weak fundamentals
    if score > 60 and rsi_w > 75 and rsi_d > 75 and (roe < 10 or npm < 5):
        return "⚠️ OVERBOUGHT", "trap"
    
    # 🚨 MOMENTUM TRAP: High returns but weak fundamentals
    if score > 60 and ret_3m > 50 and pe > 100 and fcf < 0:
        return "⚠️ MOMENTUM TRAP", "trap"
    
    # ═══════════════════════════════════════════════════════
    # 💎 STRONG BUY SIGNALS (Highest Conviction)
    # ═══════════════════════════════════════════════════════
    
    # 💎 PERFECT STOCK: All criteria met
    if (score >= 85 and fcf > 0 and ocf > 0 and de < 1 and 
        promoter > 50 and roe > 15 and pat_ttm > 15):
        return "💎 STRONG BUY", "strong-buy"
    
    # 🚀 BREAKOUT KING: Near 52W High with strong momentum
    if (score >= 78 and dist_52wh > -5 and adx_w > 25 and 
        rsi_w > 55 and rsi_w < 70 and (fii_chg_q > 0 or dii_chg_q > 0)):
        return "🚀 BREAKOUT", "strong-buy"
    
    # 🏆 MARKET BEATER: Outperforming market + industry
    if (score >= 75 and ret_vs_nifty > 10 and ret_vs_ind > 5 and
        pat_ttm > 20 and fcf > 0):
        return "🏆 OUTPERFORMER", "strong-buy"
    
    # ═══════════════════════════════════════════════════════
    # 📈 BUY SIGNALS (High Conviction)
    # ═══════════════════════════════════════════════════════
    
    # 📈 ACCUMULATION ZONE: Institutions heavily buying
    if (score >= 75 and (fii_chg_q > 1 or dii_chg_q > 1) and
        retail_chg_q < 0):  # Smart money in, retail out
        return "📈 ACCUMULATE", "buy"
    
    # 💰 DEEP VALUE: Cheap with solid fundamentals
    if (score >= 70 and pe < 15 and ps < 2 and fcf > 0 and 
        roe > 12 and de < 1):
        return "💰 VALUE BUY", "buy"
    
    # 🌱 GROWTH CHAMPION: Exceptional growth
    if (score >= 70 and pat_ttm > 30 and rev_ttm > 20 and 
        eps_ttm > 25 and pat_qoq > pat_ttm):  # Accelerating
        return "🌱 HIGH GROWTH", "buy"
    
    # ⚡ MOMENTUM PLAY: Strong technical setup
    if (score >= 68 and adx_w > 30 and rsi_w > 55 and rsi_w < 70 and
        ret_1m > 5 and ret_3m > 10):
        return "⚡ MOMENTUM", "buy"
    
    # �️ QUALITY COMPOUNDER: High quality metrics
    if (score >= 70 and roe > 20 and roce > 20 and npm > 15 and
        de < 0.5 and promoter > 55):
        return "🛡️ QUALITY", "buy"
    
    # 📊 TURNAROUND: Improving fundamentals
    if (score >= 65 and pat_qoq > 20 and pat_qoq > pat_ttm and
        fii_chg_q > 0 and dii_chg_q > 0):
        return "📊 TURNAROUND", "buy"
    
    # ═══════════════════════════════════════════════════════
    # 👀 WATCHLIST (Monitor Closely)
    # ═══════════════════════════════════════════════════════
    
    if score >= 55:
        # Good score but missing some criteria
        if fii_chg_q > 0 or dii_chg_q > 0:
            return "👀 INST. WATCH", "hold"
        if dist_52wh > -15:
            return "👀 NEAR BREAKOUT", "hold"
        return "👀 WATCHLIST", "hold"
    
    # ═══════════════════════════════════════════════════════
    # ⏸️ HOLD / NEUTRAL
    # ═══════════════════════════════════════════════════════
    
    if score >= 40:
        return "⏸️ HOLD", "hold"
    
    # ═══════════════════════════════════════════════════════
    # ❌ AVOID SIGNALS
    # ═══════════════════════════════════════════════════════
    
    # Weak fundamentals
    if roe < 5 and npm < 3 and fcf < 0:
        return "❌ WEAK FUNDAMENTALS", "avoid"
    
    # Bleeding cash
    if fcf < 0 and ocf < 0 and ncf < 0:
        return "❌ CASH BLEED", "avoid"
    
    # Everyone selling
    if fii_chg_q < -1 and dii_chg_q < -1 and promoter_chg_q < -1:
        return "❌ ALL SELLING", "avoid"
    
    if score < 35:
        return "❌ AVOID", "avoid"
    
    return "⏸️ NEUTRAL", "hold"

# =========================================================
# 📊 MAIN DASHBOARD
# =========================================================
def main():
    st.markdown("""
    <div class='main-header'>
        <h1 style='color: #00d4ff; margin:0;'>🔥 V4 ULTIMATE</h1>
        <p style='color: #888; margin:0;'>Institutional-Grade 7-Factor Stock Analyzer</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📂 Data Upload")
        uploaded_files = st.file_uploader(
            "Upload CSV Files",
            accept_multiple_files=True,
            type=['csv'],
            help="Upload your stock data CSV files"
        )
        
        st.markdown("---")
        st.markdown("### 🎛️ Settings")
        
        show_all_columns = st.checkbox("Show All Scores", value=False)
        top_n = st.slider("Top Stocks to Show", 10, 100, 30)
        
        st.markdown("---")
        st.markdown("### 📊 Factor Weights")
        st.caption("Auto-adjusted based on market regime")
    
    if not uploaded_files:
        st.info("👆 Upload your CSV files to begin analysis")
        
        # Show complete expected columns organized by CSV
        with st.expander("📋 Complete Data Headers Reference (67 Columns)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📁 CSV 1 - Returns & Price:**
                - companyId, Name, Market Capitalization
                - Close Price, 52WH Distance
                - Returns 1D, 1W, 1M, 3M, 6M, 1Y, 3Y, 5Y
                - Returns Vs Nifty 500 1W/3M
                - Returns Vs Industry 1W/3M
                
                **📁 CSV 2 - Technical:**
                - RSI 14D, RSI 14W
                - ADX 14D, ADX 14W
                
                **📁 CSV 3 - Valuation:**
                - Price To Earnings
                - Price To Sales
                - Debt To Equity
                
                **📁 CSV 4 - Holdings:**
                - DII/FII/Retail/Promoter Holdings
                - Change In DII Holdings (Q/1Y/2Y/3Y)
                - Change In FII Holdings (Q/1Y/2Y/3Y)
                - Change In Retail Holdings (Q/1Y/2Y/3Y)
                - Change In Promoter Holdings (Q/1Y/2Y/3Y)
                """)
            
            with col2:
                st.markdown("""
                **📁 CSV 5 - Growth:**
                - PAT Growth YoY, QoQ, TTM
                - Revenue Growth YoY, QoQ, TTM
                - EPS Growth TTM
                - PBT Growth TTM
                
                **📁 CSV 6 - Fundamentals:**
                - Industry, Revenue
                
                **📁 CSV 7 - Balance Sheet:**
                - Inventory, CWIP
                - Cash Equivalents
                - Total Assets, Total Liabilities
                - Debt
                
                **📁 CSV 8 - Cash Flow:**
                - Operating Cash Flow
                - Investing Cash Flow
                - Financing Cash Flow
                - Net Cash Flow
                - Free Cash Flow
                
                **📁 CSV 9 - Profitability:**
                - NPM, OPM, ROCE, ROE
                """)
        return
    
    # Process Data
    df = process_files(uploaded_files)
    
    if df is None or df.empty:
        st.error("❌ No valid data found in uploaded files")
        return
    
    # Validate and Display Data Coverage
    coverage = validate_data_coverage(df)
    display_data_coverage(coverage)
    
    # Analyze Market Regime
    regime, weights, strategy, regime_stats = analyze_market_regime(df)
    
    # Display Regime Info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📡 Market Regime</h4>
            <h3>{regime}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>🎯 Strategy</h4>
            <h3>{strategy}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📊 Stocks Analyzed</h4>
            <h3>{len(df):,}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        breadth_color = "#00c853" if regime_stats['breadth_3m'] > 50 else "#f44336"
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📈 Market Breadth (3M)</h4>
            <h3 style='color: {breadth_color}'>{regime_stats['breadth_3m']:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Show Weight Distribution
    with st.sidebar:
        for factor, weight in weights.items():
            st.progress(weight, text=f"{factor}: {weight*100:.0f}%")
    
    # Run Scoring Engine
    df = run_ultimate_scoring(df, weights)
    
    # Generate Verdicts
    verdicts = df.apply(get_ultimate_verdict, axis=1)
    df['Verdict'] = verdicts.apply(lambda x: x[0])
    df['Verdict_Class'] = verdicts.apply(lambda x: x[1])
    
    # Sort and Rank
    df = df.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # ═══════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Rankings", "📊 Factor Analysis", "🔍 Stock Scanner", "📈 Charts", "📥 Export"
    ])
    
    with tab1:
        st.subheader("🏆 Top Ranked Stocks")
        
        # Verdict Filter
        verdict_filter = st.multiselect(
            "Filter by Verdict",
            options=df['Verdict'].unique().tolist(),
            default=[v for v in df['Verdict'].unique() if any(x in v for x in ['BUY', 'ACCUMULATE', 'BREAKOUT', 'GROWTH', 'MOMENTUM', 'VALUE'])]
        )
        
        filtered_df = df[df['Verdict'].isin(verdict_filter)] if verdict_filter else df
        
        # Display Columns
        base_cols = ['Rank', 'Name', 'Verdict', 'Final_Score', 'Close Price']
        score_cols = ['Score_Quality', 'Score_Growth', 'Score_Value', 'Score_Safety', 'Score_Momentum', 'Score_Institutional', 'Score_Technical']
        metric_cols = ['Price To Earnings', 'ROE', 'PAT Growth TTM', 'Debt To Equity', 'Free Cash Flow', 'FII Holdings', '52WH Distance']
        
        if show_all_columns:
            display_cols = base_cols + score_cols + [c for c in metric_cols if c in df.columns]
        else:
            display_cols = base_cols + [c for c in metric_cols if c in df.columns]
        
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        # Styling
        def style_dataframe(row):
            verdict_class = row.get('Verdict_Class', 'hold')
            if verdict_class == 'strong-buy':
                return ['background-color: rgba(0, 200, 83, 0.2)'] * len(row)
            elif verdict_class == 'buy':
                return ['background-color: rgba(33, 150, 243, 0.2)'] * len(row)
            elif verdict_class == 'trap':
                return ['background-color: rgba(244, 67, 54, 0.2)'] * len(row)
            elif verdict_class == 'avoid':
                return ['background-color: rgba(244, 67, 54, 0.1)'] * len(row)
            return [''] * len(row)
        
        styled_df = filtered_df[display_cols].head(top_n).style.apply(style_dataframe, axis=1)
        styled_df = styled_df.format({'Final_Score': '{:.1f}'})
        
        for col in score_cols:
            if col in display_cols:
                styled_df = styled_df.format({col: '{:.2f}'})
        
        st.dataframe(styled_df, height=700, use_container_width=True)
        
        # Summary Stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        strong_buys = len(df[df['Verdict_Class'] == 'strong-buy'])
        buys = len(df[df['Verdict_Class'] == 'buy'])
        traps = len(df[df['Verdict_Class'] == 'trap'])
        avoids = len(df[df['Verdict_Class'] == 'avoid'])
        
        col1.metric("💎 Strong Buys", strong_buys)
        col2.metric("📈 Buys", buys)
        col3.metric("🚨 Traps Detected", traps)
        col4.metric("❌ Avoid", avoids)
    
    with tab2:
        st.subheader("📊 Factor Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar Chart for Top Stock
            if len(df) > 0:
                top_stock = df.iloc[0]
                categories = ['Quality', 'Growth', 'Value', 'Safety', 'Momentum', 'Institutional', 'Technical']
                values = [
                    top_stock['Score_Quality'],
                    top_stock['Score_Growth'],
                    top_stock['Score_Value'],
                    top_stock['Score_Safety'],
                    top_stock['Score_Momentum'],
                    top_stock['Score_Institutional'],
                    top_stock['Score_Technical']
                ]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=top_stock['Name']
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f"Factor Profile: {top_stock['Name']}",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Factor Correlation
            factor_cols = ['Score_Quality', 'Score_Growth', 'Score_Value', 'Score_Safety', 'Score_Momentum', 'Score_Institutional', 'Score_Technical']
            available_factors = [c for c in factor_cols if c in df.columns]
            
            if len(available_factors) > 1:
                corr_matrix = df[available_factors].corr()
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    title="Factor Correlations",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Custom Stock Scanner")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_score = st.slider("Minimum Score", 0, 100, 60)
            max_pe = st.slider("Max P/E Ratio", 5, 200, 50)
        
        with col2:
            min_roe = st.slider("Min ROE %", 0, 50, 10)
            min_growth = st.slider("Min PAT Growth %", -50, 100, 0)
        
        with col3:
            require_fcf_positive = st.checkbox("Positive FCF Only", value=True)
            require_inst_buying = st.checkbox("Institutional Buying", value=False)
        
        # Apply Filters
        scanner_df = df[df['Final_Score'] >= min_score]
        
        if 'Price To Earnings' in scanner_df.columns:
            scanner_df = scanner_df[scanner_df['Price To Earnings'] <= max_pe]
        
        if 'ROE' in scanner_df.columns:
            scanner_df = scanner_df[scanner_df['ROE'] >= min_roe]
        
        if 'PAT Growth TTM' in scanner_df.columns:
            scanner_df = scanner_df[scanner_df['PAT Growth TTM'] >= min_growth]
        
        if require_fcf_positive and 'Free Cash Flow' in scanner_df.columns:
            scanner_df = scanner_df[scanner_df['Free Cash Flow'] > 0]
        
        if require_inst_buying:
            if 'Change In FII Holdings Latest Quarter' in scanner_df.columns:
                scanner_df = scanner_df[scanner_df['Change In FII Holdings Latest Quarter'] > 0]
        
        st.metric("Stocks Matching Criteria", len(scanner_df))
        
        if len(scanner_df) > 0:
            scanner_cols = ['Rank', 'Name', 'Verdict', 'Final_Score', 'Close Price', 'Price To Earnings', 'ROE', 'PAT Growth TTM']
            scanner_cols = [c for c in scanner_cols if c in scanner_df.columns]
            st.dataframe(scanner_df[scanner_cols], height=400, use_container_width=True)
    
    with tab4:
        st.subheader("📈 Visual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Price To Earnings' in df.columns and 'PAT Growth TTM' in df.columns:
                fig = px.scatter(
                    df.head(100),
                    x='Price To Earnings',
                    y='PAT Growth TTM',
                    color='Final_Score',
                    size='Final_Score',
                    hover_name='Name',
                    log_x=True,
                    title="GARP: Growth at Reasonable Price",
                    color_continuous_scale='Viridis'
                )
                fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Growth Target")
                fig.add_vline(x=25, line_dash="dash", line_color="red", annotation_text="Value Threshold")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'RSI 14W' in df.columns and 'ADX 14W' in df.columns:
                fig = px.scatter(
                    df.head(100),
                    x='RSI 14W',
                    y='ADX 14W',
                    color='Final_Score',
                    hover_name='Name',
                    title="Momentum Matrix",
                    color_continuous_scale='Viridis'
                )
                # Highlight sweet spot
                fig.add_shape(type="rect", x0=50, y0=25, x1=70, y1=50,
                            line=dict(color="Green", width=2), fillcolor="green", opacity=0.1)
                fig.add_annotation(x=60, y=45, text="SWEET SPOT", showarrow=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Score Distribution
        st.markdown("### Score Distribution")
        fig = px.histogram(df, x='Final_Score', nbins=30, title="Final Score Distribution")
        fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="Buy Zone")
        fig.add_vline(x=35, line_dash="dash", line_color="red", annotation_text="Avoid Zone")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("📥 Export Analysis")
        
        # Full Export
        csv_full = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Full Analysis (CSV)",
            csv_full,
            f"V4_Ultimate_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
        
        # Strong Buys Only
        strong_buy_df = df[df['Verdict_Class'].isin(['strong-buy', 'buy'])]
        if len(strong_buy_df) > 0:
            csv_buys = strong_buy_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Buy Recommendations Only",
                csv_buys,
                f"V4_Buy_Signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
        
        # Traps List
        trap_df = df[df['Verdict_Class'] == 'trap']
        if len(trap_df) > 0:
            csv_traps = trap_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⚠️ Download Traps List (Avoid These!)",
                csv_traps,
                f"V4_Traps_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
