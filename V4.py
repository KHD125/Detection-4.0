import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# 🚀 V4 ULTIMATE - INSTITUTIONAL GRADE STOCK ANALYZER
# =========================================================

st.set_page_config(
    page_title="V4 Ultimate | Stock Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 CLEAN MINIMAL UI - PROFESSIONAL STYLING
# =========================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #fafafa;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Clean header */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Metric cards - Clean design */
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    .metric-value.green { color: #10b981; }
    .metric-value.red { color: #ef4444; }
    .metric-value.blue { color: #3b82f6; }
    
    /* Verdict badges */
    .verdict-strong-buy {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .verdict-buy {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .verdict-hold {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .verdict-avoid {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .verdict-trap {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    /* Clean dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8e8e8;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #3b82f6;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
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
# 🔥 FIXED SCORING ENGINE - NO SOUP, NO PENALTY
# =========================================================
def run_ultimate_scoring(df, base_weights):
    """
    FIXED Algorithm:
    1. NO SOUP: Stocks are classified into STYLE (Value/Growth/Momentum/Balanced)
       and scored within their style - no cross-penalization
    2. DYNAMIC WEIGHTS: Each stock gets personalized weights based on its strengths
    3. NO MISSING DATA PENALTY: Uses median fallback, counts available data
    """
    n = len(df)
    
    # ═══════════════════════════════════════════════════════
    # HELPER FUNCTIONS - NO PENALTY APPROACH
    # ═══════════════════════════════════════════════════════
    
    def safe_get(col, default=None):
        """
        Get column with smart fallback - NO PENALTY for missing data.
        Returns (series, has_data_flag)
        """
        if col in df.columns:
            series = df[col].copy()
            # Use median for missing, not 0 (no penalty)
            median_val = series.median()
            if pd.isna(median_val):
                median_val = default if default is not None else 0
            filled = series.fillna(median_val)
            return filled, True
        # Column doesn't exist - use neutral default
        default_val = default if default is not None else 0
        return pd.Series([default_val] * n, index=df.index), False
    
    def smart_rank(series, lower_better=False, available=True):
        """
        Percentile ranking (0 to 1).
        If data not available, returns 0.5 (neutral) - NO PENALTY.
        """
        if not available:
            return pd.Series([0.5] * n, index=df.index)
        
        s = series.copy()
        if lower_better:
            s = -s
        ranked = s.rank(pct=True, method='average')
        # Missing values get median rank (0.5) - NO PENALTY
        return ranked.fillna(0.5)
    
    def weighted_avg(scores_weights):
        """
        Calculate weighted average, ignoring factors with no data.
        Redistributes weights among available factors.
        """
        total_weight = sum(w for _, w, avail in scores_weights if avail)
        if total_weight == 0:
            return pd.Series([0.5] * n, index=df.index)
        
        result = pd.Series([0.0] * n, index=df.index)
        for score, weight, avail in scores_weights:
            if avail:
                result += score * (weight / total_weight)
        return result
    
    # ═══════════════════════════════════════════════════════
    # EXTRACT ALL DATA WITH AVAILABILITY FLAGS
    # ═══════════════════════════════════════════════════════
    
    # Profitability
    roe, has_roe = safe_get('ROE', 12)
    roce, has_roce = safe_get('ROCE', 12)
    npm, has_npm = safe_get('NPM', 8)
    opm, has_opm = safe_get('OPM', 12)
    
    # Growth
    pat_ttm, has_pat_ttm = safe_get('PAT Growth TTM', 10)
    pat_yoy, has_pat_yoy = safe_get('PAT Growth YoY', 10)
    rev_ttm, has_rev_ttm = safe_get('Revenue Growth TTM', 10)
    rev_yoy, has_rev_yoy = safe_get('Revenue Growth YoY', 10)
    eps_ttm, has_eps_ttm = safe_get('EPS Growth TTM', 10)
    
    # Valuation
    pe, has_pe = safe_get('Price To Earnings', 25)
    ps, has_ps = safe_get('Price To Sales', 3)
    
    # Safety
    de, has_de = safe_get('Debt To Equity', 0.5)
    promoter, has_promoter = safe_get('Promoter Holdings', 50)
    fcf, has_fcf = safe_get('Free Cash Flow', 0)
    ocf, has_ocf = safe_get('Operating Cash Flow', 0)
    cash, has_cash = safe_get('Cash Equivalents', 0)
    debt, has_debt = safe_get('Debt', 1)
    
    # Momentum
    rsi_w, has_rsi_w = safe_get('RSI 14W', 50)
    adx_w, has_adx_w = safe_get('ADX 14W', 25)
    ret_1m, has_ret_1m = safe_get('Returns 1M', 0)
    ret_3m, has_ret_3m = safe_get('Returns 3M', 0)
    ret_6m, has_ret_6m = safe_get('Returns 6M', 0)
    ret_1y, has_ret_1y = safe_get('Returns 1Y', 0)
    
    # Institutional
    fii, has_fii = safe_get('FII Holdings', 5)
    dii, has_dii = safe_get('DII Holdings', 10)
    fii_chg, has_fii_chg = safe_get('Change In FII Holdings Latest Quarter', 0)
    dii_chg, has_dii_chg = safe_get('Change In DII Holdings Latest Quarter', 0)
    prom_chg, has_prom_chg = safe_get('Change In Promoter Holdings Latest Quarter', 0)
    
    # Technical
    dist_52wh, has_52wh = safe_get('52WH Distance', -15)
    ret_vs_nifty, has_vs_nifty = safe_get('Returns Vs Nifty 500 3M', 0)
    ret_vs_ind, has_vs_ind = safe_get('Returns Vs Industry 3M', 0)
    
    # ═══════════════════════════════════════════════════════
    # STEP 1: CLASSIFY STOCK STYLE (Avoid the "Soup" Problem)
    # ═══════════════════════════════════════════════════════
    
    # Value indicators: Low PE, Low PS
    pe_capped = pe.clip(1, 200)
    is_cheap = (pe_capped < pe_capped.median())
    
    # Growth indicators: High earnings/revenue growth
    is_growing = (pat_ttm > pat_ttm.median()) | (rev_ttm > rev_ttm.median())
    
    # Momentum indicators: High returns, strong RSI
    is_momentum = (ret_3m > ret_3m.median()) & (rsi_w > 50)
    
    # Classify each stock's style
    # 1 = Value, 2 = Growth, 3 = Momentum, 4 = Balanced (GARP)
    stock_style = pd.Series([4] * n, index=df.index)  # Default: Balanced
    
    # Pure Value: Cheap but not growing fast or momentum
    stock_style = stock_style.where(~(is_cheap & ~is_growing & ~is_momentum), 1)
    
    # Pure Growth: Growing but not cheap
    stock_style = stock_style.where(~(is_growing & ~is_cheap & ~is_momentum), 2)
    
    # Pure Momentum: High momentum regardless of value
    stock_style = stock_style.where(~(is_momentum & ~is_cheap), 3)
    
    # GARP (Best): Cheap AND Growing
    stock_style = stock_style.where(~(is_cheap & is_growing), 4)
    
    df['Stock_Style'] = stock_style.map({1: 'Value', 2: 'Growth', 3: 'Momentum', 4: 'GARP'})
    
    # ═══════════════════════════════════════════════════════
    # STEP 2: CALCULATE FACTOR SCORES (No cross-contamination)
    # ═══════════════════════════════════════════════════════
    
    # QUALITY SCORE - Same for all styles
    quality_components = [
        (smart_rank(roe, available=has_roe), 0.30, has_roe),
        (smart_rank(roce, available=has_roce), 0.25, has_roce),
        (smart_rank(npm, available=has_npm), 0.25, has_npm),
        (smart_rank(opm, available=has_opm), 0.20, has_opm),
    ]
    df['Score_Quality'] = weighted_avg(quality_components)
    
    # GROWTH SCORE
    growth_components = [
        (smart_rank(pat_ttm, available=has_pat_ttm), 0.35, has_pat_ttm),
        (smart_rank(rev_ttm, available=has_rev_ttm), 0.30, has_rev_ttm),
        (smart_rank(eps_ttm, available=has_eps_ttm), 0.20, has_eps_ttm),
        (smart_rank(pat_yoy, available=has_pat_yoy), 0.15, has_pat_yoy),
    ]
    df['Score_Growth'] = weighted_avg(growth_components)
    
    # VALUE SCORE (Lower PE/PS = Better)
    value_components = [
        (smart_rank(pe_capped, lower_better=True, available=has_pe), 0.60, has_pe),
        (smart_rank(ps.clip(0.1, 50), lower_better=True, available=has_ps), 0.40, has_ps),
    ]
    df['Score_Value'] = weighted_avg(value_components)
    
    # SAFETY SCORE
    de_capped = de.clip(0, 5)
    safety_components = [
        (smart_rank(de_capped, lower_better=True, available=has_de), 0.30, has_de),
        (smart_rank(promoter, available=has_promoter), 0.25, has_promoter),
        (smart_rank(fcf, available=has_fcf), 0.25, has_fcf),
        (smart_rank(ocf, available=has_ocf), 0.20, has_ocf),
    ]
    df['Score_Safety'] = weighted_avg(safety_components)
    
    # MOMENTUM SCORE
    momentum_components = [
        (smart_rank(ret_3m, available=has_ret_3m), 0.30, has_ret_3m),
        (smart_rank(ret_6m, available=has_ret_6m), 0.25, has_ret_6m),
        (smart_rank(ret_1m, available=has_ret_1m), 0.20, has_ret_1m),
        (smart_rank(ret_1y, available=has_ret_1y), 0.15, has_ret_1y),
        (smart_rank(adx_w, available=has_adx_w), 0.10, has_adx_w),
    ]
    df['Score_Momentum'] = weighted_avg(momentum_components)
    
    # INSTITUTIONAL SCORE
    total_inst = fii + dii
    has_total_inst = has_fii or has_dii
    inst_components = [
        (smart_rank(fii_chg, available=has_fii_chg), 0.35, has_fii_chg),
        (smart_rank(dii_chg, available=has_dii_chg), 0.30, has_dii_chg),
        (smart_rank(total_inst, available=has_total_inst), 0.20, has_total_inst),
        (smart_rank(prom_chg, available=has_prom_chg), 0.15, has_prom_chg),
    ]
    df['Score_Institutional'] = weighted_avg(inst_components)
    
    # TECHNICAL SCORE
    tech_components = [
        (smart_rank(dist_52wh, available=has_52wh), 0.35, has_52wh),
        (smart_rank(ret_vs_nifty, available=has_vs_nifty), 0.30, has_vs_nifty),
        (smart_rank(ret_vs_ind, available=has_vs_ind), 0.25, has_vs_ind),
        (smart_rank(adx_w, available=has_adx_w), 0.10, has_adx_w),
    ]
    df['Score_Technical'] = weighted_avg(tech_components)
    
    # ═══════════════════════════════════════════════════════
    # STEP 3: DYNAMIC WEIGHTS BASED ON STOCK STYLE
    # ═══════════════════════════════════════════════════════
    
    def get_dynamic_weights(style, base_w):
        """
        Adjust weights based on stock's natural style.
        Value stocks weighted more on Value, less on Momentum.
        Growth stocks weighted more on Growth, less on Value.
        This PREVENTS the soup problem!
        """
        if style == 1:  # Value Style
            return {
                'Quality': base_w['Quality'] * 1.0,
                'Growth': base_w['Growth'] * 0.7,
                'Value': base_w['Value'] * 1.5,      # Boost Value
                'Safety': base_w['Safety'] * 1.2,
                'Momentum': base_w['Momentum'] * 0.5,  # Reduce Momentum penalty
                'Institutional': base_w['Institutional'] * 1.0,
                'Technical': base_w['Technical'] * 0.6,
            }
        elif style == 2:  # Growth Style
            return {
                'Quality': base_w['Quality'] * 1.0,
                'Growth': base_w['Growth'] * 1.5,    # Boost Growth
                'Value': base_w['Value'] * 0.5,      # Reduce Value penalty
                'Safety': base_w['Safety'] * 0.8,
                'Momentum': base_w['Momentum'] * 1.2,
                'Institutional': base_w['Institutional'] * 1.0,
                'Technical': base_w['Technical'] * 1.0,
            }
        elif style == 3:  # Momentum Style
            return {
                'Quality': base_w['Quality'] * 0.8,
                'Growth': base_w['Growth'] * 1.0,
                'Value': base_w['Value'] * 0.3,      # Don't penalize for being expensive
                'Safety': base_w['Safety'] * 0.7,
                'Momentum': base_w['Momentum'] * 1.5,  # Boost Momentum
                'Institutional': base_w['Institutional'] * 1.2,
                'Technical': base_w['Technical'] * 1.3,
            }
        else:  # GARP / Balanced (style == 4)
            return {
                'Quality': base_w['Quality'] * 1.2,
                'Growth': base_w['Growth'] * 1.2,
                'Value': base_w['Value'] * 1.2,
                'Safety': base_w['Safety'] * 1.0,
                'Momentum': base_w['Momentum'] * 0.8,
                'Institutional': base_w['Institutional'] * 1.0,
                'Technical': base_w['Technical'] * 0.8,
            }
    
    # Calculate final score with dynamic weights per stock
    final_scores = []
    for idx in df.index:
        style = stock_style[idx]
        w = get_dynamic_weights(style, base_weights)
        
        # Normalize weights to sum to 1
        total_w = sum(w.values())
        
        score = (
            df.loc[idx, 'Score_Quality'] * w['Quality'] / total_w +
            df.loc[idx, 'Score_Growth'] * w['Growth'] / total_w +
            df.loc[idx, 'Score_Value'] * w['Value'] / total_w +
            df.loc[idx, 'Score_Safety'] * w['Safety'] / total_w +
            df.loc[idx, 'Score_Momentum'] * w['Momentum'] / total_w +
            df.loc[idx, 'Score_Institutional'] * w['Institutional'] / total_w +
            df.loc[idx, 'Score_Technical'] * w['Technical'] / total_w
        ) * 100
        
        final_scores.append(score)
    
    df['Final_Score'] = final_scores
    df['Final_Score'] = df['Final_Score'].clip(0, 100)
    
    # ═══════════════════════════════════════════════════════
    # DATA AVAILABILITY TRACKING (For transparency, not penalty)
    # ═══════════════════════════════════════════════════════
    total_expected = 25  # Total columns we try to use
    data_available = sum([
        has_roe, has_roce, has_npm, has_opm,
        has_pat_ttm, has_pat_yoy, has_rev_ttm, has_rev_yoy, has_eps_ttm,
        has_pe, has_ps,
        has_de, has_promoter, has_fcf, has_ocf, has_cash, has_debt,
        has_rsi_w, has_adx_w, has_ret_1m, has_ret_3m, has_ret_6m, has_ret_1y,
        has_fii, has_dii, has_fii_chg, has_dii_chg, has_prom_chg,
        has_52wh, has_vs_nifty, has_vs_ind
    ])
    df['Data_Coverage'] = f"{data_available}/{total_expected}"
    
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
# 📊 MAIN DASHBOARD - CLEAN UI
# =========================================================
def main():
    # Clean Header
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem 0;'>
        <h1 style='margin:0; font-weight:700; color:#1a1a2e;'>V4 Stock Analyzer</h1>
        <p style='color:#666; margin:0.3rem 0 0 0; font-size:0.95rem;'>7-Factor Scoring • Market Regime Detection • Trap Identification</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 📂 Upload Data")
        uploaded_files = st.file_uploader(
            "CSV Files",
            accept_multiple_files=True,
            type=['csv'],
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.success(f"✓ {len(uploaded_files)} files loaded")
        
        st.markdown("---")
        st.markdown("### Settings")
        
        show_all_columns = st.checkbox("Show Factor Scores", value=False)
        top_n = st.slider("Display Top", 10, 100, 30, help="Number of stocks to show")
    
    if not uploaded_files:
        # Simple welcome message
        st.markdown("""
        <div style='text-align:center; padding:3rem; background:#f8f9fa; border-radius:12px; margin:2rem 0;'>
            <h3 style='color:#333; margin-bottom:0.5rem;'>Upload Your CSV Files to Start</h3>
            <p style='color:#666;'>Drag and drop your stock data files in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Expected columns in simple format
        with st.expander("📋 Expected Data Columns"):
            st.markdown("""
            **Required columns across your CSV files:**
            
            | Category | Columns |
            |----------|---------|
            | **Identity** | companyId, Name, Industry |
            | **Price** | Close Price, Market Cap, 52WH Distance |
            | **Returns** | Returns 1D/1W/1M/3M/6M/1Y/3Y/5Y |
            | **Technical** | RSI 14D/14W, ADX 14D/14W |
            | **Valuation** | Price To Earnings, Price To Sales, Debt To Equity |
            | **Holdings** | FII/DII/Retail/Promoter Holdings + Changes |
            | **Growth** | PAT/Revenue Growth (YoY/QoQ/TTM), EPS Growth |
            | **Cash Flow** | Operating/Free/Net Cash Flow |
            | **Profitability** | ROE, ROCE, NPM, OPM |
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
    
    # Clean summary bar
    st.markdown(f"""
    <div style='display:grid; grid-template-columns:repeat(4, 1fr); gap:1rem; margin:0.5rem 0 1.5rem 0;'>
        <div class='metric-box'><div class='metric-label'>Market Regime</div><div class='metric-value'>{regime}</div></div>
        <div class='metric-box'><div class='metric-label'>Strategy</div><div class='metric-value' style='font-size:1rem;'>{strategy}</div></div>
        <div class='metric-box'><div class='metric-label'>Stocks</div><div class='metric-value'>{len(df):,}</div></div>
        <div class='metric-box'><div class='metric-label'>Breadth 3M</div><div class='metric-value' style='color:{"#00c853" if regime_stats["breadth_3m"] > 50 else "#f44336"}'>{regime_stats["breadth_3m"]:.0f}%</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show weights in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Factor Weights")
        for factor, weight in weights.items():
            st.markdown(f"**{factor}**: {weight*100:.0f}%")
            st.progress(weight)
    
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
    # TABS - CLEAN LAYOUT
    # ═══════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Rankings", " Scanner", "� Charts", "📥 Export"
    ])
    
    with tab1:
        # Style and Verdict Filters
        col_filter1, col_filter2 = st.columns([1, 2])
        with col_filter1:
            style_options = df['Stock_Style'].unique().tolist()
            style_filter = st.multiselect("Stock Style", options=style_options, default=style_options)
        
        with col_filter2:
            verdict_options = df['Verdict'].unique().tolist()
            buy_verdicts = [v for v in verdict_options if any(x in v for x in ['BUY', 'ACCUMULATE', 'BREAKOUT', 'GROWTH', 'MOMENTUM', 'VALUE', 'QUALITY', 'OUTPERFORMER', 'TURNAROUND'])]
            verdict_filter = st.multiselect("Verdict", options=verdict_options, default=buy_verdicts if buy_verdicts else None)
        
        # Apply both filters
        filtered_df = df.copy()
        if style_filter:
            filtered_df = filtered_df[filtered_df['Stock_Style'].isin(style_filter)]
        if verdict_filter:
            filtered_df = filtered_df[filtered_df['Verdict'].isin(verdict_filter)]
        
        # Display Columns - Include Stock_Style
        base_cols = ['Rank', 'Name', 'Stock_Style', 'Verdict', 'Final_Score']
        price_cols = ['Close Price', 'Price To Earnings']
        metric_cols = ['ROE', 'PAT Growth TTM', 'Debt To Equity', 'Free Cash Flow']
        score_cols = ['Score_Quality', 'Score_Growth', 'Score_Value', 'Score_Safety', 'Score_Momentum', 'Score_Institutional', 'Score_Technical']
        
        if show_all_columns:
            display_cols = base_cols + score_cols
        else:
            display_cols = base_cols + [c for c in price_cols + metric_cols if c in df.columns]
        
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        # Color styling based on verdict
        def color_row(row):
            verdict_class = row.get('Verdict_Class', 'hold')
            colors = {
                'strong-buy': 'background-color: rgba(0, 200, 83, 0.15)',
                'buy': 'background-color: rgba(33, 150, 243, 0.12)',
                'trap': 'background-color: rgba(244, 67, 54, 0.15)',
                'avoid': 'background-color: rgba(244, 67, 54, 0.08)'
            }
            color = colors.get(verdict_class, '')
            return [color] * len(row)
        
        styled_df = filtered_df[display_cols].head(top_n).style.apply(color_row, axis=1)
        styled_df = styled_df.format({'Final_Score': '{:.1f}'})
        for col in score_cols:
            if col in display_cols:
                styled_df = styled_df.format({col: '{:.2f}'})
        
        st.dataframe(styled_df, height=600, use_container_width=True)
        
        # Quick Stats with style breakdown
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💎 Strong Buy", len(df[df['Verdict_Class'] == 'strong-buy']))
        col2.metric("📈 Buy", len(df[df['Verdict_Class'] == 'buy']))
        col3.metric("🚨 Traps", len(df[df['Verdict_Class'] == 'trap']))
        col4.metric("📊 GARP", len(df[df['Stock_Style'] == 'GARP']))
        col5.metric("💰 Value", len(df[df['Stock_Style'] == 'Value']))
    
    with tab2:
        st.markdown("#### Custom Scanner")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_score = st.slider("Min Score", 0, 100, 60)
            max_pe = st.slider("Max P/E", 5, 200, 50)
        
        with col2:
            min_roe = st.slider("Min ROE %", 0, 50, 10)
            min_growth = st.slider("Min Growth %", -50, 100, 0)
        
        with col3:
            require_fcf_positive = st.checkbox("Positive FCF", value=True)
            require_inst_buying = st.checkbox("Inst. Buying", value=False)
        
        # Apply Filters
        scanner_df = df[df['Final_Score'] >= min_score].copy()
        
        if 'Price To Earnings' in scanner_df.columns:
            scanner_df = scanner_df[scanner_df['Price To Earnings'] <= max_pe]
        
        if 'ROE' in scanner_df.columns:
            scanner_df = scanner_df[scanner_df['ROE'] >= min_roe]
        
        if 'PAT Growth TTM' in scanner_df.columns:
            scanner_df = scanner_df[scanner_df['PAT Growth TTM'] >= min_growth]
        
        if require_fcf_positive and 'Free Cash Flow' in scanner_df.columns:
            scanner_df = scanner_df[scanner_df['Free Cash Flow'] > 0]
        
        if require_inst_buying and 'Change In FII Holdings Latest Quarter' in scanner_df.columns:
            scanner_df = scanner_df[scanner_df['Change In FII Holdings Latest Quarter'] > 0]
        
        st.metric("Matches", len(scanner_df))
        
        if len(scanner_df) > 0:
            scanner_cols = ['Rank', 'Name', 'Verdict', 'Final_Score', 'Close Price', 'Price To Earnings', 'ROE', 'PAT Growth TTM']
            scanner_cols = [c for c in scanner_cols if c in scanner_df.columns]
            st.dataframe(scanner_df[scanner_cols], height=400, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # GARP Chart
            if 'Price To Earnings' in df.columns and 'PAT Growth TTM' in df.columns:
                fig = px.scatter(
                    df.head(100),
                    x='Price To Earnings',
                    y='PAT Growth TTM',
                    color='Final_Score',
                    size='Final_Score',
                    hover_name='Name',
                    log_x=True,
                    title="Growth at Reasonable Price",
                    color_continuous_scale='Viridis',
                    height=350
                )
                fig.add_hline(y=20, line_dash="dash", line_color="green")
                fig.add_vline(x=25, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Score Distribution
            fig = px.histogram(df, x='Final_Score', nbins=25, title="Score Distribution", height=300)
            fig.add_vline(x=70, line_dash="dash", line_color="green")
            fig.add_vline(x=35, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar Chart for Top Stock
            if len(df) > 0:
                top_stock = df.iloc[0]
                categories = ['Quality', 'Growth', 'Value', 'Safety', 'Momentum', 'Institutional', 'Technical']
                values = [
                    top_stock.get('Score_Quality', 0.5),
                    top_stock.get('Score_Growth', 0.5),
                    top_stock.get('Score_Value', 0.5),
                    top_stock.get('Score_Safety', 0.5),
                    top_stock.get('Score_Momentum', 0.5),
                    top_stock.get('Score_Institutional', 0.5),
                    top_stock.get('Score_Technical', 0.5)
                ]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=top_stock.get('Name', 'Top Stock')
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f"Top Stock Profile: {top_stock.get('Name', 'N/A')}",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Momentum Matrix
            if 'RSI 14W' in df.columns and 'ADX 14W' in df.columns:
                fig = px.scatter(
                    df.head(80),
                    x='RSI 14W',
                    y='ADX 14W',
                    color='Final_Score',
                    hover_name='Name',
                    title="Momentum Matrix",
                    color_continuous_scale='Viridis',
                    height=300
                )
                fig.add_shape(type="rect", x0=50, y0=25, x1=70, y1=50,
                            line=dict(color="Green"), fillcolor="green", opacity=0.1)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2, col3 = st.columns(3)
        
        # Full Export
        csv_full = df.to_csv(index=False).encode('utf-8')
        col1.download_button(
            "📥 Full Analysis",
            csv_full,
            f"V4_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
        
        # Buy Signals Only
        buy_df = df[df['Verdict_Class'].isin(['strong-buy', 'buy'])]
        if len(buy_df) > 0:
            csv_buys = buy_df.to_csv(index=False).encode('utf-8')
            col2.download_button(
                "� Buy Signals",
                csv_buys,
                f"V4_Buys_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        # Traps
        trap_df = df[df['Verdict_Class'] == 'trap']
        if len(trap_df) > 0:
            csv_traps = trap_df.to_csv(index=False).encode('utf-8')
            col3.download_button(
                "⚠️ Traps List",
                csv_traps,
                f"V4_Traps_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
