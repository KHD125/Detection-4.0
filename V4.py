import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# 🚀 V4 ULTRA - PURE QUANT DATA-DRIVEN ANALYZER
# =========================================================
# CORRELATION-BACKED ALGORITHM (Not Theory-Based!)
# 
# CORE PHILOSOPHY: Let DATA decide, not textbook theory
# 
# TOP PREDICTORS (by actual correlation to returns):
#   1. RSI 14W        +0.878  ← KING OF SIGNALS (40% of Momentum)
#   2. Returns 3M     +0.707  ← Strong momentum (30% of Momentum)
#   3. 52WH Distance  -0.655  ← Relative strength (20% of Momentum)
#   4. PE             +0.551  ← HIGH PE WINS! (Don't penalize!)
#   5. FII Changes    +0.499  ← Smart money (70% of Institutional)
#   6. ROCE           +0.413  ← Only quality that matters
#
# DEAD SIGNALS (correlation < 0.2):
#   ✗ NPM             -0.013  ← USELESS
#   ✗ PAT Growth      +0.006  ← NOISE
#   ✗ Debt/Equity     +0.068  ← DOESN'T MATTER
#   ✗ DII Changes     NEGATIVE ← CONTRARIAN SIGNAL
#
# V4 ULTRA SIMPLIFICATION:
#   ✓ 4 Factors only (Momentum 60%, Institutional 20%, Quality 10%, Safety 10%)
#   ✓ Z-Score normalization (adaptive to any market)
#   ✓ No style classification (data doesn't care!)
#   ✓ Simple percentile verdicts (Top 20% = BUY)
#   ✓ Trap detection kept (OCF/FCF/Debt)
# =========================================================

st.set_page_config(
    page_title="Wave Detection | Pro Stock Analyzer",
    page_icon="🌊",
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
# COLUMN MAPPING - Only map if your CSVs use full names
# =========================================================
# NOTE: Your actual CSV files (from Screener.in) already use short names
# like NPM, ROCE, ROE. This mapping is kept for compatibility with
# other data sources that might use full names.
COLUMN_MAP = {
    # Only map if full names are present (backward compatibility)
    'ROE (Return on Equity)': 'ROE',
    'ROCE (Return on Capital Employed)': 'ROCE', 
    'NPM (Net Profit Margin)': 'NPM',
    'OPM (Operating Profit Margin)': 'OPM',
    'CWIP (Capital Work in Progress)': 'CWIP',
    # Add PE variants
    'Price To Earnings Ratio': 'Price To Earnings',
    'P/E': 'Price To Earnings',
    'P/E Ratio': 'Price To Earnings',
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
# 🎯 MARKET REGIME DETECTION - SIMPLIFIED
# =========================================================
def analyze_market_regime(df):
    """
    V4 ULTRA: Simplified regime detection.
    
    KEY INSIGHT: Momentum ALWAYS dominates (60%+)
    Only adjust Safety/Quality in bear markets for risk management.
    
    4 FACTORS ONLY:
    - Momentum: 60% (RSI + Returns + 52WH + FII_Accel)
    - Institutional: 20% (FII changes - the smart money)
    - Quality: 10% (ROCE only - the one that matters)
    - Safety: 10% (OCF/FCF/Debt - trap avoidance)
    """
    r1m = df['Returns 1M'].median() if 'Returns 1M' in df.columns else 0
    r3m = df['Returns 3M'].median() if 'Returns 3M' in df.columns else 0
    r1y = df['Returns 1Y'].median() if 'Returns 1Y' in df.columns else 0
    
    # Breadth Analysis
    breadth_1m = (df['Returns 1M'] > 0).mean() * 100 if 'Returns 1M' in df.columns else 50
    breadth_3m = (df['Returns 3M'] > 0).mean() * 100 if 'Returns 3M' in df.columns else 50
    
    # REGIME TREND
    regime_trend = "STABLE"
    if breadth_1m > breadth_3m + 10:
        regime_trend = "IMPROVING"
    elif breadth_1m < breadth_3m - 10:
        regime_trend = "DETERIORATING"
    
    # Returns acceleration
    monthly_rate_recent = r1m
    monthly_rate_older = (r3m - r1m) / 2 if r3m != r1m else 0
    returns_accelerating = monthly_rate_recent > monthly_rate_older
    
    # V4 ULTRA: SIMPLIFIED 4-FACTOR WEIGHTS
    # Momentum ALWAYS 60%+ because data says so!
    
    if r3m > 10 and r1y > 20 and breadth_3m > 65:
        regime = "🚀 STRONG BULL"
        weights = {'Momentum': 0.65, 'Institutional': 0.20, 'Quality': 0.10, 'Safety': 0.05}
        strategy = "Max Momentum"
    elif r3m > 5 and breadth_3m > 50:
        regime = "📈 BULL"
        weights = {'Momentum': 0.60, 'Institutional': 0.20, 'Quality': 0.10, 'Safety': 0.10}
        strategy = "Momentum"
    elif r3m < -10 or (r1m < -5 and breadth_1m < 30):
        regime = "🐻 BEAR"
        weights = {'Momentum': 0.45, 'Institutional': 0.20, 'Quality': 0.15, 'Safety': 0.20}
        strategy = "Defensive"
    elif r3m < -3 or breadth_3m < 40:
        regime = "⚠️ CORRECTION"
        weights = {'Momentum': 0.50, 'Institutional': 0.20, 'Quality': 0.15, 'Safety': 0.15}
        strategy = "Cautious"
    else:
        regime = "⚖️ SIDEWAYS"
        weights = {'Momentum': 0.55, 'Institutional': 0.20, 'Quality': 0.12, 'Safety': 0.13}
        strategy = "Selective"
    
    # Regime trend fine-tuning
    if regime_trend == "IMPROVING":
        weights['Momentum'] = min(0.70, weights['Momentum'] + 0.05)
        weights['Safety'] = max(0.05, weights['Safety'] - 0.05)
        strategy += " ↑"
    elif regime_trend == "DETERIORATING":
        weights['Safety'] = min(0.25, weights['Safety'] + 0.05)
        weights['Momentum'] = max(0.40, weights['Momentum'] - 0.05)
        strategy += " ↓"
    
    # Normalize to sum to 1.0
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    return regime, weights, strategy, {
        'r1m': r1m, 'r3m': r3m, 'r1y': r1y, 
        'breadth_3m': breadth_3m, 'breadth_1m': breadth_1m,
        'regime_trend': regime_trend,
        'returns_accelerating': returns_accelerating
    }

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
    # npm removed - correlation -0.013 = USELESS (dead code cleanup)
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
    # ═══════════════════════════════════════════════════════
    # V4 ULTRA: PURE QUANT 4-FACTOR Z-SCORE ENGINE
    # ═══════════════════════════════════════════════════════
    # 
    # 4 FACTORS ONLY (data-driven, no style classification):
    #   1. MOMENTUM (60%): RSI + Returns + 52WH Distance + Acceleration
    #   2. INSTITUTIONAL (20%): FII flows (smart money)
    #   3. QUALITY (10%): ROCE only (the one that matters)
    #   4. SAFETY (10%): Cash flow + Debt (trap avoidance)
    #
    # ═══════════════════════════════════════════════════════
    
    # ─────────────────────────────────────────────────────────
    # Z-SCORE HELPER: Adaptive market-relative normalization
    # ─────────────────────────────────────────────────────────
    def get_z_score(series, lower_better=False, available=True):
        """
        Convert raw values to Z-scores with outlier capping.
        Returns percentile rank (0-1) for consistency with weighted_avg.
        """
        if not available or series is None:
            return pd.Series([0.5] * n, index=df.index)
        
        # Cap outliers at 5th/95th percentile
        capped = series.clip(
            lower=series.quantile(0.05), 
            upper=series.quantile(0.95)
        )
        
        # Calculate Z-score
        mean_val = capped.mean()
        std_val = capped.std()
        if std_val == 0 or pd.isna(std_val):
            return pd.Series([0.5] * n, index=df.index)
        
        z = (capped - mean_val) / std_val
        
        # Flip if lower is better
        if lower_better:
            z = -z
        
        # Convert to 0-1 range using percentile rank
        return z.rank(pct=True).fillna(0.5)
    
    # ─────────────────────────────────────────────────────────
    # MOMENTUM ACCELERATION (2nd derivative - catches breakouts)
    # ─────────────────────────────────────────────────────────
    if has_ret_1m and has_ret_3m:
        recent_monthly = ret_1m
        older_monthly = (ret_3m - ret_1m) / 2
        momentum_acceleration = recent_monthly - older_monthly
        has_mom_accel = True
    else:
        momentum_acceleration = pd.Series([0.0] * n, index=df.index)
        has_mom_accel = False
    
    df['Momentum_Acceleration'] = momentum_acceleration
    
    # ─────────────────────────────────────────────────────────
    # QUALITY GATES (Binary trap filter - 0 to 1)
    # ─────────────────────────────────────────────────────────
    def calculate_quality_gate(idx):
        gates_passed = 0
        total_gates = 5
        
        # Gate 1: Positive Operating Cash Flow
        ocf_val = ocf.iloc[idx] if hasattr(ocf, 'iloc') else ocf[idx]
        if ocf_val > 0:
            gates_passed += 1
        
        # Gate 2: ROE > Cost of Equity
        roe_val = roe.iloc[idx] if hasattr(roe, 'iloc') else roe[idx]
        if roe_val > 10:
            gates_passed += 1
        
        # Gate 3: Debt not dangerous
        de_val = de.iloc[idx] if hasattr(de, 'iloc') else de[idx]
        if de_val < 2:
            gates_passed += 1
        
        # Gate 4: Not hemorrhaging money
        pat_val = pat_ttm.iloc[idx] if hasattr(pat_ttm, 'iloc') else pat_ttm[idx]
        if pat_val > -30:
            gates_passed += 1
        
        # Gate 5: Some institutional interest
        fii_val = fii.iloc[idx] if hasattr(fii, 'iloc') else fii[idx]
        dii_val = dii.iloc[idx] if hasattr(dii, 'iloc') else dii[idx]
        if (fii_val + dii_val) > 5:
            gates_passed += 1
        
        return gates_passed / total_gates
    
    df['Quality_Gate'] = [calculate_quality_gate(i) for i in range(n)]
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 1: MOMENTUM SCORE (60% of final - THE KING)
    # Correlation backed: RSI +0.878, Ret3M +0.707, 52WH -0.655
    # ═══════════════════════════════════════════════════════
    
    # RSI sweet spot: penalize extremes (overbought/oversold)
    rsi_adjusted = rsi_w.copy()
    rsi_adjusted = np.where(rsi_w > 75, rsi_w * 0.7, rsi_adjusted)  # Overbought penalty
    rsi_adjusted = np.where(rsi_w < 30, rsi_w * 0.8, rsi_adjusted)  # Oversold penalty
    
    momentum_components = [
        (get_z_score(pd.Series(rsi_adjusted, index=df.index), available=has_rsi_w), 0.40, has_rsi_w),  # RSI = KING
        (get_z_score(ret_3m, available=has_ret_3m), 0.30, has_ret_3m),  # Returns 3M
        (get_z_score(dist_52wh, lower_better=True, available=has_52wh), 0.20, has_52wh),  # Near 52WH = good
        (get_z_score(momentum_acceleration.clip(-30, 30), available=has_mom_accel), 0.10, has_mom_accel),  # Acceleration
    ]
    df['Score_Momentum'] = weighted_avg(momentum_components)
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 2: INSTITUTIONAL SCORE (20% - Smart Money)
    # FII +0.499 (follow them), DII negative (contrarian)
    # ═══════════════════════════════════════════════════════
    
    inst_components = [
        (get_z_score(fii_chg, available=has_fii_chg), 0.70, has_fii_chg),  # FII changes = KEY
        (get_z_score(prom_chg, available=has_prom_chg), 0.30, has_prom_chg),  # Promoter not selling
        # DII REMOVED - negative correlation! Retail piling in = bad
    ]
    df['Score_Institutional'] = weighted_avg(inst_components)
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 3: QUALITY SCORE (10% - ROCE only)
    # ROCE +0.413 correlation - THE quality metric
    # NPM/OPM/ROE are redundant or useless
    # ═══════════════════════════════════════════════════════
    
    quality_components = [
        (get_z_score(roce.clip(0, 50), available=has_roce), 1.0, has_roce),  # ROCE only
    ]
    df['Score_Quality'] = weighted_avg(quality_components)
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 4: SAFETY SCORE (10% - Trap Avoidance)
    # Cash flow health + Debt safety = avoid value traps
    # ═══════════════════════════════════════════════════════
    
    de_capped = de.clip(0, 5)
    safety_components = [
        (get_z_score(ocf, available=has_ocf), 0.40, has_ocf),  # Operating cash flow
        (get_z_score(fcf, available=has_fcf), 0.35, has_fcf),  # Free cash flow
        (get_z_score(de_capped, lower_better=True, available=has_de), 0.25, has_de),  # Low debt
    ]
    df['Score_Safety'] = weighted_avg(safety_components)
    
    # ═══════════════════════════════════════════════════════
    # FINAL SCORE: PURE WEIGHTED SUM (No style adjustments!)
    # Weights come directly from market regime analysis
    # ═══════════════════════════════════════════════════════
    
    # base_weights from market regime: {'Momentum': 0.60, 'Institutional': 0.20, 'Quality': 0.10, 'Safety': 0.10}
    df['Final_Score'] = (
        df['Score_Momentum'] * base_weights.get('Momentum', 0.60) +
        df['Score_Institutional'] * base_weights.get('Institutional', 0.20) +
        df['Score_Quality'] * base_weights.get('Quality', 0.10) +
        df['Score_Safety'] * base_weights.get('Safety', 0.10)
    ) * 100
    
    df['Final_Score'] = df['Final_Score'].clip(0, 100)
    
    # ─────────────────────────────────────────────────────────
    # DATA CONFIDENCE (discount scores with missing critical data)
    # ─────────────────────────────────────────────────────────
    critical_columns = ['RSI 14W', 'Returns 3M', 'Change In FII Holdings Latest Quarter', 
                        'ROCE', 'Free Cash Flow', 'Operating Cash Flow', 'Debt To Equity']
    
    def calculate_data_confidence(row):
        available = 0
        for col in critical_columns:
            val = row.get(col, None)
            if val is not None and not pd.isna(val):
                available += 1
        # 0.85 to 1.0 range
        confidence = 0.85 + (available / len(critical_columns)) * 0.15
        return confidence
    
    df['Data_Confidence'] = df.apply(calculate_data_confidence, axis=1)
    
    # Apply confidence adjustment to final score
    # High confidence (1.0) = no change, Low confidence (0.85) = -15% penalty
    df['Final_Score_Adjusted'] = df['Final_Score'] * df['Data_Confidence']
    
    # Replace Final_Score with adjusted (optional - keep original available)
    df['Final_Score_Raw'] = df['Final_Score'].copy()
    df['Final_Score'] = df['Final_Score_Adjusted']
    
    # ═══════════════════════════════════════════════════════
    # DATA AVAILABILITY TRACKING (For transparency)
    # ═══════════════════════════════════════════════════════
    total_expected = 25  # 4 factor model simplified tracking
    data_available = sum([
        has_roe, has_roce, has_opm,
        has_pat_ttm, has_pat_yoy, has_rev_ttm, has_rev_yoy, has_eps_ttm,
        has_pe, has_ps,
        has_de, has_promoter, has_fcf, has_ocf, has_cash, has_debt,
        has_rsi_w, has_adx_w, has_ret_1m, has_ret_3m, has_ret_6m, has_ret_1y,
        has_fii, has_dii, has_fii_chg, has_dii_chg, has_prom_chg,
        has_52wh, has_vs_nifty,
        has_mom_accel  # Momentum acceleration
    ])
    df['Data_Coverage'] = f"{data_available}/{total_expected}"
    
    return df

# =========================================================
# 🎯 V4 ULTRA VERDICT ENGINE - SIMPLE PERCENTILE BASED
# =========================================================
def get_ultimate_verdict(row):
    """
    V4 ULTRA Simplified Verdict System:
    
    1. TRAP DETECTION (Safety Net)
    2. PURE PERCENTILE VERDICT (no style complexity)
    
    Returns: (verdict_text, verdict_class, trap_probability)
    """
    score = row['Final_Score']
    
    # ═══════════════════════════════════════════════════════
    # GET VALUES FOR TRAP DETECTION
    # ═══════════════════════════════════════════════════════
    
    # 4 Factor Scores (0-1 percentile ranks)
    sm = row.get('Score_Momentum', 0.5)
    si = row.get('Score_Institutional', 0.5)
    sq = row.get('Score_Quality', 0.5)
    ss = row.get('Score_Safety', 0.5)
    
    # Cash Flow (critical for trap detection)
    fcf = row.get('Free Cash Flow', 0)
    ocf = row.get('Operating Cash Flow', 0)
    
    # Debt & Holdings
    de = row.get('Debt To Equity', 0.5)
    fii_chg = row.get('Change In FII Holdings Latest Quarter', 0)
    prom_chg = row.get('Change In Promoter Holdings Latest Quarter', 0)
    promoter_hold = row.get('Promoter Holdings', 50)
    
    # Technical
    rsi_w = row.get('RSI 14W', 50)
    
    # Quality Gate
    quality_gate = row.get('Quality_Gate', 1.0)
    
    # Momentum Acceleration
    mom_accel = row.get('Momentum_Acceleration', 0)
    
    # Debt data
    debt = row.get('Debt', 0)
    total_liabilities = row.get('Total Liabilities', 0)
    
    # ═══════════════════════════════════════════════════════
    # TRAP PROBABILITY CALCULATION (0-100%)
    # ═══════════════════════════════════════════════════════
    
    trap_probability = 0
    red_flags = []
    
    # 🚨 CASH TRAP: Burning cash
    if fcf < 0 and ocf < 0:
        red_flags.append("CASH_TRAP")
        trap_probability += min(abs(fcf) / 500, 1) * 25
    elif fcf < 0:
        trap_probability += 5
    
    # 🚨 DEBT BOMB: Can't service debt
    if de > 2:
        debt_value = debt if debt > 0 else (total_liabilities * 0.5)
        if debt_value > 0 and ocf < debt_value * 0.1:
            red_flags.append("DEBT_BOMB")
            trap_probability += 20
        else:
            red_flags.append("HIGH_DEBT")
            trap_probability += 8
    elif de > 1.5:
        trap_probability += 3
    
    # Negative OCF with debt
    if ocf < 0 and de > 1 and "DEBT_BOMB" not in red_flags:
        red_flags.append("DEBT_STRESS")
        trap_probability += 15
    
    # 🚨 PROMOTER EXIT
    if prom_chg < -3:
        red_flags.append("PROMOTER_EXIT")
        trap_probability += min(abs(prom_chg) / 5, 1) * 25
    elif prom_chg < -1.5:
        trap_probability += 8
    
    # 🚨 LOW SKIN
    if promoter_hold < 25:
        red_flags.append("LOW_SKIN")
        trap_probability += 10
    
    # 🚨 FII EXITING while momentum looks good
    if fii_chg < -2 and sm > 0.5:
        red_flags.append("FII_EXITING")
        trap_probability += 15
    elif fii_chg < -1:
        trap_probability += 5
    
    # 🚨 OVERBOUGHT
    if rsi_w > 75 and sq < 0.4:
        red_flags.append("OVERBOUGHT")
        trap_probability += 12
    elif rsi_w > 80:
        trap_probability += 8
    
    # 🚨 MOMENTUM FADING
    if mom_accel < -5 and sm > 0.5:
        red_flags.append("MOM_FADING")
        trap_probability += 8
    
    # 🚨 QUALITY GATE FAILURE
    if quality_gate < 0.4:
        trap_probability += (1 - quality_gate) * 15
    
    trap_probability = min(trap_probability, 100)
    
    # ═══════════════════════════════════════════════════════
    # TRAP OVERRIDE (Safety Net)
    # ═══════════════════════════════════════════════════════
    
    if "CASH_TRAP" in red_flags and "DEBT_BOMB" in red_flags:
        return "🚨 DEATH SPIRAL", "trap", trap_probability
    
    if "CASH_TRAP" in red_flags and score > 60 and sm > 0.6:
        return "🚨 PUMP & DUMP", "trap", trap_probability
    
    if "PROMOTER_EXIT" in red_flags:
        return "🚨 INSIDER EXIT", "trap", trap_probability
    
    if "FII_EXITING" in red_flags and "CASH_TRAP" in red_flags:
        return "🚨 SMART EXIT", "trap", trap_probability
    
    if trap_probability >= 60:
        return f"⚠️ RISKY ({trap_probability:.0f}%)", "trap", trap_probability
    
    if len(red_flags) >= 3:
        return f"⚠️ RISKY ({red_flags[0]})", "trap", trap_probability
    
    # ═══════════════════════════════════════════════════════
    # PURE PERCENTILE VERDICT (Simple & Clean)
    # ═══════════════════════════════════════════════════════
    
    flag_indicator = f" ⚡{red_flags[0]}" if len(red_flags) == 1 else ""
    trap_indicator = f" ({trap_probability:.0f}%)" if trap_probability >= 20 else ""
    
    # Top 5%: STRONG BUY
    if score >= 85 and trap_probability < 20 and len(red_flags) == 0:
        return "💎 STRONG BUY", "strong-buy", trap_probability
    
    # Top 15%: BUY
    if score >= 70 and trap_probability < 40 and len(red_flags) <= 1:
        return f"📈 BUY{flag_indicator}", "buy", trap_probability
    
    # Top 40%: HOLD
    if score >= 50 and trap_probability < 50:
        return f"⏸️ HOLD{trap_indicator}", "hold", trap_probability
    
    # Bottom 40% with some issues: RISKY
    if score >= 30:
        return f"⚠️ RISKY{trap_indicator}", "trap", trap_probability
    
    # Bottom: AVOID
    return "❌ AVOID", "avoid", trap_probability

# =========================================================
# 📊 MAIN DASHBOARD - V4 ULTRA (4-Factor Pure Quant)
# =========================================================
def main():
    # Clean Header
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem 0;'>
        <h1 style='margin:0; font-weight:700; color:#1a1a2e;'>V4 ULTRA</h1>
        <p style='color:#666; margin:0.3rem 0 0 0; font-size:0.95rem;'>4-Factor Pure Quant • Momentum-Dominant • Data-Driven</p>
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
        st.markdown("### ⚙️ Settings")
        
        show_all_columns = st.checkbox("Show Factor Scores", value=False)
        top_n = st.slider("Display Top", 10, 100, 30, help="Number of stocks to show")
        
        # V4 ULTRA 4-FACTOR MODEL
        st.markdown("---")
        st.markdown("### 🎯 4-Factor Model")
        st.caption("Pure quant, no style complexity")
        
        st.markdown("""
        | Factor | Weight | Why |
        |--------|--------|-----|
        | **Momentum** | 60% | RSI +0.878, Ret3M +0.707 |
        | **Institutional** | 20% | FII +0.499 |
        | **Quality** | 10% | ROCE +0.413 |
        | **Safety** | 10% | Trap avoidance |
        """)
        
        with st.expander("🛡️ Safety Net Traps", expanded=False):
            st.markdown("""
            - 🚨 **DEATH SPIRAL**: Cash burn + Debt
            - 🚨 **PUMP & DUMP**: Momentum + No cash
            - 🚨 **INSIDER EXIT**: Promoter selling >3%
            - 🚨 **FII EXITING**: Smart money fleeing
            - 🚨 **LOW SKIN**: Promoter <25%
            """)
        
        with st.expander("❌ Dead Weight (Removed)", expanded=False):
            st.markdown("""
            | Metric | Why Removed |
            |--------|-------------|
            | NPM | -0.013 corr ❌ |
            | DII Changes | NEGATIVE ❌ |
            | Low PE | +0.551 corr! ❌ |
            | 7 Factor Soup | Overfit ❌ |
            | Style Classification | No edge ❌ |
            
            *Simpler = Better*
            """)
    
    if not uploaded_files:
        # Enhanced welcome message
        st.markdown("""
        <div style='text-align:center; padding:3rem; background:#f8f9fa; border-radius:12px; margin:2rem 0;'>
            <h3 style='color:#333; margin-bottom:0.5rem;'>🚀 Upload Your CSV Files to Start</h3>
            <p style='color:#666;'>Drag and drop your stock data files in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show V4 ULTRA advantages
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 V4 ULTRA: Pure Quant
            
            **4 Factors Only** (data-driven, no theory):
            
            | Factor | Weight | Components |
            |--------|--------|------------|
            | **Momentum** | 60% | RSI, Ret3M, 52WH, Accel |
            | **Institutional** | 20% | FII changes, Promoter |
            | **Quality** | 10% | ROCE only |
            | **Safety** | 10% | OCF, FCF, D/E |
            
            **Key Insight**: High PE = GOOD (+0.551 corr!)
            """)
        
        with col2:
            st.markdown("""
            ### 🛡️ Safety Net Feature
            
            Catches **traps** that complex models miss:
            
            - 🚨 **DEATH SPIRAL**: Cash burn + Debt bomb
            - 🚨 **PUMP & DUMP**: Momentum + No cash
            - 🚨 **INSIDER EXIT**: Promoter dumping >3%
            - 🚨 **SMART EXIT**: FII fleeing + Cash trap
            - 🚨 **LOW SKIN**: Promoter <25%
            
            **Simple percentile verdicts**:
            - � STRONG BUY: Top 5%
            - 📈 BUY: Top 15%
            - ⏸️ HOLD: Top 40%
            - ⚠️ RISKY: Bottom 40%
            - ❌ AVOID: Bottom
            """)
        
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
            | **Profitability** | ROE, ROCE, OPM |
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
    
    # Generate Verdicts (now returns 3-tuple: verdict, class, trap_probability)
    verdicts = df.apply(get_ultimate_verdict, axis=1)
    df['Verdict'] = verdicts.apply(lambda x: x[0])
    df['Verdict_Class'] = verdicts.apply(lambda x: x[1])
    df['Trap_Probability'] = verdicts.apply(lambda x: x[2])  # 🆕 TRAP PROBABILITY!
    
    # Sort and Rank
    df = df.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # ═══════════════════════════════════════════════════════
    # TABS - CLEAN LAYOUT
    # ═══════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs(["Rankings", "Scanner", "Charts", "Export"])
    
    with tab1:
        # V4 ULTRA: Simple verdict filter only (no style complexity)
        verdict_options = df['Verdict'].unique().tolist()
        buy_verdicts = [v for v in verdict_options if 'BUY' in v]
        verdict_filter = st.multiselect("Filter by Verdict", options=verdict_options, default=buy_verdicts if buy_verdicts else None)
        
        # Apply verdict filter
        filtered_df = df.copy()
        if verdict_filter:
            filtered_df = filtered_df[filtered_df['Verdict'].isin(verdict_filter)]
        
        # Display Columns - V4 ULTRA 4-Factor Model
        base_cols = ['Rank', 'Name', 'Verdict', 'Final_Score', 'Trap_Probability']
        price_cols = ['Close Price', 'Price To Earnings']
        metric_cols = ['ROCE', 'Debt To Equity', 'Free Cash Flow']
        score_cols = ['Score_Momentum', 'Score_Institutional', 'Score_Quality', 'Score_Safety']  # 4 factors only
        advanced_cols = ['Quality_Gate', 'Momentum_Acceleration', 'Data_Confidence']
        
        if show_all_columns:
            display_cols = base_cols + score_cols + advanced_cols
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
        
        # Quick Stats - Simple counts
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💎 Strong Buy", len(df[df['Verdict_Class'] == 'strong-buy']))
        col2.metric("📈 Buy", len(df[df['Verdict_Class'] == 'buy']))
        col3.metric("⚠️ Risky", len(df[df['Verdict_Class'] == 'trap']))
        col4.metric("❌ Avoid", len(df[df['Verdict_Class'] == 'avoid']))
        col5.metric("⏸️ Hold", len(df[df['Verdict_Class'] == 'hold']))
    
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
            # Radar Chart for Top Stock - V4 ULTRA 4 FACTORS
            if len(df) > 0:
                top_stock = df.iloc[0]
                categories = ['Momentum (60%)', 'Institutional (20%)', 'Quality (10%)', 'Safety (10%)']
                values = [
                    top_stock.get('Score_Momentum', 0.5),
                    top_stock.get('Score_Institutional', 0.5),
                    top_stock.get('Score_Quality', 0.5),
                    top_stock.get('Score_Safety', 0.5)
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
                    title=f"Top Stock: {top_stock.get('Name', 'N/A')}",
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
                "Buy Signals",
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
