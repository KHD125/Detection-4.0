import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# 🚀 V4 ULTRA+ FIXED — All bugs from evaluation resolved
# =========================================================
# FIXES APPLIED:
#   1. cache_data removed (file objects can't be hashed)
#   2. Z-score pipeline fixed (removed redundant rank step)
#   3. total_expected count corrected to match actual flags
#   4. Sector aliases consolidated (DRY — single source of truth)
#   5. Early Entry / Turnaround loops VECTORIZED
#   6. Data_Confidence vs "no penalty" contradiction resolved
#   7. Verdict overlap priority documented & cleaned
#   8. color_row fixed for Styler compatibility
# =========================================================

st.set_page_config(
    page_title="V4 ULTRA+ | Stock Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 STYLING
# =========================================================
st.markdown("""
<style>
    .stApp { background-color: #fafafa; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-box {
        background: white; border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8; text-align: center;
    }
    .metric-label {
        font-size: 0.85rem; color: #888;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 1.8rem; font-weight: 600; color: #1a1a1a; }
    .metric-value.green { color: #10b981; }
    .metric-value.red { color: #ef4444; }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8e8e8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background-color: #f1f5f9;
        border-radius: 10px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 10px 20px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# FIX #4: SECTOR THRESHOLDS — DRY (No Duplicates)
# Each sector defined ONCE. Aliases point to the same key.
# =========================================================
_SECTOR_BASE = {
    'default': {
        'debt_to_equity': {'danger': 1.0, 'high': 0.5, 'normal': 0.33},
        'pe': {'danger': 100, 'high': 50, 'normal': 25},
        'roe': {'min_good': 12, 'min_acceptable': 8},
        'roce': {'min_good': 15, 'min_acceptable': 10},
        'opm': {'min_good': 12, 'min_acceptable': 8},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 40, 'high': 60},
        'promoter_min': 25,
    },
    'bank': {
        'debt_to_equity': {'danger': 15.0, 'high': 12.0, 'normal': 8.0},
        'pe': {'danger': 50, 'high': 25, 'normal': 15},
        'roe': {'min_good': 14, 'min_acceptable': 10},
        'roce': {'min_good': 2, 'min_acceptable': 1},
        'opm': {'min_good': 20, 'min_acceptable': 15},
        'fii_accum': {'strong': 0.3, 'moderate': 0.2, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 35, 'high': 55},
        'promoter_min': 20,
    },
    'nbfc': {
        'debt_to_equity': {'danger': 10.0, 'high': 7.0, 'normal': 5.0},
        'pe': {'danger': 60, 'high': 35, 'normal': 20},
        'roe': {'min_good': 15, 'min_acceptable': 12},
        'roce': {'min_good': 3, 'min_acceptable': 2},
        'opm': {'min_good': 25, 'min_acceptable': 18},
        'fii_accum': {'strong': 0.4, 'moderate': 0.25, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 38, 'high': 58},
        'promoter_min': 25,
    },
    'it': {
        'debt_to_equity': {'danger': 1.0, 'high': 0.5, 'normal': 0.2},
        'pe': {'danger': 80, 'high': 50, 'normal': 30},
        'roe': {'min_good': 20, 'min_acceptable': 15},
        'roce': {'min_good': 25, 'min_acceptable': 18},
        'opm': {'min_good': 20, 'min_acceptable': 15},
        'fii_accum': {'strong': 0.4, 'moderate': 0.25, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 42, 'high': 62},
        'promoter_min': 30,
    },
    'fmcg': {
        'debt_to_equity': {'danger': 1.5, 'high': 0.8, 'normal': 0.3},
        'pe': {'danger': 100, 'high': 60, 'normal': 40},
        'roe': {'min_good': 25, 'min_acceptable': 18},
        'roce': {'min_good': 30, 'min_acceptable': 22},
        'opm': {'min_good': 18, 'min_acceptable': 14},
        'fii_accum': {'strong': 0.3, 'moderate': 0.2, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 45, 'high': 65},
        'promoter_min': 40,
    },
    'pharma': {
        'debt_to_equity': {'danger': 2.0, 'high': 1.0, 'normal': 0.5},
        'pe': {'danger': 80, 'high': 45, 'normal': 25},
        'roe': {'min_good': 18, 'min_acceptable': 12},
        'roce': {'min_good': 18, 'min_acceptable': 12},
        'opm': {'min_good': 20, 'min_acceptable': 15},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 40, 'high': 60},
        'promoter_min': 35,
    },
    'infrastructure': {
        'debt_to_equity': {'danger': 4.0, 'high': 2.5, 'normal': 1.5},
        'pe': {'danger': 60, 'high': 35, 'normal': 20},
        'roe': {'min_good': 12, 'min_acceptable': 8},
        'roce': {'min_good': 12, 'min_acceptable': 8},
        'opm': {'min_good': 12, 'min_acceptable': 8},
        'fii_accum': {'strong': 0.6, 'moderate': 0.35, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 38, 'high': 58},
        'promoter_min': 30,
    },
    'capital_goods': {
        'debt_to_equity': {'danger': 3.5, 'high': 2.0, 'normal': 1.2},
        'pe': {'danger': 70, 'high': 40, 'normal': 25},
        'roe': {'min_good': 15, 'min_acceptable': 10},
        'roce': {'min_good': 15, 'min_acceptable': 10},
        'opm': {'min_good': 14, 'min_acceptable': 10},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 40, 'high': 60},
        'promoter_min': 30,
    },
    'metal': {
        'debt_to_equity': {'danger': 3.5, 'high': 2.0, 'normal': 1.2},
        'pe': {'danger': 30, 'high': 15, 'normal': 8},
        'roe': {'min_good': 15, 'min_acceptable': 10},
        'roce': {'min_good': 15, 'min_acceptable': 10},
        'opm': {'min_good': 20, 'min_acceptable': 12},
        'fii_accum': {'strong': 0.6, 'moderate': 0.35, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 35, 'high': 55},
        'promoter_min': 30,
    },
    'power': {
        'debt_to_equity': {'danger': 5.0, 'high': 3.0, 'normal': 2.0},
        'pe': {'danger': 40, 'high': 20, 'normal': 12},
        'roe': {'min_good': 12, 'min_acceptable': 8},
        'roce': {'min_good': 10, 'min_acceptable': 7},
        'opm': {'min_good': 25, 'min_acceptable': 18},
        'fii_accum': {'strong': 0.4, 'moderate': 0.25, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 40, 'high': 58},
        'promoter_min': 40,
    },
    'auto': {
        'debt_to_equity': {'danger': 2.5, 'high': 1.5, 'normal': 0.8},
        'pe': {'danger': 60, 'high': 35, 'normal': 20},
        'roe': {'min_good': 18, 'min_acceptable': 12},
        'roce': {'min_good': 18, 'min_acceptable': 12},
        'opm': {'min_good': 14, 'min_acceptable': 10},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 40, 'high': 60},
        'promoter_min': 35,
    },
    'realty': {
        'debt_to_equity': {'danger': 4.0, 'high': 2.5, 'normal': 1.5},
        'pe': {'danger': 50, 'high': 30, 'normal': 15},
        'roe': {'min_good': 12, 'min_acceptable': 8},
        'roce': {'min_good': 10, 'min_acceptable': 6},
        'opm': {'min_good': 25, 'min_acceptable': 18},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 35, 'high': 55},
        'promoter_min': 40,
    },
    'chemical': {
        'debt_to_equity': {'danger': 2.5, 'high': 1.5, 'normal': 0.8},
        'pe': {'danger': 60, 'high': 35, 'normal': 22},
        'roe': {'min_good': 18, 'min_acceptable': 14},
        'roce': {'min_good': 18, 'min_acceptable': 14},
        'opm': {'min_good': 16, 'min_acceptable': 12},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 40, 'high': 60},
        'promoter_min': 40,
    },
    'textile': {
        'debt_to_equity': {'danger': 2.5, 'high': 1.5, 'normal': 1.0},
        'pe': {'danger': 40, 'high': 25, 'normal': 15},
        'roe': {'min_good': 14, 'min_acceptable': 10},
        'roce': {'min_good': 14, 'min_acceptable': 10},
        'opm': {'min_good': 12, 'min_acceptable': 8},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 38, 'high': 58},
        'promoter_min': 35,
    },
}

# Alias map — points to the canonical key. ZERO duplication.
_SECTOR_ALIASES = {
    'finance': 'bank', 'lending': 'bank', 'credit': 'bank', 'loan': 'bank',
    'software': 'it', 'technology': 'it', 'tech': 'it', 'digital': 'it',
    'internet': 'it', 'saas': 'it',
    'consumer': 'fmcg', 'food': 'fmcg', 'beverage': 'fmcg', 'personal': 'fmcg',
    'healthcare': 'pharma', 'drug': 'pharma', 'biotech': 'pharma', 'hospital': 'pharma',
    'infra': 'infrastructure', 'construct': 'infrastructure',
    'engineering': 'infrastructure', 'capital': 'infrastructure',
    'mining': 'metal', 'steel': 'metal', 'aluminium': 'metal',
    'copper': 'metal', 'iron': 'metal',
    'electric': 'power', 'energy': 'power', 'utility': 'power', 'utilities': 'power',
    'vehicle': 'auto', 'motor': 'auto', 'automobile': 'auto', 'tyre': 'auto',
    'real estate': 'realty', 'property': 'realty', 'housing': 'realty',
    'petrochem': 'chemical', 'specialty': 'chemical',
    'apparel': 'textile', 'garment': 'textile', 'fabric': 'textile',
    'financial service': 'nbfc', 'insurance': 'nbfc',
    'capital goods': 'capital_goods', 'construction': 'infrastructure',
}

# Keyword → canonical sector (for fuzzy matching on Industry strings)
_KEYWORD_MAP = {
    'bank': 'bank', 'nbfc': 'nbfc',
    'software': 'it', 'tech': 'it', 'digital': 'it', 'internet': 'it', 'saas': 'it',
    'pharma': 'pharma', 'drug': 'pharma', 'biotech': 'pharma', 'hospital': 'pharma', 'health': 'pharma',
    'fmcg': 'fmcg', 'consumer': 'fmcg', 'food': 'fmcg', 'beverage': 'fmcg', 'personal': 'fmcg',
    'infra': 'infrastructure', 'construct': 'infrastructure', 'engineering': 'infrastructure',
    'metal': 'metal', 'steel': 'metal', 'aluminium': 'metal', 'copper': 'metal', 'mining': 'metal',
    'power': 'power', 'electric': 'power', 'energy': 'power', 'utility': 'power',
    'auto': 'auto', 'vehicle': 'auto', 'motor': 'auto', 'tyre': 'auto',
    'real': 'realty', 'property': 'realty', 'housing': 'realty',
    'chemical': 'chemical', 'petrochem': 'chemical', 'specialty': 'chemical',
    'textile': 'textile', 'apparel': 'textile', 'garment': 'textile', 'fabric': 'textile',
    'capital good': 'capital_goods',
}


def get_sector_thresholds(industry):
    """Single lookup: direct → alias → keyword fuzzy → default."""
    if not industry or pd.isna(industry):
        return _SECTOR_BASE['default']

    ind = str(industry).lower().strip()

    # 1. Direct match in base
    if ind in _SECTOR_BASE:
        return _SECTOR_BASE[ind]

    # 2. Direct match in aliases
    if ind in _SECTOR_ALIASES:
        return _SECTOR_BASE[_SECTOR_ALIASES[ind]]

    # 3. Keyword fuzzy scan (first match wins)
    for kw, canonical in _KEYWORD_MAP.items():
        if kw in ind:
            return _SECTOR_BASE[canonical]

    return _SECTOR_BASE['default']


# =========================================================
# ONE-OFF INCOME DETECTION (unchanged logic, kept as-is)
# =========================================================
def detect_one_off_income(pat_growth, rev_growth, opm_current, opm_expected=None):
    if pd.isna(pat_growth) or pd.isna(rev_growth):
        return False, 0, "INSUFFICIENT_DATA"

    suspicious = False
    conf = 0
    reasons = []

    if pat_growth > 20 and rev_growth < pat_growth * 0.5:
        suspicious, conf = True, conf + 40
        reasons.append("PAT>>REV")
    if pat_growth > 10 and rev_growth < 0:
        suspicious, conf = True, conf + 50
        reasons.append("PAT+REV-")
    if pat_growth > 50 and abs(rev_growth) < 5:
        suspicious, conf = True, conf + 35
        reasons.append("PAT_SPIKE")
    if opm_expected and opm_current and opm_current < opm_expected * 0.7 and pat_growth > 15:
        suspicious, conf = True, conf + 30
        reasons.append("OPM_WEAK")

    return suspicious, min(conf, 100), (','.join(reasons) if reasons else "CLEAN")


# =========================================================
# DATA PROCESSING — FIX #1: removed @st.cache_data
# File objects can't be hashed reliably by Streamlit.
# We use st.session_state to cache manually instead.
# =========================================================
def process_files(uploaded_files):
    if not uploaded_files:
        return None

    master = pd.DataFrame()
    prog = st.progress(0)
    status = st.empty()

    for i, f in enumerate(uploaded_files):
        try:
            status.text(f"Processing: {f.name}")
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()

            if master.empty:
                master = df
            else:
                key = 'companyId' if 'companyId' in master.columns and 'companyId' in df.columns else 'Name'
                new_cols = [c for c in df.columns if c not in master.columns or c == key]
                if len(new_cols) > 1:
                    master = pd.merge(master, df[new_cols], on=key, how='outer')
        except Exception as e:
            st.warning(f"⚠️ Skipped {f.name}: {e}")
        prog.progress((i + 1) / len(uploaded_files))

    prog.empty()
    status.empty()

    # --- Numeric coercion ---
    non_num = {'companyId', 'Name', 'Industry', 'Sector', 'Fundamentals Source', 'Verdict'}
    num_cols = [c for c in master.columns if c not in non_num]
    for c in num_cols:
        master[c] = pd.to_numeric(master[c], errors='coerce')

    # --- Smart NaN fill ---
    growth_kw = ("Growth", "Change", "Returns", "Flow")
    for c in num_cols:
        if any(kw in c for kw in growth_kw):
            master[c].fillna(0, inplace=True)
        else:
            master[c].fillna(master[c].median(), inplace=True)

    return master


# =========================================================
# MARKET REGIME (unchanged logic)
# =========================================================
def analyze_market_regime(df):
    r1m = df['Returns 1M'].median() if 'Returns 1M' in df.columns else 0
    r3m = df['Returns 3M'].median() if 'Returns 3M' in df.columns else 0
    r1y = df['Returns 1Y'].median() if 'Returns 1Y' in df.columns else 0

    b1m = (df['Returns 1M'] > 0).mean() * 100 if 'Returns 1M' in df.columns else 50
    b3m = (df['Returns 3M'] > 0).mean() * 100 if 'Returns 3M' in df.columns else 50

    trend = "IMPROVING" if b1m > b3m + 10 else ("DETERIORATING" if b1m < b3m - 10 else "STABLE")

    if r3m > 10 and r1y > 20 and b3m > 65:
        regime, w, strat = "🚀 STRONG BULL", {'Momentum': .65, 'Institutional': .20, 'Quality': .10, 'Safety': .05}, "Max Momentum"
    elif r3m > 5 and b3m > 50:
        regime, w, strat = "📈 BULL", {'Momentum': .60, 'Institutional': .20, 'Quality': .10, 'Safety': .10}, "Momentum"
    elif r3m < -10 or (r1m < -5 and b1m < 30):
        regime, w, strat = "🐻 BEAR", {'Momentum': .45, 'Institutional': .20, 'Quality': .15, 'Safety': .20}, "Defensive"
    elif r3m < -3 or b3m < 40:
        regime, w, strat = "⚠️ CORRECTION", {'Momentum': .50, 'Institutional': .20, 'Quality': .15, 'Safety': .15}, "Cautious"
    else:
        regime, w, strat = "⚖️ SIDEWAYS", {'Momentum': .55, 'Institutional': .20, 'Quality': .12, 'Safety': .13}, "Selective"

    if trend == "IMPROVING":
        w['Momentum'] = min(.70, w['Momentum'] + .05)
        w['Safety'] = max(.05, w['Safety'] - .05)
        strat += " ↑"
    elif trend == "DETERIORATING":
        w['Safety'] = min(.25, w['Safety'] + .05)
        w['Momentum'] = max(.40, w['Momentum'] - .05)
        strat += " ↓"

    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}

    stats = {'r1m': r1m, 'r3m': r3m, 'r1y': r1y, 'breadth_3m': b3m,
             'breadth_1m': b1m, 'regime_trend': trend}
    return regime, w, strat, stats


# =========================================================
# FIX #2 & #3 & #5: SCORING ENGINE
#   - Z-score: proper min-max to [0,1], NO redundant rank
#   - total_expected matches actual flag count (33)
#   - Early Entry + Turnaround fully VECTORIZED (no loops)
#   - FIX #6: removed Data_Confidence multiplier
#     → "no penalty" philosophy is now consistent
# =========================================================
def run_ultimate_scoring(df, base_weights):
    n = len(df)

    # ─── helpers ────────────────────────────────────────
    def col(name, default=0):
        """Return (Series, exists_bool). NaN filled with median or default."""
        if name in df.columns:
            s = df[name].copy()
            med = s.median()
            return s.fillna(med if not pd.isna(med) else default), True
        return pd.Series([default] * n, index=df.index), False

    def normalize(series, lower_better=False, exists=True):
        """
        FIX #2: Proper min-max normalisation to [0, 1].
        Outliers capped at 2nd/98th percentile first.
        If data doesn't exist → neutral 0.5 (no penalty).
        """
        if not exists:
            return pd.Series([0.5] * n, index=df.index)

        s = series.clip(lower=series.quantile(0.02), upper=series.quantile(0.98))
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series([0.5] * n, index=df.index)

        normed = (s - lo) / (hi - lo)          # 0 → 1
        return (1 - normed) if lower_better else normed

    def weighted_sum(components):
        """
        Weighted average that redistributes weights among available data.
        components = [(series, weight, exists), ...]
        """
        avail = [(s, w) for s, w, e in components if e]
        if not avail:
            return pd.Series([0.5] * n, index=df.index)
        tw = sum(w for _, w in avail)
        result = pd.Series(0.0, index=df.index)
        for s, w in avail:
            result += s * (w / tw)
        return result

    # ─── extract columns ────────────────────────────────
    roe,  h_roe  = col('ROE', 12)
    roce, h_roce = col('ROCE', 12)
    opm,  h_opm  = col('OPM', 12)

    pat_ttm, h_pat_ttm   = col('PAT Growth TTM', 10)
    pat_yoy, h_pat_yoy   = col('PAT Growth YoY', 10)
    rev_ttm, h_rev_ttm   = col('Revenue Growth TTM', 10)
    rev_yoy, h_rev_yoy   = col('Revenue Growth YoY', 10)
    eps_ttm, h_eps_ttm   = col('EPS Growth TTM', 10)
    pat_qoq, h_pat_qoq   = col('PAT Growth QoQ', 0)
    rev_qoq, h_rev_qoq   = col('Revenue Growth QoQ', 0)

    pe,  h_pe  = col('Price To Earnings', 25)
    ps,  h_ps  = col('Price To Sales', 3)

    de,  h_de  = col('Debt To Equity', 0.5)
    prom, h_prom = col('Promoter Holdings', 50)
    fcf, h_fcf  = col('Free Cash Flow', 0)
    ocf, h_ocf  = col('Operating Cash Flow', 0)
    cash, h_cash = col('Cash Equivalents', 0)
    debt, h_debt = col('Debt', 1)

    rsi_w, h_rsi_w = col('RSI 14W', 50)
    rsi_d, h_rsi_d = col('RSI 14D', 50)
    adx_w, h_adx_w = col('ADX 14W', 25)
    ret_1m, h_ret_1m = col('Returns 1M', 0)
    ret_3m, h_ret_3m = col('Returns 3M', 0)
    ret_6m, h_ret_6m = col('Returns 6M', 0)
    ret_1y, h_ret_1y = col('Returns 1Y', 0)

    fii,     h_fii     = col('FII Holdings', 5)
    dii,     h_dii     = col('DII Holdings', 10)
    fii_chg, h_fii_chg = col('Change In FII Holdings Latest Quarter', 0)
    dii_chg, h_dii_chg = col('Change In DII Holdings Latest Quarter', 0)
    prom_chg, h_prom_chg = col('Change In Promoter Holdings Latest Quarter', 0)
    fii_chg_1y, h_fii_chg_1y = col('Change In FII Holdings 1 Year', 0)

    dist_52wh, h_52wh   = col('52WH Distance', -15)
    ret_vs_nifty, h_vs_nifty = col('Returns Vs Nifty 500 3M', 0)

    has_industry = 'Industry' in df.columns
    industry_series = df['Industry'] if has_industry else pd.Series([''] * n, index=df.index)

    # ─── VECTORIZED: Momentum Acceleration ──────────────
    if h_ret_1m and h_ret_3m:
        mom_accel = ret_1m - ((ret_3m - ret_1m) / 2)
        h_mom_accel = True
    else:
        mom_accel = pd.Series(0.0, index=df.index)
        h_mom_accel = False
    df['Momentum_Acceleration'] = mom_accel

    # ─── VECTORIZED: RSI Velocity ────────────────────────
    if h_rsi_w and h_rsi_d:
        rsi_vel = rsi_w - rsi_d
        h_rsi_vel = True
    else:
        rsi_vel = pd.Series(0.0, index=df.index)
        h_rsi_vel = False
    df['RSI_Velocity'] = rsi_vel

    # ─── FIX #5: VECTORIZED Early Entry Score ───────────
    # Build per-row sector thresholds as vectorized arrays
    fii_strong  = industry_series.map(lambda x: get_sector_thresholds(x)['fii_accum']['strong'])
    fii_mod     = industry_series.map(lambda x: get_sector_thresholds(x)['fii_accum']['moderate'])
    fii_weak    = industry_series.map(lambda x: get_sector_thresholds(x)['fii_accum']['weak'])
    rsi_lo      = industry_series.map(lambda x: get_sector_thresholds(x)['rsi_sweet_spot']['low'])
    rsi_hi      = industry_series.map(lambda x: get_sector_thresholds(x)['rsi_sweet_spot']['high'])

    ee = pd.Series(0.0, index=df.index)
    ee_flags = pd.Series('', index=df.index)

    # Signal 1: FII accumulation (sector-adjusted)
    s1a = (fii_chg > fii_strong)
    s1b = (fii_chg > fii_mod) & ~s1a
    s1c = (fii_chg > fii_weak) & ~s1a & ~s1b
    ee += s1a * 25 + s1b * 15 + s1c * 8
    ee_flags = ee_flags.where(~s1a, ee_flags + 'FII_ACCUM,')
    ee_flags = ee_flags.where(~s1b, ee_flags + 'FII_MOD,')

    # Signal 2: FII 1Y trend
    if h_fii_chg_1y:
        s2 = (fii_chg_1y > 1) & (fii_chg > 0)
        ee += s2 * 20
        ee_flags = ee_flags.where(~s2, ee_flags + 'FII_TREND,')

    # Signal 3: RSI in sweet spot (sector-adjusted)
    s3a = (rsi_w >= rsi_lo) & (rsi_w <= rsi_hi)
    s3b = ((rsi_w >= (rsi_lo - 5)) & (rsi_w < rsi_lo)) | ((rsi_w > rsi_hi) & (rsi_w <= (rsi_hi + 5)))
    ee += s3a * 20 + s3b * 10
    ee_flags = ee_flags.where(~s3a, ee_flags + 'RSI_READY,')

    # Signal 4: Price hasn't run yet
    s4a = (ret_3m >= -5) & (ret_3m <= 15)
    s4b = (ret_3m < -5)
    ee += s4a * 20 + s4b * 5
    ee_flags = ee_flags.where(~s4a, ee_flags + 'NOT_EXTENDED,')

    # Signal 5: Momentum acceleration positive
    s5 = (mom_accel > 0)
    ee += s5 * 15
    ee_flags = ee_flags.where(~s5, ee_flags + 'MOM_BUILDING,')

    df['Early_Entry_Score']   = (ee / 100).clip(0, 1)
    df['Early_Entry_Signals'] = ee_flags.str.rstrip(',')

    # ─── FIX #5: VECTORIZED Turnaround Score ─────────────
    de_normal  = industry_series.map(lambda x: get_sector_thresholds(x)['debt_to_equity']['normal'])
    de_high    = industry_series.map(lambda x: get_sector_thresholds(x)['debt_to_equity']['high'])
    de_danger  = industry_series.map(lambda x: get_sector_thresholds(x)['debt_to_equity']['danger'])
    opm_good   = industry_series.map(lambda x: get_sector_thresholds(x)['opm']['min_good'])
    opm_accept = industry_series.map(lambda x: get_sector_thresholds(x)['opm']['min_acceptable'])

    is_beaten = (ret_1y < 0) | (ret_3m < -5)

    # One-off income detection (vectorised logic)
    oneoff_high = ((pat_qoq > 10) & (rev_qoq < 0)) | \
                  ((pat_qoq > 20) & (rev_qoq < pat_qoq * 0.5)) | \
                  ((pat_qoq > 50) & (rev_qoq.abs() < 5))
    oneoff_mod  = ((pat_qoq > 20) & (rev_qoq < pat_qoq * 0.5)) & ~oneoff_high

    ta = pd.Series(0.0, index=df.index)
    ta_flags = pd.Series('', index=df.index)
    ta_warn  = pd.Series('', index=df.index)

    # Penalty for one-off
    ta -= oneoff_high * 30
    ta -= oneoff_mod  * 10
    ta_warn = ta_warn.where(~oneoff_high, ta_warn + 'ONE_OFF,')
    ta_warn = ta_warn.where(~oneoff_mod,  ta_warn + 'CHECK_INCOME,')

    # Signal 1: PAT improving + Revenue backing it
    real_turn = is_beaten & (pat_qoq > pat_yoy) & (pat_qoq > 0) & (rev_qoq > 0)
    pat_accel = is_beaten & (pat_qoq > pat_yoy) & (pat_qoq > 0) & (rev_qoq <= 0) & ~oneoff_high
    pat_pos   = is_beaten & (pat_qoq > 0) & ~real_turn & ~pat_accel
    ta += real_turn * 30 + pat_accel * 20 + pat_pos * 10
    ta_flags = ta_flags.where(~real_turn, ta_flags + 'REAL_TURNAROUND,')
    ta_flags = ta_flags.where(~pat_accel, ta_flags + 'PAT_ACCEL,')
    ta_flags = ta_flags.where(~pat_pos,   ta_flags + 'PAT_POS_QOQ,')

    # Signal 2: Revenue growing
    rev_g2 = is_beaten & (rev_qoq > 5)
    rev_g1 = is_beaten & (rev_qoq > 0) & ~rev_g2
    ta += rev_g2 * 20 + rev_g1 * 10
    ta_flags = ta_flags.where(~rev_g2, ta_flags + 'REV_GROWING,')

    # Signal 3: Margin OK (sector-adjusted)
    m_good = is_beaten & (opm > opm_good)
    m_ok   = is_beaten & (opm > opm_accept) & ~m_good
    ta += m_good * 15 + m_ok * 8
    ta_flags = ta_flags.where(~m_good, ta_flags + 'MARGIN_OK,')

    # Signal 4: Promoter buying during stress
    ins_buy  = is_beaten & (prom_chg > 0.5) & (ret_3m < 0)
    ins_pos  = is_beaten & (prom_chg > 0) & ~ins_buy
    ta += ins_buy * 25 + ins_pos * 10
    ta_flags = ta_flags.where(~ins_buy, ta_flags + 'INSIDER_BUY,')

    # Signal 5: Debt under control (sector-adjusted)
    d_low  = is_beaten & (de < de_normal)
    d_ok   = is_beaten & (de < de_high) & ~d_low
    d_bad  = is_beaten & (de > de_danger)
    ta += d_low * 15 + d_ok * 8 - d_bad * 10
    ta_flags = ta_flags.where(~d_low, ta_flags + 'LOW_DEBT,')
    ta_warn  = ta_warn.where(~d_bad,  ta_warn  + 'EXCESS_DEBT,')

    # Signal 6: OCF positive
    ocf_pos = is_beaten & (ocf > 0)
    ta += ocf_pos * 20
    ta_flags = ta_flags.where(~ocf_pos, ta_flags + 'OCF_POS,')

    # Zero out non-beaten-down stocks
    ta = ta.where(is_beaten, 0)
    ta_flags = ta_flags.where(is_beaten, '')
    ta_warn  = ta_warn.where(is_beaten, '')

    df['Turnaround_Score']    = (ta / 100).clip(0, 1)
    df['Turnaround_Signals']  = ta_flags.str.rstrip(',')
    df['Turnaround_Warnings'] = ta_warn.str.rstrip(',')

    # ─── VECTORIZED Quality Gate ─────────────────────────
    roe_min_v  = industry_series.map(lambda x: get_sector_thresholds(x)['roe']['min_acceptable'])
    de_high_v  = industry_series.map(lambda x: get_sector_thresholds(x)['debt_to_equity']['high'])

    qg  = (ocf > 0).astype(int)
    qg += (roe > roe_min_v).astype(int)
    qg += (de < de_high_v).astype(int)
    qg += (pat_ttm > -30).astype(int)
    qg += ((fii + dii) > 5).astype(int)
    df['Quality_Gate'] = qg / 5.0

    # ═══ 4-FACTOR SCORES ═════════════════════════════════

    # FACTOR 1: MOMENTUM (60%)
    rsi_adj = rsi_w.copy()
    rsi_adj = np.where(rsi_w > 75, rsi_w * 0.7, rsi_adj)
    rsi_adj = np.where(rsi_w < 30, rsi_w * 0.8, rsi_adj)

    df['Score_Momentum'] = weighted_sum([
        (normalize(pd.Series(rsi_adj, index=df.index), exists=h_rsi_w),  0.35, h_rsi_w),
        (normalize(ret_3m, exists=h_ret_3m),                             0.25, h_ret_3m),
        (normalize(dist_52wh, lower_better=True, exists=h_52wh),         0.20, h_52wh),
        (normalize(mom_accel.clip(-30, 30), exists=h_mom_accel),         0.12, h_mom_accel),
        (normalize(rsi_vel.clip(-20, 20), exists=h_rsi_vel),             0.08, h_rsi_vel),
    ])

    # FACTOR 2: INSTITUTIONAL (20%)
    if h_fii_chg and h_fii_chg_1y:
        fii_accum_s = (fii_chg + fii_chg_1y * 0.5).clip(-5, 5)
        h_fii_accum = True
    else:
        fii_accum_s, h_fii_accum = fii_chg, h_fii_chg

    df['Score_Institutional'] = weighted_sum([
        (normalize(fii_accum_s, exists=h_fii_accum), 0.65, h_fii_accum),
        (normalize(prom_chg, exists=h_prom_chg),     0.35, h_prom_chg),
    ])

    # FACTOR 3: QUALITY (10%) — ROCE only
    df['Score_Quality'] = normalize(roce.clip(0, 50), exists=h_roce)

    # FACTOR 4: SAFETY (10%)
    df['Score_Safety'] = weighted_sum([
        (normalize(ocf, exists=h_ocf),                          0.40, h_ocf),
        (normalize(fcf, exists=h_fcf),                          0.35, h_fcf),
        (normalize(de.clip(0, 5), lower_better=True, exists=h_de), 0.25, h_de),
    ])

    # ═══ FINAL SCORE (pure weighted, no confidence penalty) ═
    df['Final_Score'] = (
        df['Score_Momentum']    * base_weights.get('Momentum',    0.60) +
        df['Score_Institutional'] * base_weights.get('Institutional', 0.20) +
        df['Score_Quality']     * base_weights.get('Quality',     0.10) +
        df['Score_Safety']      * base_weights.get('Safety',      0.10)
    ) * 100
    df['Final_Score'] = df['Final_Score'].clip(0, 100)

    # ─── FIX #3: Data coverage count (matches actual flags = 33) ──
    CRITICAL_COLS = [
        'RSI 14W', 'RSI 14D', 'ADX 14W',
        'Returns 1M', 'Returns 3M', 'Returns 6M', 'Returns 1Y',
        '52WH Distance', 'Returns Vs Nifty 500 3M',
        'Price To Earnings', 'Price To Sales', 'Debt To Equity',
        'ROE', 'ROCE', 'OPM',
        'PAT Growth TTM', 'PAT Growth YoY', 'PAT Growth QoQ',
        'Revenue Growth TTM', 'Revenue Growth YoY', 'Revenue Growth QoQ',
        'EPS Growth TTM',
        'FII Holdings', 'DII Holdings', 'Promoter Holdings',
        'Change In FII Holdings Latest Quarter',
        'Change In FII Holdings 1 Year',
        'Change In DII Holdings Latest Quarter',
        'Change In Promoter Holdings Latest Quarter',
        'Operating Cash Flow', 'Free Cash Flow',
        'Cash Equivalents', 'Debt',
    ]  # 33 items — matches reality

    present = sum(1 for c in CRITICAL_COLS if c in df.columns)
    df['Data_Coverage'] = f"{present}/{len(CRITICAL_COLS)}"

    return df


# =========================================================
# FIX #7: VERDICT ENGINE — clear priority, no overlaps
# Priority order (first match wins, documented):
#   1. TRAP overrides (safety net — always first)
#   2. EARLY ENTRY / ACCUMULATION
#   3. TURNAROUND / RECOVERY
#   4. STANDARD VERDICTS (Strong Buy → Buy → Hold → Avoid)
# =========================================================
def get_ultimate_verdict(row):
    score = row['Final_Score']
    sm = row.get('Score_Momentum', 0.5)
    si = row.get('Score_Institutional', 0.5)
    sq = row.get('Score_Quality', 0.5)
    ss = row.get('Score_Safety', 0.5)

    fcf_v  = row.get('Free Cash Flow', 0)
    ocf_v  = row.get('Operating Cash Flow', 0)
    de_v   = row.get('Debt To Equity', 0.5)
    fii_c  = row.get('Change In FII Holdings Latest Quarter', 0)
    prom_c = row.get('Change In Promoter Holdings Latest Quarter', 0)
    prom_h = row.get('Promoter Holdings', 50)
    rsi_v  = row.get('RSI 14W', 50)
    qg     = row.get('Quality_Gate', 1.0)
    ma     = row.get('Momentum_Acceleration', 0)
    ee_sc  = row.get('Early_Entry_Score', 0)
    ee_sig = row.get('Early_Entry_Signals', '')
    ta_sc  = row.get('Turnaround_Score', 0)
    ta_sig = row.get('Turnaround_Signals', '')
    ta_wrn = row.get('Turnaround_Warnings', '')
    rsi_vel = row.get('RSI_Velocity', 0)
    industry = row.get('Industry', '')
    debt_v = row.get('Debt', 0)
    tot_liab = row.get('Total Liabilities', 0)

    thresh = get_sector_thresholds(industry)
    de_t = thresh['debt_to_equity']

    # ── Trap probability ──────────────────────────────────
    trap_p = 0
    flags = []

    if fcf_v < 0 and ocf_v < 0:
        flags.append("CASH_TRAP")
        trap_p += min(abs(fcf_v) / 500, 1) * 25
    elif fcf_v < 0:
        trap_p += 5

    if de_v > de_t['danger']:
        dv = debt_v if debt_v > 0 else tot_liab * 0.5
        if dv > 0 and ocf_v < dv * 0.1:
            flags.append("DEBT_BOMB"); trap_p += 20
        else:
            flags.append("EXCESS_DEBT"); trap_p += 12
    elif de_v > de_t['high']:
        flags.append("HIGH_DEBT"); trap_p += 5

    if ocf_v < 0 and de_v > de_t['normal'] and "DEBT_BOMB" not in flags:
        flags.append("DEBT_STRESS"); trap_p += 12

    if 'ONE_OFF' in ta_wrn:
        flags.append("ONE_OFF_INCOME"); trap_p += 15
    elif 'CHECK_INCOME' in ta_wrn:
        trap_p += 5

    if prom_c < -3:
        flags.append("PROMOTER_EXIT"); trap_p += min(abs(prom_c) / 5, 1) * 25
    elif prom_c < -1.5:
        trap_p += 8

    if prom_h < 25:
        flags.append("LOW_SKIN"); trap_p += 10

    if fii_c < -2 and sm > 0.5:
        flags.append("FII_EXITING"); trap_p += 15
    elif fii_c < -1:
        trap_p += 5

    if rsi_v > 75 and sq < 0.4:
        flags.append("OVERBOUGHT"); trap_p += 12
    elif rsi_v > 80:
        trap_p += 8

    if ma < -5 and sm > 0.5:
        flags.append("MOM_FADING"); trap_p += 8

    if qg < 0.4:
        trap_p += (1 - qg) * 15

    trap_p = min(trap_p, 100)

    # ── PRIORITY 1: TRAP OVERRIDES ────────────────────────
    if "CASH_TRAP" in flags and "DEBT_BOMB" in flags:
        return "🚨 DEATH SPIRAL", "trap", trap_p
    if "CASH_TRAP" in flags and score > 60 and sm > 0.6:
        return "🚨 PUMP & DUMP", "trap", trap_p
    if "PROMOTER_EXIT" in flags:
        return "🚨 INSIDER EXIT", "trap", trap_p
    if "FII_EXITING" in flags and "CASH_TRAP" in flags:
        return "🚨 SMART EXIT", "trap", trap_p
    if trap_p >= 60:
        return f"⚠️ RISKY ({trap_p:.0f}%)", "trap", trap_p
    if len(flags) >= 3:
        return f"⚠️ RISKY ({flags[0]})", "trap", trap_p

    # ── PRIORITY 2: EARLY ENTRY ───────────────────────────
    sig_count = len([s for s in ee_sig.split(',') if s]) if ee_sig else 0
    if ee_sc >= 0.6 and trap_p < 25 and sq >= 0.4 and sig_count >= 3:
        return "🎯 EARLY ENTRY", "early-entry", trap_p
    if ee_sc >= 0.45 and 'FII_ACCUM' in ee_sig and trap_p < 30:
        return "🔍 ACCUMULATION", "accumulation", trap_p

    # ── PRIORITY 3: TURNAROUND ────────────────────────────
    has_one_off = 'ONE_OFF' in ta_wrn or 'CHECK_INCOME' in ta_wrn
    if ta_sc >= 0.6 and trap_p < 40 and not has_one_off:
        if 'REAL_TURNAROUND' in ta_sig:
            return "🔄 TURNAROUND ✓", "turnaround", trap_p
        if 'INSIDER_BUY' in ta_sig:
            return "🔄 TURNAROUND", "turnaround", trap_p
        if 'PAT_ACCEL' in ta_sig and 'REV_GROWING' in ta_sig:
            return "📊 RECOVERY", "turnaround", trap_p
    if ta_sc >= 0.5 and has_one_off and trap_p < 50 and 'INSIDER_BUY' in ta_sig:
        return "🔄 TURNAROUND ⚠️", "turnaround", trap_p
    if ta_sc >= 0.4 and 'OCF_POS' in ta_sig and trap_p < 35 and not has_one_off:
        return "🌱 IMPROVING", "turnaround", trap_p

    # ── PRIORITY 4: STANDARD VERDICTS ────────────────────
    vel_bonus = rsi_vel > 5 and ma > 0
    flag_tag  = f" ⚡{flags[0]}" if len(flags) == 1 else ""
    trap_tag  = f" ({trap_p:.0f}%)" if trap_p >= 20 else ""

    if score >= 85 and trap_p < 20 and len(flags) == 0:
        return "💎 STRONG BUY", "strong-buy", trap_p
    if score >= 80 and vel_bonus and trap_p < 20 and len(flags) == 0:
        return "💎 STRONG BUY ↗", "strong-buy", trap_p
    if score >= 70 and trap_p < 40 and len(flags) <= 1:
        return f"📈 BUY{flag_tag}", "buy", trap_p
    if score >= 50 and trap_p < 50:
        return f"⏸️ HOLD{trap_tag}", "hold", trap_p
    if score >= 30:
        return f"⚠️ RISKY{trap_tag}", "trap", trap_p

    return "❌ AVOID", "avoid", trap_p


# =========================================================
# 📊 MAIN DASHBOARD
# =========================================================
def main():
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem 0;'>
        <h1 style='margin:0; font-weight:700; color:#1a1a2e;'>V4 ULTRA+</h1>
        <p style='color:#666; margin:0.3rem 0 0 0; font-size:0.95rem;'>
            4-Factor Pure Quant • Early Entry • Turnaround Finder • Sector-Aware
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Upload Data")
        uploaded_files = st.file_uploader(
            "CSV Files", accept_multiple_files=True, type=['csv'],
            label_visibility="collapsed"
        )
        if uploaded_files:
            st.success(f"✓ {len(uploaded_files)} files loaded")

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_all = st.checkbox("Show Factor Scores", value=False)
        top_n   = st.slider("Display Top N", 10, 100, 30)

        st.markdown("---")
        st.markdown("### 🎯 4-Factor Model")
        st.markdown("""
        | Factor | Weight | Key Signals |
        |--------|--------|-------------|
        | Momentum | 60% | RSI, Ret3M, 52WH, Velocity |
        | Institutional | 20% | FII Trend, Promoter |
        | Quality | 10% | ROCE |
        | Safety | 10% | OCF, FCF, D/E |
        """)

        with st.expander("🎯 Early Entry Detection"):
            st.markdown("FII accumulating + RSI building + price hasn't run yet. Thresholds auto-adjust by sector.")
        with st.expander("🔄 Turnaround Detection"):
            st.markdown("PAT + Revenue both growing = REAL turnaround. One-off income detection flags fake recoveries.")
        with st.expander("🏭 Sector Thresholds"):
            st.markdown("Banks D/E normal=8, danger=15 | IT D/E normal=0.2, danger=1 | Metals D/E normal=1.2, danger=3.5")
        with st.expander("🛡️ Trap Types"):
            st.markdown("DEATH SPIRAL | PUMP & DUMP | INSIDER EXIT | ONE_OFF_INCOME | EXCESS_DEBT")

    # ── No data state ─────────────────────────────────────
    if not uploaded_files:
        st.markdown("""
        <div style='text-align:center; padding:3rem; background:#f8f9fa; border-radius:12px; margin:2rem 0;'>
            <h3 style='color:#333;'>🚀 Upload your CSV files in the sidebar to begin</h3>
            <p style='color:#666;'>Supports 9-file Screener.in export format</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Manual session-state cache (FIX #1) ───────────────
    file_key = str([(f.name, f.size) for f in uploaded_files])
    if 'cached_key' not in st.session_state or st.session_state['cached_key'] != file_key:
        st.session_state['cached_df']  = process_files(uploaded_files)
        st.session_state['cached_key'] = file_key

    df = st.session_state['cached_df']
    if df is None or df.empty:
        st.error("❌ No valid data found.")
        return

    # ── Market Regime ─────────────────────────────────────
    regime, weights, strategy, stats = analyze_market_regime(df)

    st.markdown(f"""
    <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:0.5rem 0 1.5rem 0;'>
        <div class='metric-box'><div class='metric-label'>Regime</div><div class='metric-value'>{regime}</div></div>
        <div class='metric-box'><div class='metric-label'>Strategy</div><div class='metric-value' style='font-size:1rem;'>{strategy}</div></div>
        <div class='metric-box'><div class='metric-label'>Stocks</div><div class='metric-value'>{len(df):,}</div></div>
        <div class='metric-box'><div class='metric-label'>Breadth 3M</div>
            <div class='metric-value' style='color:{"#10b981" if stats["breadth_3m"]>50 else "#ef4444"}'>{stats["breadth_3m"]:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Live Weights")
        for f, w in weights.items():
            st.markdown(f"**{f}**: {w*100:.0f}%")
            st.progress(w)

    # ── Score + Verdict ───────────────────────────────────
    df = run_ultimate_scoring(df, weights)

    verdicts = df.apply(get_ultimate_verdict, axis=1)
    df['Verdict']          = verdicts.apply(lambda x: x[0])
    df['Verdict_Class']    = verdicts.apply(lambda x: x[1])
    df['Trap_Probability'] = verdicts.apply(lambda x: x[2])

    # Sort & Rank
    df = df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))

    # ── TABS ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Rankings", "🔍 Scanner", "📈 Charts", "📥 Export"])

    # ════════════════════════════════════════════════════════
    # TAB 1: RANKINGS
    # ════════════════════════════════════════════════════════
    with tab1:
        verdict_options = df['Verdict'].unique().tolist()
        buy_defaults    = [v for v in verdict_options if 'BUY' in v or 'ENTRY' in v or 'TURN' in v]
        verdict_filter  = st.multiselect("Filter by Verdict", options=verdict_options,
                                         default=buy_defaults if buy_defaults else None)

        filtered = df[df['Verdict'].isin(verdict_filter)] if verdict_filter else df.copy()

        # Columns to show
        base  = ['Rank', 'Name', 'Verdict', 'Final_Score', 'Trap_Probability']
        price = ['Close Price', 'Price To Earnings']
        fund  = ['ROCE', 'Debt To Equity', 'Free Cash Flow']
        scores = ['Score_Momentum', 'Score_Institutional', 'Score_Quality', 'Score_Safety']
        detect = ['Early_Entry_Score', 'Turnaround_Score', 'RSI_Velocity',
                  'Early_Entry_Signals', 'Turnaround_Signals', 'Turnaround_Warnings', 'Industry']

        if show_all:
            cols = base + scores + detect
        else:
            cols = base + price + fund

        cols = [c for c in cols if c in filtered.columns]

        # ── FIX #8: color_row — works correctly with Styler ──
        # Styler.apply(axis=1) passes a Series; use .name to get index
        verdict_color_map = {
            'strong-buy':    'background-color: rgba(16,185,129,0.15)',
            'buy':           'background-color: rgba(59,130,246,0.12)',
            'early-entry':   'background-color: rgba(255,193,7,0.18)',
            'accumulation':  'background-color: rgba(255,152,0,0.15)',
            'turnaround':    'background-color: rgba(156,39,176,0.15)',
            'hold':          'background-color: rgba(245,158,11,0.10)',
            'trap':          'background-color: rgba(239,68,68,0.15)',
            'avoid':         'background-color: rgba(239,68,68,0.08)',
        }

        def color_row(row):
            # row is a Series when axis=1; row.name = the DataFrame index
            vc = filtered.loc[row.name, 'Verdict_Class'] if row.name in filtered.index else 'hold'
            color = verdict_color_map.get(vc, '')
            return [color] * len(row)

        display_df = filtered[cols].head(top_n)

        styler = display_df.style.apply(color_row, axis=1)

        # Format numeric columns safely
        fmt_map = {}
        if 'Final_Score' in cols:        fmt_map['Final_Score']        = '{:.1f}'
        if 'Trap_Probability' in cols:   fmt_map['Trap_Probability']   = '{:.0f}%'
        for sc in scores:
            if sc in cols: fmt_map[sc] = '{:.2f}'
        if 'Early_Entry_Score' in cols:  fmt_map['Early_Entry_Score']  = '{:.2f}'
        if 'Turnaround_Score' in cols:   fmt_map['Turnaround_Score']   = '{:.2f}'
        if 'Close Price' in cols:        fmt_map['Close Price']        = '₹{:.2f}'
        if 'Price To Earnings' in cols:  fmt_map['Price To Earnings']  = '{:.1f}'
        if 'ROCE' in cols:               fmt_map['ROCE']               = '{:.1f}%'
        if 'Debt To Equity' in cols:     fmt_map['Debt To Equity']     = '{:.2f}'

        styler = styler.format(fmt_map, na_rep='—')
        st.dataframe(styler, height=600, use_container_width=True)

        # Quick stats row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("💎 Strong Buy",  len(df[df['Verdict_Class'] == 'strong-buy']))
        c2.metric("📈 Buy",         len(df[df['Verdict_Class'] == 'buy']))
        c3.metric("🎯 Early Entry", len(df[df['Verdict_Class'].isin(['early-entry','accumulation'])]))
        c4.metric("🔄 Turnaround",  len(df[df['Verdict_Class'] == 'turnaround']))
        c5.metric("⚠️ Risky/Trap",  len(df[df['Verdict_Class'] == 'trap']))
        c6.metric("⏸️ Hold",        len(df[df['Verdict_Class'] == 'hold']))

    # ════════════════════════════════════════════════════════
    # TAB 2: CUSTOM SCANNER
    # ════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 🔍 Custom Scanner")
        c1, c2, c3 = st.columns(3)

        with c1:
            min_score   = st.slider("Min Score",   0, 100, 60)
            max_pe      = st.slider("Max P/E",     5, 200, 50)
        with c2:
            min_roe     = st.slider("Min ROE %",   0, 50,  10)
            min_growth  = st.slider("Min PAT Growth %", -50, 100, 0)
        with c3:
            req_fcf     = st.checkbox("Positive FCF Only", value=True)
            req_inst    = st.checkbox("Institutional Buying", value=False)
            req_early   = st.checkbox("Early Entry Only", value=False)

        sc_df = df[df['Final_Score'] >= min_score].copy()

        if 'Price To Earnings' in sc_df.columns:
            sc_df = sc_df[sc_df['Price To Earnings'] <= max_pe]
        if 'ROE' in sc_df.columns:
            sc_df = sc_df[sc_df['ROE'] >= min_roe]
        if 'PAT Growth TTM' in sc_df.columns:
            sc_df = sc_df[sc_df['PAT Growth TTM'] >= min_growth]
        if req_fcf and 'Free Cash Flow' in sc_df.columns:
            sc_df = sc_df[sc_df['Free Cash Flow'] > 0]
        if req_inst and 'Change In FII Holdings Latest Quarter' in sc_df.columns:
            sc_df = sc_df[sc_df['Change In FII Holdings Latest Quarter'] > 0]
        if req_early:
            sc_df = sc_df[sc_df['Verdict_Class'].isin(['early-entry', 'accumulation'])]

        col_a, col_b = st.columns(2)
        col_a.metric("Total Matches", len(sc_df))
        col_b.metric("Avg Score",     f"{sc_df['Final_Score'].mean():.1f}" if len(sc_df) > 0 else "—")

        if len(sc_df) > 0:
            show_cols = ['Rank','Name','Verdict','Final_Score','Close Price',
                         'Price To Earnings','ROE','ROCE','PAT Growth TTM',
                         'Free Cash Flow','Debt To Equity']
            show_cols = [c for c in show_cols if c in sc_df.columns]
            st.dataframe(sc_df[show_cols].head(top_n), height=450, use_container_width=True)
        else:
            st.info("No stocks match the current filters. Try relaxing the criteria.")

    # ════════════════════════════════════════════════════════
    # TAB 3: CHARTS
    # ════════════════════════════════════════════════════════
    with tab3:
        c1, c2 = st.columns(2)

        with c1:
            # ── GARP Scatter ────────────────────────────────
            if 'Price To Earnings' in df.columns and 'PAT Growth TTM' in df.columns:
                plot_df = df.dropna(subset=['Price To Earnings','PAT Growth TTM']).head(100)
                fig = px.scatter(
                    plot_df, x='Price To Earnings', y='PAT Growth TTM',
                    color='Final_Score', size='Final_Score',
                    hover_name='Name', hover_data={'Verdict': True, 'Final_Score': ':.1f'},
                    log_x=True, title="📊 Growth at Reasonable Price (GARP)",
                    color_continuous_scale='Viridis', height=380
                )
                fig.add_hline(y=20,  line_dash="dash", line_color="green", annotation_text="Growth 20%")
                fig.add_vline(x=25,  line_dash="dash", line_color="red",   annotation_text="P/E 25")
                fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

            # ── Score Distribution ──────────────────────────
            fig = px.histogram(df, x='Final_Score', nbins=30,
                               title="📊 Score Distribution", height=320,
                               color_discrete_sequence=['#3b82f6'])
            fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="Buy zone")
            fig.add_vline(x=35, line_dash="dash", line_color="red",   annotation_text="Avoid zone")
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

            # ── Trap Probability Distribution ───────────────
            fig = px.histogram(df, x='Trap_Probability', nbins=25,
                               title="🛡️ Trap Probability Distribution", height=300,
                               color_discrete_sequence=['#ef4444'])
            fig.add_vline(x=60, line_dash="dash", line_color="purple", annotation_text="Danger 60%")
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # ── Radar: Top 3 stocks ─────────────────────────
            if len(df) >= 3:
                categories = ['Momentum','Institutional','Quality','Safety','Early Entry','Turnaround']
                fig = go.Figure()

                colors = ['#10b981', '#3b82f6', '#f59e0b']
                for rank_idx in range(3):
                    row = df.iloc[rank_idx]
                    vals = [
                        row.get('Score_Momentum', 0.5),
                        row.get('Score_Institutional', 0.5),
                        row.get('Score_Quality', 0.5),
                        row.get('Score_Safety', 0.5),
                        row.get('Early_Entry_Score', 0),
                        row.get('Turnaround_Score', 0),
                    ]
                    fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=categories + [categories[0]],
                        fill='toself', name=f"#{rank_idx+1} {row.get('Name','?')}",
                        line_color=colors[rank_idx]
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="🎯 Top 3 Stocks — Factor Radar",
                    height=380, legend=dict(orientation='h', yanchor='bottom', y=-0.2)
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Momentum Matrix ─────────────────────────────
            if 'RSI 14W' in df.columns and 'ADX 14W' in df.columns:
                fig = px.scatter(
                    df.head(80), x='RSI 14W', y='ADX 14W',
                    color='Final_Score', hover_name='Name',
                    hover_data={'Verdict': True},
                    title="⚡ Momentum Matrix (RSI vs ADX)",
                    color_continuous_scale='Viridis', height=330
                )
                # Green zone: RSI 50-70, ADX 25-50 = strong confirmed momentum
                fig.add_shape(type="rect", x0=50, y0=25, x1=70, y1=50,
                              line=dict(color="green", width=2),
                              fillcolor="green", opacity=0.08)
                fig.add_annotation(x=60, y=48, text="Sweet Spot",
                                   showarrow=False, font=dict(color="green", size=11))
                fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

            # ── Verdict Breakdown Pie ───────────────────────
            verdict_counts = df['Verdict_Class'].value_counts()
            color_map = {
                'strong-buy': '#10b981', 'buy': '#3b82f6',
                'early-entry': '#ffc107', 'accumulation': '#ff9800',
                'turnaround': '#9c27b0', 'hold': '#f59e0b',
                'trap': '#ef4444', 'avoid': '#dc2626',
            }
            fig = px.pie(
                verdict_counts.reset_index(),
                names='Verdict_Class', values='count',
                color='Verdict_Class', color_discrete_map=color_map,
                title="📊 Verdict Breakdown", height=320
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 4: EXPORT
    # ════════════════════════════════════════════════════════
    with tab4:
        st.markdown("#### 📥 Download Reports")
        ts = datetime.now().strftime('%Y%m%d_%H%M')

        # Row 1: Main exports
        e1, e2, e3 = st.columns(3)

        e1.download_button(
            "📄 Full Analysis",
            df.to_csv(index=False).encode('utf-8'),
            f"V4_Full_{ts}.csv", "text/csv",
            use_container_width=True
        )

        buy_df = df[df['Verdict_Class'].isin(['strong-buy','buy'])]
        if len(buy_df) > 0:
            e2.download_button(
                "📈 Buy Signals Only",
                buy_df.to_csv(index=False).encode('utf-8'),
                f"V4_Buys_{ts}.csv", "text/csv",
                use_container_width=True
            )
        else:
            e2.info("No buy signals")

        trap_df = df[df['Verdict_Class'] == 'trap']
        if len(trap_df) > 0:
            e3.download_button(
                "⚠️ Traps & Avoid",
                trap_df.to_csv(index=False).encode('utf-8'),
                f"V4_Traps_{ts}.csv", "text/csv",
                use_container_width=True
            )
        else:
            e3.info("No traps detected")

        # Row 2: Opportunity exports
        e4, e5 = st.columns(2)

        early_df = df[df['Verdict_Class'].isin(['early-entry','accumulation'])]
        if len(early_df) > 0:
            e4.download_button(
                "🎯 Early Entry Picks",
                early_df.to_csv(index=False).encode('utf-8'),
                f"V4_EarlyEntry_{ts}.csv", "text/csv",
                use_container_width=True
            )
        else:
            e4.info("No early entry signals")

        turn_df = df[df['Verdict_Class'] == 'turnaround']
        if len(turn_df) > 0:
            e5.download_button(
                "🔄 Turnaround Picks",
                turn_df.to_csv(index=False).encode('utf-8'),
                f"V4_Turnarounds_{ts}.csv", "text/csv",
                use_container_width=True
            )
        else:
            e5.info("No turnaround signals")

        # Summary stats card
        st.markdown("---")
        st.markdown("#### 📋 Export Summary")
        summary_data = {
            'Category': ['Total Stocks','Strong Buy','Buy','Early Entry','Accumulation',
                         'Turnaround','Hold','Risky/Trap','Avoid'],
            'Count': [
                len(df),
                len(df[df['Verdict_Class']=='strong-buy']),
                len(df[df['Verdict_Class']=='buy']),
                len(df[df['Verdict_Class']=='early-entry']),
                len(df[df['Verdict_Class']=='accumulation']),
                len(df[df['Verdict_Class']=='turnaround']),
                len(df[df['Verdict_Class']=='hold']),
                len(df[df['Verdict_Class']=='trap']),
                len(df[df['Verdict_Class']=='avoid']),
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=False)


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
