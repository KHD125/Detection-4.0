import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# 🚀 V4 ULTRA+ - ENHANCED QUANT DATA-DRIVEN ANALYZER
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
# V4 ULTRA+ ENHANCEMENTS (Addressing Weaknesses):
#   ✓ 4 Core Factors (Momentum 60%, Institutional 20%, Quality 10%, Safety 10%)
#   ✓ Z-Score normalization (adaptive to any market)
#   ✓ EARLY ENTRY DETECTION - Catch accumulation before breakout
#   ✓ TURNAROUND DETECTION - Identify recovering stocks
#   ✓ MOMENTUM VELOCITY - 2nd derivative catches breakouts early
#   ✓ FII ACCUMULATION PHASE - Smart money entering quietly
#   ✓ SEQUENTIAL IMPROVEMENT - QoQ trajectory matters
#   ✓ Enhanced Trap detection (OCF/FCF/Debt)
# =========================================================

st.set_page_config(
    page_title="V4 ULTRA+ | Advanced Stock Analyzer",
    page_icon="🚀",
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
    .verdict-early-entry {
        background: linear-gradient(135deg, #ffc107, #ff9800);
        color: #1a1a1a;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .verdict-accumulation {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .verdict-turnaround {
        background: linear-gradient(135deg, #9c27b0, #7b1fa2);
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
# 🎯 SECTOR-AWARE ADAPTIVE THRESHOLDS
# =========================================================
# Different sectors have VERY different "normal" ranges!
# Banking: High D/E is normal (they borrow to lend)
# IT: High PE is normal (growth premium)
# Infra: Low ROE is normal (capital intensive)
# FMCG: High PE, low D/E is normal
#
# This prevents "false positives" from hardcoded thresholds
# =========================================================

SECTOR_THRESHOLDS = {
    # Format: 'sector_keyword': {'metric': (danger_threshold, normal_threshold), ...}
    # danger_threshold = absolute red flag
    # normal_threshold = acceptable for this sector
    
    'default': {
        'debt_to_equity': {'danger': 3.0, 'high': 2.0, 'normal': 1.0},
        'pe': {'danger': 100, 'high': 50, 'normal': 25},
        'roe': {'min_good': 12, 'min_acceptable': 8},
        'roce': {'min_good': 15, 'min_acceptable': 10},
        'opm': {'min_good': 12, 'min_acceptable': 8},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 40, 'high': 60},
        'promoter_min': 25,
    },
    
    # BANKING & NBFC - High leverage is their business model
    'bank': {
        'debt_to_equity': {'danger': 15.0, 'high': 12.0, 'normal': 8.0},  # Banks borrow to lend!
        'pe': {'danger': 50, 'high': 25, 'normal': 15},
        'roe': {'min_good': 14, 'min_acceptable': 10},
        'roce': {'min_good': 2, 'min_acceptable': 1},  # ROCE meaningless for banks
        'opm': {'min_good': 20, 'min_acceptable': 15},  # NIM based
        'fii_accum': {'strong': 0.3, 'moderate': 0.2, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 35, 'high': 55},  # Banks move slower
        'promoter_min': 20,  # PSU banks have lower promoter
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
    'finance': {  # Alias for banking
        'debt_to_equity': {'danger': 12.0, 'high': 8.0, 'normal': 5.0},
        'pe': {'danger': 50, 'high': 30, 'normal': 18},
        'roe': {'min_good': 14, 'min_acceptable': 10},
        'roce': {'min_good': 2, 'min_acceptable': 1},
        'opm': {'min_good': 20, 'min_acceptable': 15},
        'fii_accum': {'strong': 0.3, 'moderate': 0.2, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 35, 'high': 55},
        'promoter_min': 20,
    },
    
    # IT & TECH - High PE is growth premium, low debt
    'it': {
        'debt_to_equity': {'danger': 1.0, 'high': 0.5, 'normal': 0.2},  # Should be nearly debt-free
        'pe': {'danger': 80, 'high': 50, 'normal': 30},  # High PE is normal
        'roe': {'min_good': 20, 'min_acceptable': 15},
        'roce': {'min_good': 25, 'min_acceptable': 18},
        'opm': {'min_good': 20, 'min_acceptable': 15},
        'fii_accum': {'strong': 0.4, 'moderate': 0.25, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 42, 'high': 62},
        'promoter_min': 30,
    },
    'software': {  # Alias for IT
        'debt_to_equity': {'danger': 1.0, 'high': 0.5, 'normal': 0.2},
        'pe': {'danger': 80, 'high': 50, 'normal': 30},
        'roe': {'min_good': 20, 'min_acceptable': 15},
        'roce': {'min_good': 25, 'min_acceptable': 18},
        'opm': {'min_good': 20, 'min_acceptable': 15},
        'fii_accum': {'strong': 0.4, 'moderate': 0.25, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 42, 'high': 62},
        'promoter_min': 30,
    },
    'technology': {  # Alias for IT
        'debt_to_equity': {'danger': 1.0, 'high': 0.5, 'normal': 0.2},
        'pe': {'danger': 80, 'high': 50, 'normal': 30},
        'roe': {'min_good': 20, 'min_acceptable': 15},
        'roce': {'min_good': 25, 'min_acceptable': 18},
        'opm': {'min_good': 20, 'min_acceptable': 15},
        'fii_accum': {'strong': 0.4, 'moderate': 0.25, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 42, 'high': 62},
        'promoter_min': 30,
    },
    
    # FMCG & CONSUMER - Stable, low debt, premium valuations
    'fmcg': {
        'debt_to_equity': {'danger': 1.5, 'high': 0.8, 'normal': 0.3},
        'pe': {'danger': 100, 'high': 60, 'normal': 40},  # Pays for stability
        'roe': {'min_good': 25, 'min_acceptable': 18},
        'roce': {'min_good': 30, 'min_acceptable': 22},
        'opm': {'min_good': 18, 'min_acceptable': 14},
        'fii_accum': {'strong': 0.3, 'moderate': 0.2, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 45, 'high': 65},
        'promoter_min': 40,
    },
    'consumer': {  # Alias
        'debt_to_equity': {'danger': 1.5, 'high': 0.8, 'normal': 0.3},
        'pe': {'danger': 100, 'high': 60, 'normal': 40},
        'roe': {'min_good': 25, 'min_acceptable': 18},
        'roce': {'min_good': 30, 'min_acceptable': 22},
        'opm': {'min_good': 18, 'min_acceptable': 14},
        'fii_accum': {'strong': 0.3, 'moderate': 0.2, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 45, 'high': 65},
        'promoter_min': 40,
    },
    
    # PHARMA - R&D intensive, moderate debt
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
    'healthcare': {  # Alias
        'debt_to_equity': {'danger': 2.0, 'high': 1.0, 'normal': 0.5},
        'pe': {'danger': 80, 'high': 45, 'normal': 25},
        'roe': {'min_good': 18, 'min_acceptable': 12},
        'roce': {'min_good': 18, 'min_acceptable': 12},
        'opm': {'min_good': 20, 'min_acceptable': 15},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 40, 'high': 60},
        'promoter_min': 35,
    },
    
    # INFRASTRUCTURE & CAPITAL GOODS - High debt, low margins normal
    'infrastructure': {
        'debt_to_equity': {'danger': 4.0, 'high': 2.5, 'normal': 1.5},  # Capital intensive
        'pe': {'danger': 60, 'high': 35, 'normal': 20},
        'roe': {'min_good': 12, 'min_acceptable': 8},  # Lower ROE normal
        'roce': {'min_good': 12, 'min_acceptable': 8},
        'opm': {'min_good': 12, 'min_acceptable': 8},  # Lower margins normal
        'fii_accum': {'strong': 0.6, 'moderate': 0.35, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 38, 'high': 58},
        'promoter_min': 30,
    },
    'capital goods': {
        'debt_to_equity': {'danger': 3.5, 'high': 2.0, 'normal': 1.2},
        'pe': {'danger': 70, 'high': 40, 'normal': 25},
        'roe': {'min_good': 15, 'min_acceptable': 10},
        'roce': {'min_good': 15, 'min_acceptable': 10},
        'opm': {'min_good': 14, 'min_acceptable': 10},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 40, 'high': 60},
        'promoter_min': 30,
    },
    'construction': {
        'debt_to_equity': {'danger': 4.0, 'high': 2.5, 'normal': 1.5},
        'pe': {'danger': 50, 'high': 30, 'normal': 18},
        'roe': {'min_good': 14, 'min_acceptable': 10},
        'roce': {'min_good': 12, 'min_acceptable': 8},
        'opm': {'min_good': 10, 'min_acceptable': 7},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 38, 'high': 58},
        'promoter_min': 35,
    },
    
    # METALS & MINING - Cyclical, high debt during expansion
    'metal': {
        'debt_to_equity': {'danger': 3.5, 'high': 2.0, 'normal': 1.2},
        'pe': {'danger': 30, 'high': 15, 'normal': 8},  # Cyclical = low PE
        'roe': {'min_good': 15, 'min_acceptable': 10},
        'roce': {'min_good': 15, 'min_acceptable': 10},
        'opm': {'min_good': 20, 'min_acceptable': 12},
        'fii_accum': {'strong': 0.6, 'moderate': 0.35, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 35, 'high': 55},  # More volatile
        'promoter_min': 30,
    },
    'mining': {  # Alias
        'debt_to_equity': {'danger': 3.5, 'high': 2.0, 'normal': 1.2},
        'pe': {'danger': 30, 'high': 15, 'normal': 8},
        'roe': {'min_good': 15, 'min_acceptable': 10},
        'roce': {'min_good': 15, 'min_acceptable': 10},
        'opm': {'min_good': 20, 'min_acceptable': 12},
        'fii_accum': {'strong': 0.6, 'moderate': 0.35, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 35, 'high': 55},
        'promoter_min': 30,
    },
    'steel': {  # Alias
        'debt_to_equity': {'danger': 3.5, 'high': 2.0, 'normal': 1.2},
        'pe': {'danger': 30, 'high': 15, 'normal': 8},
        'roe': {'min_good': 15, 'min_acceptable': 10},
        'roce': {'min_good': 15, 'min_acceptable': 10},
        'opm': {'min_good': 20, 'min_acceptable': 12},
        'fii_accum': {'strong': 0.6, 'moderate': 0.35, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 35, 'high': 55},
        'promoter_min': 30,
    },
    
    # POWER & UTILITIES - Regulated, high debt normal
    'power': {
        'debt_to_equity': {'danger': 5.0, 'high': 3.0, 'normal': 2.0},  # Capital intensive, regulated
        'pe': {'danger': 40, 'high': 20, 'normal': 12},
        'roe': {'min_good': 12, 'min_acceptable': 8},
        'roce': {'min_good': 10, 'min_acceptable': 7},
        'opm': {'min_good': 25, 'min_acceptable': 18},
        'fii_accum': {'strong': 0.4, 'moderate': 0.25, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 40, 'high': 58},
        'promoter_min': 40,  # Often PSU
    },
    'utilities': {  # Alias
        'debt_to_equity': {'danger': 5.0, 'high': 3.0, 'normal': 2.0},
        'pe': {'danger': 40, 'high': 20, 'normal': 12},
        'roe': {'min_good': 12, 'min_acceptable': 8},
        'roce': {'min_good': 10, 'min_acceptable': 7},
        'opm': {'min_good': 25, 'min_acceptable': 18},
        'fii_accum': {'strong': 0.4, 'moderate': 0.25, 'weak': 0.1},
        'rsi_sweet_spot': {'low': 40, 'high': 58},
        'promoter_min': 40,
    },
    
    # AUTO - Cyclical, moderate debt
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
    'automobile': {  # Alias
        'debt_to_equity': {'danger': 2.5, 'high': 1.5, 'normal': 0.8},
        'pe': {'danger': 60, 'high': 35, 'normal': 20},
        'roe': {'min_good': 18, 'min_acceptable': 12},
        'roce': {'min_good': 18, 'min_acceptable': 12},
        'opm': {'min_good': 14, 'min_acceptable': 10},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 40, 'high': 60},
        'promoter_min': 35,
    },
    
    # REALTY - Very high debt normal, cyclical
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
    'real estate': {  # Alias
        'debt_to_equity': {'danger': 4.0, 'high': 2.5, 'normal': 1.5},
        'pe': {'danger': 50, 'high': 30, 'normal': 15},
        'roe': {'min_good': 12, 'min_acceptable': 8},
        'roce': {'min_good': 10, 'min_acceptable': 6},
        'opm': {'min_good': 25, 'min_acceptable': 18},
        'fii_accum': {'strong': 0.5, 'moderate': 0.3, 'weak': 0.15},
        'rsi_sweet_spot': {'low': 35, 'high': 55},
        'promoter_min': 40,
    },
    
    # CHEMICALS - Moderate debt, decent margins
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
    
    # TEXTILES - Low margins, moderate debt
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

def get_sector_thresholds(industry):
    """
    Get sector-specific thresholds based on industry name.
    Uses fuzzy matching to find the best sector fit.
    """
    if not industry or pd.isna(industry):
        return SECTOR_THRESHOLDS['default']
    
    industry_lower = str(industry).lower().strip()
    
    # Direct match first
    if industry_lower in SECTOR_THRESHOLDS:
        return SECTOR_THRESHOLDS[industry_lower]
    
    # Fuzzy match - check if any keyword is in the industry name
    for sector_key in SECTOR_THRESHOLDS.keys():
        if sector_key != 'default' and sector_key in industry_lower:
            return SECTOR_THRESHOLDS[sector_key]
    
    # Check for common patterns
    if any(kw in industry_lower for kw in ['bank', 'lending', 'credit', 'loan']):
        return SECTOR_THRESHOLDS['bank']
    if any(kw in industry_lower for kw in ['software', 'tech', 'digital', 'internet', 'saas']):
        return SECTOR_THRESHOLDS['it']
    if any(kw in industry_lower for kw in ['pharma', 'drug', 'biotech', 'hospital', 'health']):
        return SECTOR_THRESHOLDS['pharma']
    if any(kw in industry_lower for kw in ['fmcg', 'consumer', 'food', 'beverage', 'personal']):
        return SECTOR_THRESHOLDS['fmcg']
    if any(kw in industry_lower for kw in ['infra', 'construct', 'engineering', 'capital']):
        return SECTOR_THRESHOLDS['infrastructure']
    if any(kw in industry_lower for kw in ['metal', 'steel', 'aluminium', 'copper', 'mining']):
        return SECTOR_THRESHOLDS['metal']
    if any(kw in industry_lower for kw in ['power', 'electric', 'energy', 'utility']):
        return SECTOR_THRESHOLDS['power']
    if any(kw in industry_lower for kw in ['auto', 'vehicle', 'motor', 'tyre']):
        return SECTOR_THRESHOLDS['auto']
    if any(kw in industry_lower for kw in ['real', 'property', 'housing']):
        return SECTOR_THRESHOLDS['realty']
    if any(kw in industry_lower for kw in ['chemical', 'petrochem', 'specialty']):
        return SECTOR_THRESHOLDS['chemical']
    if any(kw in industry_lower for kw in ['textile', 'apparel', 'garment', 'fabric']):
        return SECTOR_THRESHOLDS['textile']
    if any(kw in industry_lower for kw in ['nbfc', 'financial service', 'insurance']):
        return SECTOR_THRESHOLDS['nbfc']
    
    return SECTOR_THRESHOLDS['default']

# =========================================================
# 🚨 ONE-OFF INCOME DETECTION
# =========================================================
# Catches "fake" turnarounds where profit jumped due to:
#   - Asset sales (Other Income)
#   - One-time gains
#   - Accounting tricks
#
# Key insight: If PAT grows but Revenue doesn't, it's likely ONE-OFF!
# =========================================================

def detect_one_off_income(pat_growth, revenue_growth, opm_current, opm_expected=None):
    """
    Detect if profit growth is likely from one-off income rather than business.
    
    Returns: (is_suspicious, confidence, reason)
    
    Logic:
    1. PAT growth >> Revenue growth = likely other income
    2. PAT positive but Revenue negative = very suspicious
    3. OPM collapse but PAT up = definitely other income
    """
    if pd.isna(pat_growth) or pd.isna(revenue_growth):
        return False, 0, "INSUFFICIENT_DATA"
    
    is_suspicious = False
    confidence = 0
    reasons = []
    
    # Case 1: PAT growing much faster than Revenue (>2x difference)
    if pat_growth > 20 and revenue_growth < pat_growth * 0.5:
        is_suspicious = True
        confidence += 40
        reasons.append("PAT>>REV")
    
    # Case 2: PAT positive but Revenue negative = RED FLAG!
    if pat_growth > 10 and revenue_growth < 0:
        is_suspicious = True
        confidence += 50
        reasons.append("PAT+REV-")
    
    # Case 3: Huge PAT jump with flat revenue
    if pat_growth > 50 and abs(revenue_growth) < 5:
        is_suspicious = True
        confidence += 35
        reasons.append("PAT_SPIKE")
    
    # Case 4: OPM lower than expected but PAT up (other income propping up)
    if opm_expected and opm_current:
        if opm_current < opm_expected * 0.7 and pat_growth > 15:
            is_suspicious = True
            confidence += 30
            reasons.append("OPM_WEAK")
    
    confidence = min(confidence, 100)
    reason = ','.join(reasons) if reasons else "CLEAN"
    
    return is_suspicious, confidence, reason

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
    
    # FII Multi-Period (for accumulation detection)
    'fii_multiperiod': [
        'Change In FII Holdings Latest Quarter', 'Change In FII Holdings 1 Year'
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
    
    # Multi-period FII (for accumulation detection)
    fii_chg_1y, has_fii_chg_1y = safe_get('Change In FII Holdings 1 Year', 0)
    
    # RSI Daily (for velocity calculation)
    rsi_d, has_rsi_d = safe_get('RSI 14D', 50)
    
    # Additional Growth for Turnaround Detection
    pat_qoq, has_pat_qoq = safe_get('PAT Growth QoQ', 0)
    rev_qoq, has_rev_qoq = safe_get('Revenue Growth QoQ', 0)
    
    # ═══════════════════════════════════════════════════════
    # V4 ULTRA+: ENHANCED QUANT 4-FACTOR Z-SCORE ENGINE
    # ═══════════════════════════════════════════════════════
    # 
    # 4 CORE FACTORS (data-driven, no style classification):
    #   1. MOMENTUM (60%): RSI + Returns + 52WH Distance + Acceleration
    #   2. INSTITUTIONAL (20%): FII flows (smart money)
    #   3. QUALITY (10%): ROCE only (the one that matters)
    #   4. SAFETY (10%): Cash flow + Debt (trap avoidance)
    #
    # ENHANCED DETECTION (Addressing Weaknesses):
    #   + EARLY ENTRY: FII accumulating + RSI building + low returns = catch before breakout
    #   + TURNAROUND: Sequential QoQ improvement + margin expansion + debt reduction
    #   + RSI VELOCITY: Rate of RSI change catches momentum building
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
    # RSI VELOCITY (Rate of change - catches momentum BUILDING)
    # Addresses "Late to Party" weakness
    # ─────────────────────────────────────────────────────────
    if has_rsi_w and has_rsi_d:
        # RSI velocity: weekly vs daily divergence indicates momentum building
        rsi_velocity = rsi_w - rsi_d  # Positive = weekly stronger = momentum building
        has_rsi_velocity = True
    else:
        rsi_velocity = pd.Series([0.0] * n, index=df.index)
        has_rsi_velocity = False
    
    df['RSI_Velocity'] = rsi_velocity
    
    # ─────────────────────────────────────────────────────────
    # EARLY ENTRY SCORE (Catch accumulation before breakout)
    # Addresses "Late to Party" weakness
    # NOW WITH SECTOR-AWARE THRESHOLDS!
    # ─────────────────────────────────────────────────────────
    
    # Get industry column for sector-aware thresholds
    has_industry = 'Industry' in df.columns
    
    def calculate_early_entry_score(idx):
        """
        Detect ACCUMULATION PHASE before price breakout.
        High score = Smart money entering quietly.
        NOW USES SECTOR-SPECIFIC THRESHOLDS!
        """
        score = 0
        signals = []
        
        # Get sector thresholds
        industry_val = df['Industry'].iloc[idx] if has_industry else None
        thresholds = get_sector_thresholds(industry_val)
        fii_thresh = thresholds['fii_accum']
        rsi_range = thresholds['rsi_sweet_spot']
        
        # Signal 1: FII accumulating (SECTOR-ADJUSTED threshold)
        fii_q = fii_chg.iloc[idx] if hasattr(fii_chg, 'iloc') else fii_chg[idx]
        if fii_q > fii_thresh['strong']:
            score += 25
            signals.append('FII_ACCUM')
        elif fii_q > fii_thresh['moderate']:
            score += 15
            signals.append('FII_MOD')
        elif fii_q > fii_thresh['weak']:
            score += 8
        
        # Signal 2: FII consistently buying (1Y trend positive)
        if has_fii_chg_1y:
            fii_1y = fii_chg_1y.iloc[idx] if hasattr(fii_chg_1y, 'iloc') else fii_chg_1y[idx]
            if fii_1y > 1 and fii_q > 0:
                score += 20
                signals.append('FII_TREND')
        
        # Signal 3: RSI in SECTOR-ADJUSTED sweet spot
        rsi_val = rsi_w.iloc[idx] if hasattr(rsi_w, 'iloc') else rsi_w[idx]
        rsi_low = rsi_range['low']
        rsi_high = rsi_range['high']
        if rsi_low <= rsi_val <= rsi_high:
            score += 20
            signals.append('RSI_READY')
        elif (rsi_low - 5) <= rsi_val < rsi_low or rsi_high < rsi_val <= (rsi_high + 5):
            score += 10
        
        # Signal 4: Price hasn't run yet (Returns 3M < 15%)
        ret_3m_val = ret_3m.iloc[idx] if hasattr(ret_3m, 'iloc') else ret_3m[idx]
        if -5 <= ret_3m_val <= 15:
            score += 20
            signals.append('NOT_EXTENDED')
        elif ret_3m_val < -5:
            score += 5  # Oversold but not ideal
        
        # Signal 5: Momentum acceleration positive (building)
        mom_acc = momentum_acceleration.iloc[idx] if hasattr(momentum_acceleration, 'iloc') else momentum_acceleration[idx]
        if mom_acc > 0:
            score += 15
            signals.append('MOM_BUILDING')
        
        return score / 100, signals
    
    early_entry_results = [calculate_early_entry_score(i) for i in range(n)]
    df['Early_Entry_Score'] = [r[0] for r in early_entry_results]
    df['Early_Entry_Signals'] = [','.join(r[1]) if r[1] else '' for r in early_entry_results]
    
    # ─────────────────────────────────────────────────────────
    # TURNAROUND SCORE (Identify recovering companies)
    # Addresses "Dislikes Turnarounds" weakness
    # NOW WITH ONE-OFF INCOME DETECTION & SECTOR THRESHOLDS!
    # ─────────────────────────────────────────────────────────
    def calculate_turnaround_score(idx):
        """
        Detect RECOVERY trajectory in beaten-down stocks.
        High score = Company improving even if past numbers look bad.
        
        NOW INCLUDES:
        - Sector-aware debt thresholds (banks can have high D/E)
        - One-off income detection (catches fake turnarounds)
        """
        score = 0
        signals = []
        warnings = []
        
        # Get sector thresholds
        industry_val = df['Industry'].iloc[idx] if has_industry else None
        thresholds = get_sector_thresholds(industry_val)
        de_thresh = thresholds['debt_to_equity']
        opm_thresh = thresholds['opm']
        
        # Get values
        pat_qoq_val = pat_qoq.iloc[idx] if hasattr(pat_qoq, 'iloc') else pat_qoq[idx]
        pat_yoy_val = pat_yoy.iloc[idx] if hasattr(pat_yoy, 'iloc') else pat_yoy[idx]
        rev_qoq_val = rev_qoq.iloc[idx] if hasattr(rev_qoq, 'iloc') else rev_qoq[idx]
        ret_3m_val = ret_3m.iloc[idx] if hasattr(ret_3m, 'iloc') else ret_3m[idx]
        ret_1y_val = ret_1y.iloc[idx] if hasattr(ret_1y, 'iloc') else ret_1y[idx]
        prom_chg_val = prom_chg.iloc[idx] if hasattr(prom_chg, 'iloc') else prom_chg[idx]
        de_val = de.iloc[idx] if hasattr(de, 'iloc') else de[idx]
        opm_val = opm.iloc[idx] if hasattr(opm, 'iloc') else opm[idx]
        ocf_val = ocf.iloc[idx] if hasattr(ocf, 'iloc') else ocf[idx]
        
        # Only consider if stock has been weak (potential turnaround candidate)
        is_beaten_down = ret_1y_val < 0 or ret_3m_val < -5
        
        if not is_beaten_down:
            return 0, [], []  # Not a turnaround candidate
        
        # 🚨 ONE-OFF INCOME CHECK - Critical for avoiding "falling knife"
        is_one_off, one_off_confidence, one_off_reason = detect_one_off_income(
            pat_qoq_val, 
            rev_qoq_val, 
            opm_val,
            opm_thresh['min_acceptable']
        )
        
        if is_one_off and one_off_confidence >= 50:
            # HIGH CONFIDENCE one-off income - heavily penalize
            warnings.append(f'ONE_OFF({one_off_reason})')
            score -= 30  # Penalty for likely fake turnaround
        elif is_one_off and one_off_confidence >= 30:
            # MODERATE confidence - add warning but don't kill score
            warnings.append(f'CHECK_INCOME({one_off_reason})')
            score -= 10
        
        # Signal 1: PAT improving QoQ (even if negative YoY)
        # BUT only if backed by Revenue growth (not one-off)
        if pat_qoq_val > pat_yoy_val and pat_qoq_val > 0:
            if rev_qoq_val > 0:  # Revenue also growing = REAL turnaround
                score += 30
                signals.append('REAL_TURNAROUND')
            elif not is_one_off:
                score += 20
                signals.append('PAT_ACCEL')
            else:
                score += 5  # Discounted due to one-off suspicion
        elif pat_qoq_val > 0:
            score += 10
            signals.append('PAT_POS_QOQ')
        
        # Signal 2: REVENUE growing (more reliable than PAT)
        if rev_qoq_val > 5:
            score += 20
            signals.append('REV_GROWING')
        elif rev_qoq_val > 0:
            score += 10
        
        # Signal 3: Margin expansion (OPM improving) - SECTOR ADJUSTED
        if opm_val > opm_thresh['min_good']:
            score += 15
            signals.append('MARGIN_OK')
        elif opm_val > opm_thresh['min_acceptable']:
            score += 8
        
        # Signal 4: Promoter BUYING during stress (strong conviction)
        if prom_chg_val > 0.5 and ret_3m_val < 0:
            score += 25
            signals.append('INSIDER_BUY')
        elif prom_chg_val > 0:
            score += 10
        
        # Signal 5: Debt under control - SECTOR ADJUSTED!
        # Banks can have D/E of 8-10, while IT should be near 0
        if de_val < de_thresh['normal']:
            score += 15
            signals.append('LOW_DEBT')
        elif de_val < de_thresh['high']:
            score += 8
        elif de_val > de_thresh['danger']:
            score -= 10
            warnings.append('EXCESS_DEBT')
        
        # Signal 6: Operating cash flow positive (real recovery, not accounting)
        if ocf_val > 0:
            score += 20
            signals.append('OCF_POS')
        
        # Ensure score is in valid range
        final_score = max(0, min(score / 100, 1.0))
        
        return final_score, signals, warnings
    
    turnaround_results = [calculate_turnaround_score(i) for i in range(n)]
    df['Turnaround_Score'] = [r[0] for r in turnaround_results]
    df['Turnaround_Signals'] = [','.join(r[1]) if r[1] else '' for r in turnaround_results]
    df['Turnaround_Warnings'] = [','.join(r[2]) if r[2] else '' for r in turnaround_results]
    
    # ─────────────────────────────────────────────────────────
    # QUALITY GATES (Binary trap filter - 0 to 1)
    # NOW WITH SECTOR-AWARE THRESHOLDS!
    # ─────────────────────────────────────────────────────────
    def calculate_quality_gate(idx):
        gates_passed = 0
        total_gates = 5
        
        # Get sector thresholds
        industry_val = df['Industry'].iloc[idx] if has_industry else None
        thresholds = get_sector_thresholds(industry_val)
        
        # Gate 1: Positive Operating Cash Flow
        ocf_val = ocf.iloc[idx] if hasattr(ocf, 'iloc') else ocf[idx]
        if ocf_val > 0:
            gates_passed += 1
        
        # Gate 2: ROE > Cost of Equity (SECTOR ADJUSTED)
        roe_val = roe.iloc[idx] if hasattr(roe, 'iloc') else roe[idx]
        roe_min = thresholds['roe']['min_acceptable']
        if roe_val > roe_min:
            gates_passed += 1
        
        # Gate 3: Debt not dangerous (SECTOR ADJUSTED!)
        de_val = de.iloc[idx] if hasattr(de, 'iloc') else de[idx]
        de_high = thresholds['debt_to_equity']['high']
        if de_val < de_high:
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
        (get_z_score(pd.Series(rsi_adjusted, index=df.index), available=has_rsi_w), 0.35, has_rsi_w),  # RSI = KING
        (get_z_score(ret_3m, available=has_ret_3m), 0.25, has_ret_3m),  # Returns 3M
        (get_z_score(dist_52wh, lower_better=True, available=has_52wh), 0.20, has_52wh),  # Near 52WH = good
        (get_z_score(momentum_acceleration.clip(-30, 30), available=has_mom_accel), 0.12, has_mom_accel),  # Acceleration
        (get_z_score(rsi_velocity.clip(-20, 20), available=has_rsi_velocity), 0.08, has_rsi_velocity),  # RSI Velocity (early signal)
    ]
    df['Score_Momentum'] = weighted_avg(momentum_components)
    
    # ═══════════════════════════════════════════════════════
    # FACTOR 2: INSTITUTIONAL SCORE (20% - Smart Money)
    # FII +0.499 (follow them), DII negative (contrarian)
    # ═══════════════════════════════════════════════════════
    
    # FII Accumulation trend (1Y consistent buying is stronger signal)
    if has_fii_chg and has_fii_chg_1y:
        fii_accumulation = (fii_chg + fii_chg_1y * 0.5).clip(-5, 5)  # Recent + trend
        has_fii_accum = True
    else:
        fii_accumulation = fii_chg
        has_fii_accum = has_fii_chg
    
    inst_components = [
        (get_z_score(fii_accumulation, available=has_fii_accum), 0.65, has_fii_accum),  # FII accumulation pattern
        (get_z_score(prom_chg, available=has_prom_chg), 0.35, has_prom_chg),  # Promoter conviction
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
    total_expected = 30  # Enhanced tracking with new signals
    data_available = sum([
        has_roe, has_roce, has_opm,
        has_pat_ttm, has_pat_yoy, has_rev_ttm, has_rev_yoy, has_eps_ttm,
        has_pe, has_ps,
        has_de, has_promoter, has_fcf, has_ocf, has_cash, has_debt,
        has_rsi_w, has_adx_w, has_ret_1m, has_ret_3m, has_ret_6m, has_ret_1y,
        has_fii, has_dii, has_fii_chg, has_dii_chg, has_prom_chg,
        has_52wh, has_vs_nifty,
        has_mom_accel,  # Momentum acceleration
        has_rsi_velocity,  # RSI velocity (new)
        has_fii_chg_1y,  # FII trend (new)
        has_pat_qoq,  # PAT QoQ for turnaround (new)
    ])
    df['Data_Coverage'] = f"{data_available}/{total_expected}"
    
    return df

# =========================================================
# 🎯 V4 ULTRA VERDICT ENGINE - SIMPLE PERCENTILE BASED
# =========================================================
def get_ultimate_verdict(row):
    """
    V4 ULTRA+ Enhanced Verdict System:
    
    1. TRAP DETECTION (Safety Net)
    2. EARLY ENTRY DETECTION (Catch accumulation phase)
    3. TURNAROUND DETECTION (Identify recovery plays)
    4. PURE PERCENTILE VERDICT (no style complexity)
    
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
    
    # NEW: Early Entry & Turnaround Scores
    early_entry_score = row.get('Early_Entry_Score', 0)
    early_entry_signals = row.get('Early_Entry_Signals', '')
    turnaround_score = row.get('Turnaround_Score', 0)
    turnaround_signals = row.get('Turnaround_Signals', '')
    turnaround_warnings = row.get('Turnaround_Warnings', '')  # NEW: One-off income warnings
    rsi_velocity = row.get('RSI_Velocity', 0)
    
    # Industry for sector-aware checks
    industry = row.get('Industry', '')
    
    # Debt data
    debt = row.get('Debt', 0)
    total_liabilities = row.get('Total Liabilities', 0)
    
    # Get sector-specific thresholds for this stock
    sector_thresholds = get_sector_thresholds(industry)
    de_thresh = sector_thresholds['debt_to_equity']
    
    # ═══════════════════════════════════════════════════════
    # TRAP PROBABILITY CALCULATION (0-100%)
    # NOW WITH SECTOR-AWARE DEBT THRESHOLDS!
    # ═══════════════════════════════════════════════════════
    
    trap_probability = 0
    red_flags = []
    
    # 🚨 CASH TRAP: Burning cash
    if fcf < 0 and ocf < 0:
        red_flags.append("CASH_TRAP")
        trap_probability += min(abs(fcf) / 500, 1) * 25
    elif fcf < 0:
        trap_probability += 5
    
    # 🚨 DEBT BOMB: SECTOR-ADJUSTED thresholds!
    # Banks can have D/E of 10+ (normal), while IT should be <0.5
    if de > de_thresh['danger']:
        debt_value = debt if debt > 0 else (total_liabilities * 0.5)
        if debt_value > 0 and ocf < debt_value * 0.1:
            red_flags.append("DEBT_BOMB")
            trap_probability += 20
        else:
            red_flags.append("EXCESS_DEBT")
            trap_probability += 12
    elif de > de_thresh['high']:
        red_flags.append("HIGH_DEBT")
        trap_probability += 5
    
    # Negative OCF with high debt (sector-adjusted)
    if ocf < 0 and de > de_thresh['normal'] and "DEBT_BOMB" not in red_flags:
        red_flags.append("DEBT_STRESS")
        trap_probability += 12
    
    # 🚨 ONE-OFF INCOME WARNING (from turnaround detection)
    if 'ONE_OFF' in turnaround_warnings:
        red_flags.append("ONE_OFF_INCOME")
        trap_probability += 15
    elif 'CHECK_INCOME' in turnaround_warnings:
        trap_probability += 5
    
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
    # EARLY ENTRY DETECTION (Catch Before Breakout)
    # ═══════════════════════════════════════════════════════
    
    # Strong early entry: High accumulation score + no traps + decent fundamentals
    if early_entry_score >= 0.6 and trap_probability < 25 and sq >= 0.4:
        signal_count = len(early_entry_signals.split(',')) if early_entry_signals else 0
        if signal_count >= 3:
            return "🎯 EARLY ENTRY", "early-entry", trap_probability
    
    # Moderate early entry with FII accumulation
    if early_entry_score >= 0.45 and 'FII_ACCUM' in early_entry_signals and trap_probability < 30:
        return "🔍 ACCUMULATION", "accumulation", trap_probability
    
    # ═══════════════════════════════════════════════════════
    # TURNAROUND DETECTION (Recovery Plays)
    # NOW WITH ONE-OFF INCOME PROTECTION!
    # ═══════════════════════════════════════════════════════
    
    # Check for one-off income warnings (falling knife protection)
    has_one_off_warning = 'ONE_OFF' in turnaround_warnings or 'CHECK_INCOME' in turnaround_warnings
    
    # Strong turnaround: High recovery score + insider buying + NO ONE-OFF WARNING
    if turnaround_score >= 0.6 and trap_probability < 40 and not has_one_off_warning:
        if 'REAL_TURNAROUND' in turnaround_signals:  # Revenue + PAT both growing
            return "🔄 TURNAROUND ✓", "turnaround", trap_probability
        elif 'INSIDER_BUY' in turnaround_signals:
            return "🔄 TURNAROUND", "turnaround", trap_probability
        elif 'PAT_ACCEL' in turnaround_signals and 'REV_GROWING' in turnaround_signals:
            return "📊 RECOVERY", "turnaround", trap_probability
    
    # Turnaround with warning - show but flag it
    if turnaround_score >= 0.5 and has_one_off_warning and trap_probability < 50:
        if 'INSIDER_BUY' in turnaround_signals:  # Insider buying overrides some concern
            return "🔄 TURNAROUND ⚠️", "turnaround", trap_probability
        else:
            return "⚠️ CHECK INCOME", "hold", trap_probability
    
    # Moderate turnaround with improving trajectory (no one-off)
    if turnaround_score >= 0.4 and 'OCF_POS' in turnaround_signals and trap_probability < 35:
        if not has_one_off_warning:
            return "🌱 IMPROVING", "turnaround", trap_probability
        else:
            return "🌱 IMPROVING ⚠️", "hold", trap_probability
    
    # ═══════════════════════════════════════════════════════
    # PURE PERCENTILE VERDICT (Simple & Clean)
    # ═══════════════════════════════════════════════════════
    
    flag_indicator = f" ⚡{red_flags[0]}" if len(red_flags) == 1 else ""
    trap_indicator = f" ({trap_probability:.0f}%)" if trap_probability >= 20 else ""
    
    # RSI Velocity bonus for strong buys (momentum building)
    velocity_bonus = rsi_velocity > 5 and mom_accel > 0
    
    # Top 5%: STRONG BUY
    if score >= 85 and trap_probability < 20 and len(red_flags) == 0:
        return "💎 STRONG BUY", "strong-buy", trap_probability
    
    # Score 80-85 with velocity bonus = Strong Buy
    if score >= 80 and velocity_bonus and trap_probability < 20 and len(red_flags) == 0:
        return "💎 STRONG BUY ↗", "strong-buy", trap_probability
    
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
        <h1 style='margin:0; font-weight:700; color:#1a1a2e;'>V4 ULTRA+</h1>
        <p style='color:#666; margin:0.3rem 0 0 0; font-size:0.95rem;'>4-Factor Pure Quant • Early Entry Detection • Turnaround Finder</p>
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
        
        # V4 ULTRA+ ENHANCED MODEL
        st.markdown("---")
        st.markdown("### 🎯 4-Factor Model")
        st.caption("Pure quant + Enhanced Detection")
        
        st.markdown("""
        | Factor | Weight | Components |
        |--------|--------|------------|
        | **Momentum** | 60% | RSI, Ret3M, 52WH, Velocity |
        | **Institutional** | 20% | FII Trend, Promoter |
        | **Quality** | 10% | ROCE |
        | **Safety** | 10% | OCF, FCF, D/E |
        """)
        
        with st.expander("🎯 Early Entry Detection", expanded=False):
            st.markdown("""
            **Catches accumulation BEFORE breakout:**
            - 🔍 FII quietly accumulating (sector-adjusted)
            - 📊 RSI building (sector-specific sweet spot)
            - 📈 Momentum acceleration positive
            - 💰 Price hasn't run yet (<15%)
            
            *Thresholds auto-adjust by sector!*
            """)
        
        with st.expander("🔄 Turnaround Detection", expanded=False):
            st.markdown("""
            **Identifies recovery plays:**
            - 📈 PAT + Revenue both growing = REAL turnaround
            - 💪 Promoter buying during dip
            - 📊 Margin expansion (sector-adjusted)
            - 💵 OCF turning positive
            
            **🚨 One-Off Income Detection:**
            - Flags PAT growth without Revenue growth
            - Catches asset sales disguised as turnarounds
            - Warns: "CHECK INCOME" if suspicious
            """)
        
        with st.expander("🏭 Sector-Aware Thresholds", expanded=False):
            st.markdown("""
            **Auto-adjusts for industry norms:**
            
            | Sector | D/E Normal | D/E Danger |
            |--------|------------|------------|
            | **Banking** | 8.0 | 15.0 |
            | **IT/Tech** | 0.2 | 1.0 |
            | **Infra** | 1.5 | 4.0 |
            | **FMCG** | 0.3 | 1.5 |
            | **Power** | 2.0 | 5.0 |
            | **Metals** | 1.2 | 3.5 |
            
            *No more false flags on banks for "high debt"!*
            """)
        
        with st.expander("🛡️ Safety Net Traps", expanded=False):
            st.markdown("""
            - 🚨 **DEATH SPIRAL**: Cash burn + Debt
            - 🚨 **PUMP & DUMP**: Momentum + No cash
            - 🚨 **INSIDER EXIT**: Promoter selling >3%
            - 🚨 **ONE_OFF_INCOME**: Fake profit (asset sale)
            - 🚨 **EXCESS_DEBT**: Beyond sector norms
            """)
        
        with st.expander("❌ Dead Weight (Removed)", expanded=False):
            st.markdown("""
            | Metric | Why Removed |
            |--------|-------------|
            | NPM | -0.013 corr ❌ |
            | DII Changes | NEGATIVE ❌ |
            | Hardcoded D/E=2 | Sector-blind ❌ |
            | Fixed RSI 40-60 | Sector-blind ❌ |
            
            *Now uses adaptive thresholds!*
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
            ### 🎯 V4 ULTRA+: Enhanced Quant
            
            **4 Core Factors + Smart Detection:**
            
            | Factor | Weight | Components |
            |--------|--------|------------|
            | **Momentum** | 60% | RSI, Ret3M, 52WH, Velocity |
            | **Institutional** | 20% | FII Trend, Promoter |
            | **Quality** | 10% | ROCE |
            | **Safety** | 10% | OCF, FCF, D/E |
            
            **🆕 Early Entry**: Catch before breakout
            
            **🆕 Turnaround**: Identify recovery plays
            """)
        
        with col2:
            st.markdown("""
            ### 🛡️ Enhanced Verdicts
            
            **🆕 Opportunity Detection:**
            - 🎯 **EARLY ENTRY**: Accumulation phase
            - 🔍 **ACCUMULATION**: FII buying quietly
            - 🔄 **TURNAROUND**: Recovery play
            - 🌱 **IMPROVING**: Getting better
            
            **Standard Verdicts:**
            - 💎 **STRONG BUY**: Top performers
            - 📈 **BUY**: Good picks
            - ⏸️ **HOLD**: Wait and watch
            
            **⚠️ Trap Detection:**
            - 🚨 DEATH SPIRAL, PUMP & DUMP
            - 🚨 INSIDER EXIT, SMART EXIT
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
        
        # Display Columns - V4 ULTRA+ Enhanced Model
        base_cols = ['Rank', 'Name', 'Verdict', 'Final_Score', 'Trap_Probability']
        price_cols = ['Close Price', 'Price To Earnings']
        metric_cols = ['ROCE', 'Debt To Equity', 'Free Cash Flow']
        score_cols = ['Score_Momentum', 'Score_Institutional', 'Score_Quality', 'Score_Safety']  # 4 core factors
        new_detection_cols = ['Early_Entry_Score', 'Turnaround_Score', 'RSI_Velocity', 'Turnaround_Warnings']  # New detectors
        advanced_cols = ['Quality_Gate', 'Momentum_Acceleration', 'Data_Confidence', 'Early_Entry_Signals', 'Turnaround_Signals', 'Industry']
        
        if show_all_columns:
            display_cols = base_cols + score_cols + new_detection_cols + advanced_cols
        else:
            display_cols = base_cols + [c for c in price_cols + metric_cols if c in df.columns]
        
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        # Color styling based on verdict
        def color_row(row):
            verdict_class = row.get('Verdict_Class', 'hold')
            colors = {
                'strong-buy': 'background-color: rgba(0, 200, 83, 0.15)',
                'buy': 'background-color: rgba(33, 150, 243, 0.12)',
                'early-entry': 'background-color: rgba(255, 193, 7, 0.18)',  # Gold for early entry
                'accumulation': 'background-color: rgba(255, 152, 0, 0.15)',  # Orange for accumulation
                'turnaround': 'background-color: rgba(156, 39, 176, 0.15)',  # Purple for turnaround
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
        
        # Quick Stats - Enhanced counts with new categories
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("💎 Strong Buy", len(df[df['Verdict_Class'] == 'strong-buy']))
        col2.metric("📈 Buy", len(df[df['Verdict_Class'] == 'buy']))
        col3.metric("🎯 Early Entry", len(df[df['Verdict_Class'].isin(['early-entry', 'accumulation'])]))
        col4.metric("🔄 Turnaround", len(df[df['Verdict_Class'] == 'turnaround']))
        col5.metric("⚠️ Risky", len(df[df['Verdict_Class'] == 'trap']))
        col6.metric("⏸️ Hold", len(df[df['Verdict_Class'] == 'hold']))
    
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
                categories = ['Momentum', 'Institutional', 'Quality', 'Safety', 'Early Entry', 'Turnaround']
                values = [
                    top_stock.get('Score_Momentum', 0.5),
                    top_stock.get('Score_Institutional', 0.5),
                    top_stock.get('Score_Quality', 0.5),
                    top_stock.get('Score_Safety', 0.5),
                    top_stock.get('Early_Entry_Score', 0),
                    top_stock.get('Turnaround_Score', 0)
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
        
        # NEW: Early Entry & Turnaround Opportunities
        col4, col5 = st.columns(2)
        
        early_df = df[df['Verdict_Class'].isin(['early-entry', 'accumulation'])]
        if len(early_df) > 0:
            csv_early = early_df.to_csv(index=False).encode('utf-8')
            col4.download_button(
                "🎯 Early Entry",
                csv_early,
                f"V4_EarlyEntry_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        turnaround_df = df[df['Verdict_Class'] == 'turnaround']
        if len(turnaround_df) > 0:
            csv_turn = turnaround_df.to_csv(index=False).encode('utf-8')
            col5.download_button(
                "🔄 Turnarounds",
                csv_turn,
                f"V4_Turnarounds_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
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
