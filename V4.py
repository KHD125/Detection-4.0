"""
Rank Trajectory Engine v2.3 — Return-Based Intelligence + Momentum Decay + Sector Alpha
========================================================================================
Professional Stock Rank Trajectory Analysis System
with Adaptive Weight Intelligence, Return-Based Alignment, Momentum Decay Warning,
Sector Alpha Detection, and Multi-Stage Selection Funnel.

CORE ARCHITECTURE:
  6-Component Adaptive Scoring → Elite Dominance Bonus → Price-Rank Multiplier
    → Momentum Decay Penalty → Sector Alpha Tag
  Weights shift dynamically by position tier (elite/strong/mid/bottom).
  Returns (ret_7d/ret_30d) confirm or flag trajectory via ×0.92 to ×1.08 multiplier.
  Momentum decay detects 11.4% trap stocks with good rank but negative returns.
  Sector alpha separates genuine leaders from sector-beta riders.

3-STAGE FUNNEL:
  Stage 1: Discovery  — Trajectory Score ≥70 or Rocket/Breakout → 50-100 candidates
  Stage 2: Validation — 5 Wave Engine rules, must pass 4/5    → 20-30 stocks
  Stage 3: Final      — TQ≥70, Leader patterns, no DOWNTREND  → 5-10 FINAL BUYS

Components: Adaptive Weights by Tier
  Elite (>90pct):  Pos 45% | Trend 12% | Vel 8% | Acc 5% | Con 18% | Res 12%
  Strong (70-90):  Pos 32% | Trend 18% | Vel 12% | Acc 8% | Con 16% | Res 14%
  Mid (40-70):     Pos 18% | Trend 22% | Vel 20% | Acc 12% | Con 14% | Res 14%
  Bottom (<40):    Pos 10% | Trend 20% | Vel 25% | Acc 18% | Con 12% | Res 15%

v2.3 Upgrades:
  1. Return-Based Alignment: Uses CSV ret_7d/ret_30d (split-adjusted by provider)
     instead of raw prices. No split detection needed — cleaner, more accurate.
  2. Momentum Decay Warning: Catches stocks with good rank but negative recent
     returns (11.4% of top-10% are traps). Penalty ×0.93 to ×1.0.
  3. Sector Alpha Check: Separates SECTOR_LEADER from SECTOR_BETA stocks.
     Stocks riding hot sectors get flagged; true alpha gets rewarded.

Version: 2.3.0
Last Updated: February 2026
"""

# ============================================
# STREAMLIT CONFIG - Must be first
# ============================================
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
st.set_page_config(
    page_title="Rank Trajectory Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import re
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
from io import BytesIO

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

MIN_WEEKS_DEFAULT = 3
MAX_DISPLAY_DEFAULT = 100

# ── Quantitative Enhancement Configuration (v3.0) ──

# Bayesian Confidence Shrinkage — shrinks scores toward population mean
# when data is insufficient. Prevents 4-week lucky stocks from ranking high.
BAYESIAN_CONFIDENCE = {
    'prior_mean': 45.0,          # Population prior (assume mediocre until proven)
    'full_confidence_weeks': 16,  # Weeks needed for 100% data confidence
    'min_confidence': 0.25,       # Minimum confidence even with 2 weeks
}

# Hurst Exponent — determines if rank series will PERSIST or REVERT
# H > 0.5 = trending (current pattern likely continues)
# H = 0.5 = random walk (no predictive power)
# H < 0.5 = mean-reverting (current pattern likely reverses)
HURST_CONFIG = {
    'min_weeks': 6,               # Need at least 6 datapoints
    'trend_persistence_h': 0.55,  # H above this = trending
    'mean_revert_h': 0.42,        # H below this = mean-reverting
    'max_boost': 1.06,            # Max multiplier for strong persistence
    'max_penalty': 0.94,          # Max penalty for mean-reversion on uptrend
}

# Information Ratio — risk-adjusted consistency (replaces raw volatility)
# IR = mean(excess_return) / std(excess_return)
# High IR = consistently outperforming. Low IR = noisy, unreliable.
INFO_RATIO_CONFIG = {
    'benchmark_growth': 0.5,      # Assumed benchmark percentile growth/week
    'excellent_ir': 0.8,          # IR above this = excellent consistency
    'good_ir': 0.3,               # IR above this = good consistency
    'poor_ir': -0.2,              # IR below this = inconsistent
}

# Trajectory Score Weights — v2.1 ADAPTIVE WEIGHT SYSTEM
# Weights dynamically shift based on WHERE the stock sits (percentile tier)
# Elite stocks: Position dominates (45%) — being at rank 5 IS the achievement
# Climbers: Movement dominates (Velocity 25%, Trend 25%) — they need to prove direction
# Bottom stocks: Acceleration matters most — are they even trying to move?

# Base weights (used for mid-range stocks, percentile 40-70)
BASE_WEIGHTS = {
    'positional': 0.25,
    'trend': 0.20,
    'velocity': 0.15,
    'acceleration': 0.10,
    'consistency': 0.15,
    'resilience': 0.15
}

# Adaptive weight profiles by percentile tier
ADAPTIVE_WEIGHTS = {
    # Elite (avg pct > 90): Position IS the score. Movement is maintenance.
    'elite': {
        'positional': 0.45, 'trend': 0.12, 'velocity': 0.08,
        'acceleration': 0.05, 'consistency': 0.18, 'resilience': 0.12
    },
    # Strong (avg pct 70-90): Balanced — good position + should keep improving
    'strong': {
        'positional': 0.32, 'trend': 0.18, 'velocity': 0.12,
        'acceleration': 0.08, 'consistency': 0.16, 'resilience': 0.14
    },
    # Mid (avg pct 40-70): Movement-focused — need to prove trajectory
    'mid': {
        'positional': 0.18, 'trend': 0.22, 'velocity': 0.20,
        'acceleration': 0.12, 'consistency': 0.14, 'resilience': 0.14
    },
    # Bottom (avg pct < 40): Acceleration-heavy — are they turning around?
    'bottom': {
        'positional': 0.10, 'trend': 0.20, 'velocity': 0.25,
        'acceleration': 0.18, 'consistency': 0.12, 'resilience': 0.15
    }
}

# Elite Dominance Bonus thresholds
ELITE_BONUS = {
    'top3_sustained': {'pct_threshold': 97, 'history_ratio': 0.60, 'floor': 88},
    'top5_sustained': {'pct_threshold': 95, 'history_ratio': 0.60, 'floor': 82},
    'top10_sustained': {'pct_threshold': 90, 'history_ratio': 0.55, 'floor': 75},
    'top20_sustained': {'pct_threshold': 80, 'history_ratio': 0.50, 'floor': 68}
}

# Return-Based Price-Rank Alignment Configuration (v3.0 — EMA-smoothed, 3-signal, recency-weighted)
PRICE_ALIGNMENT = {
    'noise_band_stable': 2.0,        # Ignore rank moves < this for stable stocks
    'noise_band_normal': 1.0,        # Ignore rank moves < this for normal stocks
    'min_weeks': 4,                  # Minimum weeks needed for alignment calculation
    'multiplier_max_boost': 1.12,    # Maximum upward multiplier (widened from 1.08)
    'multiplier_max_penalty': 0.85,  # Maximum downward multiplier (widened from 0.92)
    'confirmed_threshold': 72,       # Alignment score above this = PRICE_CONFIRMED
    'divergent_threshold': 35,       # Alignment score below this = PRICE_DIVERGENT
    'ema_span': 3,                   # EMA span for smoothing ret_7d (3-week)
    'recency_window': 4,             # Recent N weeks get 2× weight in directional signal
}

# Momentum Decay Warning Configuration (v2.3)
MOMENTUM_DECAY = {
    'min_pct_tier': 40,              # Only check stocks above this avg percentile
    'r7_severe': -5,                 # Weekly return below this triggers severe signal
    'r7_moderate': -2,               # Weekly return below this triggers moderate signal
    'r30_severe_high': -15,          # 30d return for top-ranked stocks: TRAP threshold
    'r30_moderate_high': -5,         # 30d return moderate warning for top stocks
    'r30_severe_mid': -10,           # 30d return severe for mid-ranked stocks
    'from_high_severe': -20,         # Far from high — significant correction
    'from_high_moderate': -15,       # Moderate correction from high
    'high_decay_multiplier': 0.93,   # Severe decay penalty
    'moderate_decay_multiplier': 0.96,  # Moderate decay penalty
    'mild_decay_multiplier': 0.98,   # Mild decay warning
    'severe_threshold': 60,          # Decay score above this = severe
    'moderate_threshold': 35,        # Decay score above this = moderate
    'mild_threshold': 15,            # Decay score above this = mild
}

# Sector Alpha Configuration (v2.3)
SECTOR_ALPHA = {
    'min_sector_stocks': 3,          # Minimum stocks in sector for alpha calc
    'leader_z': 1.5,                 # Z-score above this = sector leader
    'outperform_z': 0.5,             # Z-score above this = outperforming sector
    'aligned_z': -0.5,              # Z-score above this = aligned with sector
    'beta_sector_min': 60,           # Sector avg must be > this for beta detection
}

# Funnel Stage Defaults
FUNNEL_DEFAULTS = {
    'stage1_score': 70,
    'stage1_patterns': ['rocket', 'breakout', 'momentum_building'],
    'stage2_tq': 60,
    'stage2_master_score': 50,
    'stage2_min_rules': 4,
    'stage3_tq': 70,
    'stage3_require_leader': True,
    'stage3_no_downtrend_weeks': 4
}

# Grade definitions: (min_score, label, emoji)
GRADE_DEFS = [
    (85, 'S', '🏆'), (70, 'A', '🥇'), (55, 'B', '🥈'),
    (40, 'C', '🥉'), (25, 'D', '📊'), (0, 'F', '📉')
]

GRADE_COLORS = {
    'S': '#FFD700', 'A': '#00C853', 'B': '#2196F3',
    'C': '#FF9800', 'D': '#FF5722', 'F': '#F44336'
}

# Pattern definitions: key -> (emoji, name, description)
# v3.0 — 13 actionable patterns, no ghost entries, priority-ordered detection
PATTERN_DEFS = {
    'stable_elite':      ('🎯', 'Stable Elite',      'Consistently top-ranked with low volatility'),
    'rocket':            ('🚀', 'Rocket',             'Rapid strong improvement across all dimensions'),
    'breakout':          ('⚡', 'Breakout',            'Sudden significant rank jump beyond normal variance'),
    'momentum_building': ('🔥', 'Momentum Building',  'Acceleration surging — early signal before full move'),
    'at_peak':           ('🏔️', 'At Peak',            'At or near all-time best rank with sustained strength'),
    'topping_out':       ('⛰️', 'Topping Out',        'Near peak but momentum fading — potential reversal'),
    'steady_climber':    ('📈', 'Steady Climber',     'Gradual but consistent rank improvement'),
    'recovery':          ('🔄', 'Recovery',           'Bouncing back from rank deterioration'),
    'consolidating':     ('⏳', 'Consolidating',      'Tight range movement — potential breakout setup'),
    'fading':            ('📉', 'Fading',             'Rank deteriorating from recent levels'),
    'crash':             ('💥', 'Crash',              'Severe rapid rank collapse — high-risk warning'),
    'volatile':          ('🌊', 'Volatile',           'Large and unpredictable rank swings'),
    'new_entry':         ('💎', 'New Entry',           'Recently appeared or insufficient history'),
    'neutral':           ('➖', 'Neutral',             'Average stock with no strong directional signal'),
}

PATTERN_COLORS = {
    'stable_elite': '#8A2BE2', 'rocket': '#FF4500', 'breakout': '#FFD700',
    'momentum_building': '#FF6347', 'at_peak': '#FF69B4', 'topping_out': '#CD853F',
    'steady_climber': '#32CD32', 'recovery': '#00BFFF', 'consolidating': '#B8860B',
    'fading': '#808080', 'crash': '#DC143C', 'volatile': '#FF8C00',
    'new_entry': '#00CED1', 'neutral': '#A9A9A9',
}

# ============================================
# CUSTOM CSS — MINIMAL DARK DESIGN SYSTEM v3
# ============================================
st.markdown("""
<style>
    /* ── Typography ── */
    .main-header {
        font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(120deg, #FF6B35 30%, #58a6ff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0; line-height: 1.15;
    }
    .sub-header { font-size: 0.95rem; color: #8b949e; margin-top: -4px; margin-bottom: 16px; }
    .sec-head {
        font-size: 0.88rem; font-weight: 600; color: #c9d1d9; letter-spacing: 0.3px;
        margin: 20px 0 8px 0; display: flex; align-items: center; gap: 6px;
    }
    .sec-cap { font-size: 0.7rem; color: #6e7681; margin-top: -4px; margin-bottom: 10px; }

    /* ── Metric Strip ── */
    .m-strip {
        display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px;
    }
    .m-chip {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px;
        padding: 12px 0; text-align: center; flex: 1; min-width: 90px;
        transition: border-color 0.2s ease;
    }
    .m-chip:hover { border-color: #484f58; }
    .m-val { font-size: 1.45rem; font-weight: 700; color: #e6edf3; line-height: 1; }
    .m-lbl { font-size: 0.62rem; color: #8b949e; text-transform: uppercase;
             letter-spacing: 0.6px; margin-top: 3px; }
    .m-green .m-val { color: #3fb950; }
    .m-red   .m-val { color: #f85149; }
    .m-gold  .m-val { color: #d29922; }
    .m-blue  .m-val { color: #58a6ff; }
    .m-orange .m-val { color: #FF6B35; }

    /* ── Cards ── */
    .t-card {
        background: #0d1117; border: 1px solid #30363d; border-radius: 12px;
        padding: 14px; margin-bottom: 8px;
        transition: border-color 0.15s, box-shadow 0.15s;
    }
    .t-card:hover { border-color: #484f58; }
    .t-card-danger  { border-color: rgba(248,81,73,0.4); background: rgba(248,81,73,0.03); }
    .t-card-success { border-color: rgba(63,185,80,0.4); background: rgba(63,185,80,0.03);
                      box-shadow: 0 0 12px rgba(63,185,80,0.05); }
    .t-card-success:hover { border-color: #3fb950; box-shadow: 0 0 20px rgba(63,185,80,0.10); }
    .t-hd { font-weight: 700; font-size: 0.95rem; color: #e6edf3; }
    .t-sub { font-size: 0.7rem; color: #8b949e; margin-top: 1px; }
    .t-row { display: flex; justify-content: space-between; align-items: center;
             margin-top: 5px; font-size: 0.78rem; }
    .t-badge {
        display: inline-block; padding: 2px 7px; border-radius: 8px;
        font-size: 0.65rem; font-weight: 600;
    }

    /* ── Pill Tags ── */
    .pill {
        display: inline-block; padding: 2px 8px; border-radius: 10px;
        font-size: 0.65rem; font-weight: 600; margin: 1px 2px;
        border: 1px solid;
    }
    .p-grn { color: #3fb950; border-color: rgba(63,185,80,0.3); background: rgba(63,185,80,0.08); }
    .p-red { color: #f85149; border-color: rgba(248,81,73,0.3); background: rgba(248,81,73,0.08); }
    .p-gld { color: #d29922; border-color: rgba(210,153,34,0.3); background: rgba(210,153,34,0.08); }
    .p-blu { color: #58a6ff; border-color: rgba(88,166,255,0.3); background: rgba(88,166,255,0.08); }

    /* ── Progress bar ── */
    .pbar-wrap { background: #21262d; border-radius: 3px; height: 5px; margin-top: 3px; }
    .pbar { height: 5px; border-radius: 3px; }

    /* ── Grade ── */
    .grade-S { color: #FFD700; font-weight: 900; font-size: 1.5rem; }
    .grade-A { color: #3fb950; font-weight: 900; font-size: 1.5rem; }
    .grade-B { color: #58a6ff; font-weight: 900; font-size: 1.5rem; }
    .grade-C { color: #d29922; font-weight: 900; font-size: 1.5rem; }
    .grade-D { color: #FF5722; font-weight: 900; font-size: 1.5rem; }
    .grade-F { color: #f85149; font-weight: 900; font-size: 1.5rem; }

    /* ── Tags ── */
    .pattern-tag {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.78rem; background: rgba(255,107,53,0.08); color: #FF6B35;
        border: 1px solid rgba(255,107,53,0.2); margin: 2px;
    }
    .mover-up   { color: #3fb950; font-weight: 700; }
    .mover-down { color: #f85149; font-weight: 700; }

    /* ── Misc ── */
    .divider { border-top: 1px solid #21262d; margin: 16px 0; }
    .stock-card {
        background: #0d1117; border-radius: 14px; padding: 22px;
        margin-bottom: 16px; border: 1px solid #30363d;
    }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 600; border-radius: 8px 8px 0 0; }

    /* ── Funnel ── */
    .funnel-stage {
        background: #161b22; border-radius: 12px; padding: 16px;
        margin-bottom: 10px; border-left: 4px solid;
    }
    .stage-discovery  { border-left-color: #58a6ff; }
    .stage-validation { border-left-color: #d29922; }
    .stage-final      { border-left-color: #3fb950; }
    .rule-pass { color: #3fb950; font-weight: 600; }
    .rule-fail { color: #f85149; font-weight: 600; }
    .final-buy-card {
        background: #0d1117; border: 2px solid #238636; border-radius: 14px;
        padding: 18px; margin-bottom: 10px; box-shadow: 0 0 14px rgba(35,134,54,0.10);
    }
    .funnel-stat {
        background: #161b22; border-radius: 10px; padding: 14px;
        text-align: center; border: 1px solid #30363d;
    }
    .funnel-stat-value { font-size: 1.8rem; font-weight: 700; }
    .funnel-stat-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with zero/nan/inf protection"""
    try:
        if b == 0 or np.isnan(b) or np.isnan(a):
            return default
        result = a / b
        return default if np.isinf(result) else result
    except (TypeError, ValueError):
        return default


def get_grade(score: float) -> Tuple[str, str]:
    """Return (grade_letter, emoji) for a trajectory score"""
    for threshold, label, emoji in GRADE_DEFS:
        if score >= threshold:
            return label, emoji
    return 'F', '📉'


def format_rank_change(change: int) -> str:
    """Format rank change with arrow indicators"""
    if change > 0:
        return f"▲ {change}"
    elif change < 0:
        return f"▼ {abs(change)}"
    return "— 0"


def ranks_to_percentiles(ranks: List[float], totals: List[int]) -> List[float]:
    """Convert absolute ranks to percentiles (higher = better)"""
    return [(1 - r / max(t, 1)) * 100 for r, t in zip(ranks, totals)]


def calculate_tmi(ranks: List[float], totals: List[int], period: int = 14) -> float:
    """
    Trajectory Momentum Index (TMI) - RSI-style indicator for rank trajectory.
    TMI > 70 = Strong momentum, TMI < 30 = Weak/deteriorating.
    """
    pcts = ranks_to_percentiles(ranks, totals)
    if len(pcts) < 3:
        return 50.0

    changes = np.diff(pcts)
    window = min(period, len(changes))
    recent = changes[-window:]

    gains = np.where(recent > 0, recent, 0)
    losses = np.where(recent < 0, -recent, 0)

    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

    rs = safe_div(avg_gain, avg_loss, 1.0)
    tmi = 100 - safe_div(100, 1 + rs, 50)
    return float(np.clip(tmi, 0, 100))


# ============================================
# DATA LOADING ENGINE
# ============================================

def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from CSV filename: Stocks_Weekly_YYYY-MM-DD_..."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return datetime.strptime(match.group(1), '%Y-%m-%d') if match else None


def load_and_compute(uploaded_files: list) -> Tuple[Optional[pd.DataFrame], Optional[dict], Optional[list], Optional[dict]]:
    """
    Master data pipeline: Process uploaded CSVs → Build histories → Compute trajectories.
    Returns: (trajectory_df, histories, dates_iso, metadata)
    """
    # ── Step 1: Parse uploaded CSVs ──
    if not uploaded_files:
        return None, None, None, None

    weekly_data = {}
    for ufile in uploaded_files:
        date = parse_date_from_filename(ufile.name)
        if date is None:
            continue
        try:
            ufile.seek(0)
            df = pd.read_csv(ufile)
            if 'rank' in df.columns and 'ticker' in df.columns:
                df['ticker'] = df['ticker'].astype(str).str.strip()
                df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
                df['master_score'] = pd.to_numeric(df.get('master_score', 0), errors='coerce').fillna(0)
                df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)
                weekly_data[date] = df
        except Exception as e:
            logger.warning(f"Failed to load {ufile.name}: {e}")

    if not weekly_data:
        return None, None, None, None

    weekly_data = dict(sorted(weekly_data.items()))
    dates = sorted(weekly_data.keys())
    dates_iso = [d.strftime('%Y-%m-%d') for d in dates]

    # ── Step 2: Build rank histories ──
    histories = {}
    for date in dates:
        df = weekly_data[date]
        total = len(df)
        for _, row in df.iterrows():
            ticker = str(row.get('ticker', '')).strip()
            if not ticker or ticker == 'nan':
                continue

            if ticker not in histories:
                histories[ticker] = {
                    'dates': [], 'ranks': [], 'scores': [], 'prices': [],
                    'total_per_week': [],
                    'trend_qualities': [],    # TQ per week for funnel
                    'market_states': [],      # Market state per week for funnel
                    'pattern_history': [],    # Patterns per week for funnel
                    'ret_7d': [],             # v2.3: Weekly returns (split-adjusted)
                    'ret_30d': [],            # v2.3: 30-day returns
                    'ret_3m': [],             # v2.3: 3-month returns
                    'from_high_pct': [],      # v2.3: Distance from 52w high
                    'momentum_score': [],     # v2.3: Wave engine momentum score
                    'volume_score': [],       # v2.3: Wave engine volume score
                    'company_name': '', 'category': '', 'sector': '',
                    'industry': '', 'market_state': '', 'patterns': ''
                }

            h = histories[ticker]
            h['dates'].append(date.strftime('%Y-%m-%d'))
            h['ranks'].append(float(row['rank']) if pd.notna(row['rank']) else total)
            h['scores'].append(float(row['master_score']))
            h['prices'].append(float(row['price']))
            h['total_per_week'].append(total)

            # Track per-week data for funnel validation
            tq_val = row.get('trend_quality', 0)
            h['trend_qualities'].append(float(tq_val) if pd.notna(tq_val) else 0)
            ms_val = row.get('market_state', '')
            h['market_states'].append(str(ms_val).strip() if pd.notna(ms_val) else '')
            pat_val = row.get('patterns', '')
            h['pattern_history'].append(str(pat_val).strip() if pd.notna(pat_val) else '')

            # v2.3: Track return & fundamental data per week
            for col_name, hist_key in [
                ('ret_7d', 'ret_7d'), ('ret_30d', 'ret_30d'), ('ret_3m', 'ret_3m'),
                ('from_high_pct', 'from_high_pct'), ('momentum_score', 'momentum_score'),
                ('volume_score', 'volume_score')
            ]:
                col_val = row.get(col_name, None)
                if col_val is not None and pd.notna(col_val):
                    try:
                        h[hist_key].append(float(col_val))
                    except (ValueError, TypeError):
                        h[hist_key].append(float('nan'))
                else:
                    h[hist_key].append(float('nan'))

            # Always keep latest info
            for fld in ['company_name', 'category', 'sector', 'industry', 'market_state', 'patterns']:
                val = row.get(fld, '')
                if pd.notna(val) and str(val).strip():
                    h[fld] = str(val).strip()

    # ── Step 3: Compute trajectories for all tickers ──
    results = []
    for ticker, h in histories.items():
        traj = _compute_single_trajectory(h)
        traj['ticker'] = ticker
        traj['company_name'] = h['company_name']
        traj['category'] = h['category']
        traj['sector'] = h['sector']
        traj['industry'] = h['industry']
        traj['market_state'] = h['market_state']
        traj['latest_patterns'] = h['patterns']
        results.append(traj)

    # Build DataFrame and sort
    traj_df = pd.DataFrame(results)
    traj_df = traj_df.sort_values('trajectory_score', ascending=False).reset_index(drop=True)
    traj_df.insert(0, 't_rank', range(1, len(traj_df) + 1))

    # ── Step 4: Sector Alpha Post-Processing (v2.3) ──
    # Compare each stock's trajectory to its sector average to detect
    # SECTOR_LEADER vs SECTOR_BETA stocks
    sa_cfg = SECTOR_ALPHA
    sector_stats = traj_df[traj_df['weeks'] >= MIN_WEEKS_DEFAULT].groupby('sector')['trajectory_score'].agg(
        ['mean', 'count', 'std']
    )

    def _calc_sector_alpha(row):
        sector = row.get('sector', '')
        if not sector or sector not in sector_stats.index:
            return 'NEUTRAL', 0.0
        stats = sector_stats.loc[sector]
        if stats['count'] < sa_cfg['min_sector_stocks']:
            return 'NEUTRAL', 0.0

        sect_mean = float(stats['mean'])
        sect_std = max(float(stats['std']), 1.0)
        stock_score = float(row['trajectory_score'])
        alpha = stock_score - sect_mean
        z = alpha / sect_std

        if z > sa_cfg['leader_z']:
            return 'SECTOR_LEADER', round(alpha, 1)
        elif z > sa_cfg['outperform_z']:
            return 'SECTOR_OUTPERFORM', round(alpha, 1)
        elif z > sa_cfg['aligned_z']:
            return 'SECTOR_ALIGNED', round(alpha, 1)
        elif z > -1.0 and sect_mean > sa_cfg['beta_sector_min']:
            return 'SECTOR_BETA', round(alpha, 1)
        else:
            return 'SECTOR_LAGGARD', round(alpha, 1)

    alpha_results = traj_df.apply(_calc_sector_alpha, axis=1, result_type='expand')
    traj_df['sector_alpha_tag'] = alpha_results[0]
    traj_df['sector_alpha_value'] = alpha_results[1]

    # Add sector alpha icon to signal_tags
    def _add_alpha_signal(row):
        existing = str(row.get('signal_tags', ''))
        tag = row.get('sector_alpha_tag', 'NEUTRAL')
        if tag == 'SECTOR_LEADER':
            return existing + '👑'
        elif tag == 'SECTOR_BETA':
            return existing + '🏷️'
        elif tag == 'SECTOR_LAGGARD':
            return existing + '📉'
        return existing

    traj_df['signal_tags'] = traj_df.apply(_add_alpha_signal, axis=1)

    # Metadata
    metadata = {
        'total_weeks': len(dates),
        'date_range': f"{dates[0].strftime('%b %d, %Y')} → {dates[-1].strftime('%b %d, %Y')}",
        'total_tickers': len(histories),
        'first_date': dates[0].strftime('%Y-%m-%d'),
        'last_date': dates[-1].strftime('%Y-%m-%d'),
        'avg_stocks_per_week': int(np.mean([len(weekly_data[d]) for d in dates]))
    }

    return traj_df, histories, dates_iso, metadata


# ============================================
# TRAJECTORY SCORING ENGINE
# ============================================

def _compute_single_trajectory(h: dict) -> dict:
    """Compute all trajectory metrics for a single ticker's history"""
    ranks = h['ranks']
    totals = h['total_per_week']
    scores = h['scores']
    n = len(ranks)

    # Insufficient data
    if n < 2:
        pcts = ranks_to_percentiles(ranks, totals) if ranks else []
        return _empty_trajectory(ranks, totals, pcts, n)

    pcts = ranks_to_percentiles(ranks, totals)

    # ── 6-Component Scores (v2.1 Adaptive) ──
    positional = _calc_positional_quality(pcts, n)
    trend = _calc_trend(pcts, n)
    velocity = _calc_velocity_adaptive(pcts, n)  # Position-relative velocity
    acceleration = _calc_acceleration(pcts, n)
    consistency = _calc_consistency_adaptive(pcts, n)  # Position-aware consistency
    resilience = _calc_resilience(pcts, n)

    # ── Select Adaptive Weights Based on Percentile Tier ──
    avg_pct = float(np.mean(pcts))
    weights = _get_adaptive_weights(avg_pct)

    # ── Composite Score (Adaptive Weighted) ──
    trajectory_score = (
        weights['positional'] * positional +
        weights['trend'] * trend +
        weights['velocity'] * velocity +
        weights['acceleration'] * acceleration +
        weights['consistency'] * consistency +
        weights['resilience'] * resilience
    )

    # ── Elite Dominance Bonus ──
    # Sustained top-tier presence guarantees a minimum score floor
    trajectory_score = _apply_elite_bonus(trajectory_score, pcts, n)

    # ── Price-Rank Alignment Multiplier (v3.0 — EMA-smoothed, 3-signal) ──
    ret_7d = h.get('ret_7d', [])
    ret_30d = h.get('ret_30d', [])
    ret_3m = h.get('ret_3m', [])
    price_multiplier, price_label, price_alignment = _calc_price_alignment(ret_7d, ret_30d, pcts, avg_pct, ret_3m)
    pre_price_score = trajectory_score
    trajectory_score = float(np.clip(trajectory_score * price_multiplier, 0, 100))

    # ── Momentum Decay Warning (v2.3) ──
    # Catches stocks with good rank but deteriorating returns
    from_high = h.get('from_high_pct', [])
    decay_multiplier, decay_label, decay_score = _calc_momentum_decay(ret_7d, ret_30d, from_high, pcts, avg_pct)
    pre_decay_score = trajectory_score
    trajectory_score = float(np.clip(trajectory_score * decay_multiplier, 0, 100))

    # ── Grade ──
    grade, grade_emoji = get_grade(trajectory_score)

    # ── Pattern Detection (v2.0 position-aware) ──
    pattern_key = _detect_pattern(ranks, totals, pcts, positional, trend, velocity, acceleration, consistency)
    p_emoji, p_name, _ = PATTERN_DEFS[pattern_key]

    # ── Additional Metrics ──
    current_rank = int(ranks[-1])
    best_rank = int(min(ranks))
    worst_rank = int(max(ranks))
    avg_rank = round(np.mean(ranks), 1)
    rank_change = int(ranks[0] - ranks[-1])  # Positive = improved

    # Streak (consecutive weeks of rank improvement)
    streak = 0
    for i in range(len(ranks) - 1, 0, -1):
        if ranks[i] < ranks[i - 1]:
            streak += 1
        else:
            break

    # TMI
    tmi = calculate_tmi(ranks, totals)

    # Rank volatility
    rank_vol = round(np.std(ranks), 1) if n > 1 else 0

    # Score trajectory for sparkline (percentiles, higher = better)
    sparkline_data = [round(p, 1) for p in pcts]

    # Week-over-week rank changes
    if n >= 2:
        last_week_change = int(ranks[-2] - ranks[-1])  # positive = improved
    else:
        last_week_change = 0

    # Build price alignment tag (secondary pattern)
    price_tag = ''
    if price_label == 'PRICE_CONFIRMED':
        price_tag = '💰'
    elif price_label == 'PRICE_DIVERGENT':
        price_tag = '⚠️'

    # Build momentum decay tag (v2.3)
    decay_tag = ''
    if decay_label == 'DECAY_HIGH':
        decay_tag = '🔻'
    elif decay_label == 'DECAY_MODERATE':
        decay_tag = '⚡'
    elif decay_label == 'DECAY_MILD':
        decay_tag = '~'

    # Build signal tags column (combined indicator)
    signal_parts = []
    if price_tag:
        signal_parts.append(price_tag)
    if decay_tag:
        signal_parts.append(decay_tag)
    signal_tags = ''.join(signal_parts)

    return {
        'trajectory_score': round(trajectory_score, 2),
        'positional': round(positional, 2),
        'trend': round(trend, 2),
        'velocity': round(velocity, 2),
        'acceleration': round(acceleration, 2),
        'consistency': round(consistency, 2),
        'resilience': round(resilience, 2),
        'hurst': round(_estimate_hurst(pcts), 3) if n >= HURST_CONFIG['min_weeks'] else 0.5,
        'confidence': round(confidence, 3),
        'grade': grade,
        'grade_emoji': grade_emoji,
        'pattern_key': pattern_key,
        'pattern': f"{p_emoji} {p_name}",
        'price_alignment': round(price_alignment, 1),
        'price_multiplier': round(price_multiplier, 3),
        'price_label': price_label,
        'price_tag': price_tag,
        'pre_price_score': round(pre_price_score, 2),
        'decay_score': decay_score,
        'decay_multiplier': round(decay_multiplier, 3),
        'decay_label': decay_label,
        'decay_tag': decay_tag,
        'pre_decay_score': round(pre_decay_score, 2),
        'signal_tags': signal_tags,
        'current_rank': current_rank,
        'best_rank': best_rank,
        'worst_rank': worst_rank,
        'avg_rank': avg_rank,
        'rank_change': rank_change,
        'last_week_change': last_week_change,
        'streak': streak,
        'tmi': round(tmi, 1),
        'weeks': n,
        'rank_volatility': rank_vol,
        'sparkline': sparkline_data
    }


def _empty_trajectory(ranks, totals, pcts, n):
    """Return neutral trajectory for insufficient data"""
    return {
        'trajectory_score': 0, 'positional': 0, 'trend': 50, 'velocity': 50,
        'acceleration': 50, 'consistency': 50, 'resilience': 50,
        'hurst': 0.5, 'confidence': BAYESIAN_CONFIDENCE['min_confidence'],
        'grade': 'F', 'grade_emoji': '📉',
        'pattern_key': 'new_entry', 'pattern': '💎 New Entry',
        'price_alignment': 50.0, 'price_multiplier': 1.0,
        'price_label': 'NEUTRAL', 'price_tag': '',
        'pre_price_score': 0,
        'decay_score': 0, 'decay_multiplier': 1.0,
        'decay_label': '', 'decay_tag': '',
        'pre_decay_score': 0,
        'signal_tags': '',
        'sector_alpha_tag': 'NEUTRAL', 'sector_alpha_value': 0,
        'current_rank': int(ranks[-1]) if ranks else 0,
        'best_rank': int(min(ranks)) if ranks else 0,
        'worst_rank': int(max(ranks)) if ranks else 0,
        'avg_rank': round(np.mean(ranks), 1) if ranks else 0,
        'rank_change': 0, 'last_week_change': 0, 'streak': 0,
        'tmi': 50.0, 'weeks': n, 'rank_volatility': 0,
        'sparkline': [round(p, 1) for p in pcts] if pcts else []
    }


# ── Adaptive Weight Selection ──

def _get_adaptive_weights(avg_pct: float) -> dict:
    """
    Select weight profile based on stock's average percentile position.
    Uses smooth interpolation between tiers for continuous transitions.
    """
    if avg_pct >= 90:
        return ADAPTIVE_WEIGHTS['elite']
    elif avg_pct >= 70:
        # Interpolate between strong and elite
        ratio = (avg_pct - 70) / 20  # 0 at 70, 1 at 90
        strong = ADAPTIVE_WEIGHTS['strong']
        elite = ADAPTIVE_WEIGHTS['elite']
        return {k: strong[k] * (1 - ratio) + elite[k] * ratio for k in strong}
    elif avg_pct >= 40:
        # Interpolate between mid and strong
        ratio = (avg_pct - 40) / 30  # 0 at 40, 1 at 70
        mid = ADAPTIVE_WEIGHTS['mid']
        strong = ADAPTIVE_WEIGHTS['strong']
        return {k: mid[k] * (1 - ratio) + strong[k] * ratio for k in mid}
    else:
        # Interpolate between bottom and mid
        ratio = avg_pct / 40  # 0 at 0, 1 at 40
        bottom = ADAPTIVE_WEIGHTS['bottom']
        mid = ADAPTIVE_WEIGHTS['mid']
        return {k: bottom[k] * (1 - ratio) + mid[k] * ratio for k in bottom}


def _apply_elite_bonus(score: float, pcts: List[float], n: int) -> float:
    """
    Elite Dominance Bonus: If a stock has been in the top tier for a
    sustained portion of its history, it gets a guaranteed minimum score.
    
    Logic: Being rank 5 out of 2000 for 15 out of 23 weeks is NOT a B-grade
    stock. That's an S-grade achievement. The bonus ensures this.
    
    This is NOT a hack — it's domain logic: sustained excellence IS
    the best possible trajectory.
    """
    if n < 3:
        return score
    
    for tier_name, cfg in ELITE_BONUS.items():
        threshold = cfg['pct_threshold']
        required_ratio = cfg['history_ratio']
        floor = cfg['floor']
        
        # Count weeks where stock was above threshold
        weeks_above = sum(1 for p in pcts if p >= threshold)
        ratio = weeks_above / n
        
        if ratio >= required_ratio:
            # Apply floor, but also scale slightly above floor based on actual ratio
            # e.g., 90% of weeks at top3% → higher than 60% of weeks at top3%
            bonus_extra = (ratio - required_ratio) / (1.0 - required_ratio + 0.01) * 8
            effective_floor = floor + bonus_extra
            score = max(score, effective_floor)
            break  # Only apply highest qualifying tier
    
    return min(score, 100.0)


# ── Return-Based Price-Rank Alignment Engine (v3.0) ──

def _ema_smooth(values: List[float], span: int = 3) -> List[float]:
    """Exponential Moving Average smoothing for return series. NaN-safe."""
    if not values:
        return []
    alpha = 2.0 / (span + 1)
    smoothed = []
    prev = None
    for v in values:
        if v is None or np.isnan(v):
            smoothed.append(prev if prev is not None else float('nan'))
        elif prev is None:
            prev = v
            smoothed.append(v)
        else:
            prev = alpha * v + (1 - alpha) * prev
            smoothed.append(prev)
    return smoothed


def _calc_price_alignment(ret_7d: List[float], ret_30d: List[float],
                          pcts: List[float], avg_pct: float,
                          ret_3m: Optional[List[float]] = None) -> Tuple[float, str, float]:
    """
    Return-Based Price-Rank Alignment Multiplier (v3.0).

    UPGRADES from v2.3:
      • EMA-smoothed ret_7d (3-week) — eliminates single-week noise spikes
      • Recency weighting — last 4 weeks count 2× in directional agreement
      • Signal 3: ret_3m conviction — 3-month return confirms sustained trend
      • Wider multiplier range — ×0.85 to ×1.12 for stronger signal impact

    THREE SIGNALS:

    Signal 1 — EMA-Smoothed Directional Agreement (40%):
      Does EMA(ret_7d, 3) direction match rank percentile movement?
      Recent 4 weeks get 2× weight. Wider noise band for elite stocks.

    Signal 2 — Return Quality Confirmation (30%):
      Are recent ret_30d values positive for highly ranked stocks?
      Catches TRAP stocks: high rank + deeply negative 30d return.

    Signal 3 — 3-Month Conviction (30%):
      Is ret_3m positive? Strongest predictor for positional trades.
      A stock with positive 3m return has confirmed medium-term trend.

    MULTIPLIER RANGE: ×0.85 (strong divergence) to ×1.12 (strong confirmation)

    Returns: (multiplier, label, alignment_score)
    """
    cfg = PRICE_ALIGNMENT
    n = len(pcts)
    if ret_3m is None:
        ret_3m = []

    # ── Guard: Need valid return data ──
    valid_ret7 = [r for r in ret_7d if r is not None and not np.isnan(r)]
    if len(valid_ret7) < cfg['min_weeks'] or n < cfg['min_weeks']:
        return 1.0, 'NEUTRAL', 50.0

    # ── EMA-smooth ret_7d to reduce noise ──
    ema_span = cfg.get('ema_span', 3)
    smoothed_r7 = _ema_smooth(ret_7d, span=ema_span)

    # Build aligned quads: (ema_r7, r30, r3m, percentile)
    quads = []
    for i in range(n):
        r7e = smoothed_r7[i] if i < len(smoothed_r7) else float('nan')
        r30 = ret_30d[i] if i < len(ret_30d) else float('nan')
        r3m = ret_3m[i] if i < len(ret_3m) else float('nan')
        if r7e is not None and not np.isnan(r7e):
            quads.append((r7e, r30, r3m, pcts[i]))

    if len(quads) < cfg['min_weeks']:
        return 1.0, 'NEUTRAL', 50.0

    # ── Signal 1: EMA-Smoothed Directional Agreement (40%) — recency-weighted ──
    agree = 0.0
    total_weight = 0.0
    noise_band = cfg['noise_band_stable'] if avg_pct > 80 else cfg['noise_band_normal']
    recency_window = cfg.get('recency_window', 4)
    nq = len(quads)

    for i in range(1, nq):
        r7e = quads[i][0]
        r_chg = quads[i][3] - quads[i - 1][3]  # Percentile change

        # Skip noise — tiny rank moves for elite stocks
        if abs(r_chg) < noise_band and abs(r7e) < 1.0:
            continue

        # Recency weight: last `recency_window` weeks get 2×
        w = 2.0 if (nq - 1 - i) < recency_window else 1.0
        total_weight += w

        # Positive EMA(ret_7d) should align with improving percentile
        if r7e > 0 and r_chg > 0:
            agree += 1.0 * w    # Both positive — strong agreement
        elif r7e < -1.0 and r_chg < -1.0:
            agree += 0.8 * w    # Both negative — at least consistent
        elif abs(r7e) < 1.0:
            agree += 0.3 * w    # Return near zero — not really disagreeing
        else:
            agree -= 0.3 * w    # Divergent direction

    if total_weight > 0:
        dir_score = float(np.clip((agree / total_weight) * 50 + 50, 0, 100))
    else:
        # No significant rank moves — fallback to latest ret_30d
        latest_r30 = float('nan')
        for _, r30, _, _ in reversed(quads):
            if r30 is not None and not np.isnan(r30):
                latest_r30 = r30
                break

        if np.isnan(latest_r30):
            dir_score = 55.0
        elif avg_pct >= 80 and latest_r30 >= 0:
            dir_score = 72.0
        elif avg_pct >= 80 and latest_r30 >= -10:
            dir_score = 55.0
        elif avg_pct >= 80:
            dir_score = 35.0
        else:
            dir_score = 55.0

    # ── Signal 2: Return Quality Confirmation (30%) ──
    recent_window = min(6, len(quads))
    recent_r30 = [q[1] for q in quads[-recent_window:]
                  if q[1] is not None and not np.isnan(q[1])]

    if recent_r30:
        avg_r30 = float(np.mean(recent_r30))
        if avg_pct >= 70:
            if avg_r30 > 10:
                quality_score = 85.0
            elif avg_r30 > 0:
                quality_score = 65.0
            elif avg_r30 > -10:
                quality_score = 45.0
            else:
                quality_score = 20.0   # TRAP: rank high but returns very negative
        else:
            if avg_r30 > 20:
                quality_score = 80.0
            elif avg_r30 > 5:
                quality_score = 60.0
            elif avg_r30 > -5:
                quality_score = 50.0
            else:
                quality_score = 35.0
    else:
        quality_score = 50.0

    # ── Signal 3: 3-Month Conviction (30%) ──
    # ret_3m is the strongest predictor for positional trades — confirms sustained trend
    recent_r3m = [q[2] for q in quads[-recent_window:]
                  if q[2] is not None and not np.isnan(q[2])]

    if recent_r3m:
        avg_r3m = float(np.mean(recent_r3m))
        if avg_pct >= 70:
            # High-ranked: 3m positive = strong conviction, negative = divergence
            if avg_r3m > 20:
                conviction_score = 90.0
            elif avg_r3m > 10:
                conviction_score = 78.0
            elif avg_r3m > 0:
                conviction_score = 62.0
            elif avg_r3m > -10:
                conviction_score = 40.0
            else:
                conviction_score = 18.0  # SEVERE: 3m deeply negative on ranked stock
        else:
            # Lower-ranked: any positive 3m is a turnaround signal
            if avg_r3m > 30:
                conviction_score = 85.0
            elif avg_r3m > 10:
                conviction_score = 70.0
            elif avg_r3m > 0:
                conviction_score = 55.0
            elif avg_r3m > -15:
                conviction_score = 42.0
            else:
                conviction_score = 30.0
    else:
        conviction_score = 50.0  # No data — neutral, don't penalize

    # ── Signal 3b: Conviction Momentum (direction of ret_3m trend) ──
    # A stock going ret_3m: -5% → +8% (recovering) is better than +15% → +10% (fading)
    # Uses last 3 valid ret_3m values to detect building vs fading momentum
    all_r3m = [q[2] for q in quads if q[2] is not None and not np.isnan(q[2])]
    if len(all_r3m) >= 3:
        tail = all_r3m[-3:]
        r3m_delta = tail[-1] - tail[0]   # Positive = 3m return improving
        if r3m_delta > 10:
            conviction_momentum = 15.0    # Strong building
        elif r3m_delta > 3:
            conviction_momentum = 8.0     # Mildly building
        elif r3m_delta > -3:
            conviction_momentum = 0.0     # Flat
        elif r3m_delta > -10:
            conviction_momentum = -8.0    # Mildly fading
        else:
            conviction_momentum = -15.0   # Strongly fading
        # Blend into conviction_score (capped 0-100)
        conviction_score = float(np.clip(conviction_score + conviction_momentum, 0, 100))

    # ── Signal Disagreement Penalty ──
    # When signals contradict each other strongly, confidence is LOW → pull score down
    # E.g., Signal1=80, Signal2=20, Signal3=50 → signals are unreliable
    signals = [dir_score, quality_score, conviction_score]
    signal_spread = max(signals) - min(signals)
    if signal_spread > 50:
        disagreement_penalty = 6.0   # Strong contradiction
    elif signal_spread > 30:
        disagreement_penalty = 3.0   # Moderate contradiction
    else:
        disagreement_penalty = 0.0   # Signals agree — no penalty

    # ── Composite Alignment Score (3 signals + penalties) ──
    alignment = 0.40 * dir_score + 0.30 * quality_score + 0.30 * conviction_score - disagreement_penalty
    alignment = float(np.clip(alignment, 0, 100))

    # ── Convert to Multiplier (wider range: ×0.85 to ×1.12) ──
    conf_thresh = cfg['confirmed_threshold']
    div_thresh = cfg['divergent_threshold']
    max_boost = cfg['multiplier_max_boost']
    max_pen = cfg['multiplier_max_penalty']

    if alignment >= conf_thresh:
        t = (alignment - conf_thresh) / (100 - conf_thresh)
        multiplier = 1.04 + t * (max_boost - 1.04)
        label = 'PRICE_CONFIRMED'
    elif alignment >= 50:
        t = (alignment - 50) / (conf_thresh - 50)
        multiplier = 1.00 + t * 0.04
        label = 'NEUTRAL'
    elif alignment >= div_thresh:
        t = (alignment - div_thresh) / (50 - div_thresh)
        multiplier = 0.96 + t * 0.04
        label = 'NEUTRAL'
    else:
        t = alignment / div_thresh
        multiplier = max_pen + t * (0.96 - max_pen)
        label = 'PRICE_DIVERGENT'

    return float(multiplier), label, float(alignment)


# ── Momentum Decay Warning Engine (v2.3) ──

def _calc_momentum_decay(ret_7d: List[float], ret_30d: List[float],
                         from_high: List[float], pcts: List[float],
                         avg_pct: float) -> Tuple[float, str, int]:
    """
    Momentum Decay Warning — catches stocks with good rank but deteriorating returns.

    TRAP DETECTION: Deep audit found 11.4% of top-10% stocks have negative 30d returns.
    These stocks ranked well based on PAST momentum that has now faded.
    The rank hasn't dropped yet because ranking is lagging.

    Example: Stock at rank 151 (top 7%) with ret_30d = -22.49% → DECAY_HIGH

    SIGNALS:
      1. Weekly return (ret_7d) negative → short-term momentum loss
      2. 30-day return (ret_30d) negative on ranked stock → THE TRAP
      3. Far from 52-week high (from_high_pct) → correction underway
      4. Consecutive negative weekly returns → sustained decay

    Returns: (penalty_multiplier, warning_label, decay_score)
    """
    cfg = MOMENTUM_DECAY
    n = len(pcts)
    if n < 3:
        return 1.0, '', 0

    # Only check stocks above the minimum percentile tier
    if avg_pct < cfg['min_pct_tier']:
        return 1.0, '', 0

    # Get latest available values
    def _get_latest(lst):
        for val in reversed(lst):
            if val is not None and not np.isnan(val):
                return val
        return None

    latest_r7 = _get_latest(ret_7d)
    latest_r30 = _get_latest(ret_30d)
    latest_from_high = _get_latest(from_high)

    if latest_r7 is None and latest_r30 is None:
        return 1.0, '', 0

    r7 = latest_r7 if latest_r7 is not None else 0
    r30 = latest_r30 if latest_r30 is not None else 0
    fh = latest_from_high if latest_from_high is not None else 0

    decay_score = 0

    # Signal 1: Recent weekly return negative
    if r7 < cfg['r7_severe']:
        decay_score += 30
    elif r7 < cfg['r7_moderate']:
        decay_score += 15
    elif r7 < 0:
        decay_score += 5

    # Signal 2: 30-day return negative on a highly ranked stock (THE TRAP!)
    if avg_pct >= 70 and r30 < cfg['r30_severe_high']:
        decay_score += 40   # Severe — high rank but deeply negative 30d
    elif avg_pct >= 70 and r30 < cfg['r30_moderate_high']:
        decay_score += 25
    elif avg_pct >= 60 and r30 < cfg['r30_severe_mid']:
        decay_score += 20
    elif r30 < -20:
        decay_score += 15

    # Signal 3: Far from high — stock has corrected significantly
    if fh < cfg['from_high_severe'] and avg_pct >= 60:
        decay_score += 20
    elif fh < cfg['from_high_moderate'] and avg_pct >= 60:
        decay_score += 10

    # Signal 4: Consecutive negative weekly returns
    recent_r7 = [r for r in ret_7d[-4:] if r is not None and not np.isnan(r)]
    if len(recent_r7) >= 2:
        neg_count = sum(1 for r in recent_r7 if r < -1)
        if neg_count >= 3:
            decay_score += 15
        elif neg_count >= 2:
            decay_score += 8

    decay_score = min(decay_score, 100)

    # Convert to penalty multiplier
    if decay_score >= cfg['severe_threshold']:
        multiplier = cfg['high_decay_multiplier']
        label = 'DECAY_HIGH'
    elif decay_score >= cfg['moderate_threshold']:
        multiplier = cfg['moderate_decay_multiplier']
        label = 'DECAY_MODERATE'
    elif decay_score >= cfg['mild_threshold']:
        multiplier = cfg['mild_decay_multiplier']
        label = 'DECAY_MILD'
    else:
        multiplier = 1.0
        label = ''

    return multiplier, label, decay_score


# ── Hurst Exponent Engine (v3.0) ──

def _estimate_hurst(series: List[float]) -> float:
    """
    Estimate Hurst exponent using Rescaled Range (R/S) analysis.
    
    H > 0.5: Persistent (trending) — current direction likely continues
    H = 0.5: Random walk — no predictive power
    H < 0.5: Anti-persistent (mean-reverting) — current pattern likely reverses
    
    Uses simplified R/S method suitable for short series (6-25 weeks).
    """
    n = len(series)
    if n < 6:
        return 0.5  # Insufficient data → assume random walk
    
    arr = np.array(series, dtype=float)
    
    # Use multiple sub-series lengths for robust estimation
    rs_values = []
    lengths = []
    
    for seg_len in range(3, n // 2 + 1):
        n_segs = n // seg_len
        if n_segs < 1:
            break
        rs_seg = []
        for i in range(n_segs):
            seg = arr[i * seg_len:(i + 1) * seg_len]
            mean_seg = np.mean(seg)
            deviations = seg - mean_seg
            cumulative = np.cumsum(deviations)
            r = np.max(cumulative) - np.min(cumulative)  # Range
            s = np.std(seg, ddof=1) if np.std(seg, ddof=1) > 0 else 1e-10  # Std dev
            rs_seg.append(r / s)
        
        avg_rs = np.mean(rs_seg)
        if avg_rs > 0:
            rs_values.append(np.log(avg_rs))
            lengths.append(np.log(seg_len))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Linear regression: log(R/S) = H * log(n) + c
    # H is the slope
    x = np.array(lengths)
    y = np.array(rs_values)
    n_pts = len(x)
    slope = (n_pts * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
            max(n_pts * np.sum(x ** 2) - np.sum(x) ** 2, 1e-10)
    
    # Clamp to valid range [0.1, 0.9]
    return float(np.clip(slope, 0.1, 0.9))


def _calc_hurst_multiplier(pcts: List[float], trend_score: float) -> float:
    """
    Convert Hurst exponent to a trajectory score multiplier.
    
    LOGIC:
    - Trending (H > 0.55) + uptrend (trend > 55) → BOOST (×1.01 to ×1.06)
      The uptrend is likely to persist.
    - Trending (H > 0.55) + downtrend (trend < 45) → PENALTY (×0.94 to ×0.99)
      The downtrend is likely to persist.
    - Mean-reverting (H < 0.42) + uptrend → mild penalty (×0.97 to ×1.00)
      The uptrend is likely to reverse — don't trust it fully.
    - Mean-reverting (H < 0.42) + downtrend → mild boost (×1.01 to ×1.03)
      The downtrend is likely to reverse — recovery possible.
    - Random walk (0.42-0.55) → no adjustment (×1.00)
    """
    cfg = HURST_CONFIG
    n = len(pcts)
    if n < cfg['min_weeks']:
        return 1.0
    
    h = _estimate_hurst(pcts)
    
    is_uptrend = trend_score > 55
    is_downtrend = trend_score < 45
    
    if h >= cfg['trend_persistence_h']:
        # PERSISTENT series — current direction continues
        strength = (h - cfg['trend_persistence_h']) / (0.9 - cfg['trend_persistence_h'])
        strength = min(strength, 1.0)
        if is_uptrend:
            return 1.01 + strength * (cfg['max_boost'] - 1.01)
        elif is_downtrend:
            return 0.99 - strength * (0.99 - cfg['max_penalty'])
        else:
            return 1.0
    
    elif h <= cfg['mean_revert_h']:
        # MEAN-REVERTING — current direction likely reverses
        strength = (cfg['mean_revert_h'] - h) / (cfg['mean_revert_h'] - 0.1)
        strength = min(strength, 1.0)
        if is_uptrend:
            return 1.0 - strength * 0.03   # Mild penalty: uptrend may reverse
        elif is_downtrend:
            return 1.01 + strength * 0.02  # Mild boost: downtrend may reverse
        else:
            return 1.0
    
    else:
        # RANDOM WALK zone — no adjustment
        return 1.0


# ── Component Score Calculators (v2.1 Adaptive Engine) ──

def _calc_positional_quality(pcts: List[float], n: int) -> float:
    """
    Score based on WHERE the stock currently ranks.
    
    v2.1 Enhancement: Non-linear scaling — top positions get exponential boost.
    Rank 1/2000 (99.95%) and Rank 10/2000 (99.5%) are BOTH extraordinary.
    But rank 500/2000 (75%) is merely good. Linear scaling doesn't capture this.
    
    Uses sigmoid-boosted percentile:
    - Below 50th percentile: compressed (0-30 score range)
    - 50-80th percentile: linear-ish (30-65 score range)
    - 80-95th percentile: expanding (65-85 score range)
    - Above 95th percentile: premium zone (85-100 score range)
    
    Time-weighted: 55% current, 25% recent 4-week, 20% overall avg.
    """
    if n < 1:
        return 0.0
    
    current_pct = pcts[-1]
    recent_window = min(4, n)
    recent_avg = np.mean(pcts[-recent_window:])
    overall_avg = np.mean(pcts)
    
    # Time-weighted blend
    raw_pct = 0.55 * current_pct + 0.25 * recent_avg + 0.20 * overall_avg
    
    # Non-linear sigmoid boost for top positions
    # This makes rank 5 vs rank 50 vs rank 200 properly differentiated
    if raw_pct >= 95:
        score = 85 + (raw_pct - 95) * 3.0   # 95→85, 100→100
    elif raw_pct >= 80:
        score = 65 + (raw_pct - 80) * 1.33  # 80→65, 95→85
    elif raw_pct >= 50:
        score = 30 + (raw_pct - 50) * 1.17  # 50→30, 80→65
    else:
        score = raw_pct * 0.6               # 0→0, 50→30
    
    return float(np.clip(score, 0, 100))


def _calc_trend(pcts: List[float], n: int) -> float:
    """
    Weighted linear regression of percentile trajectory (recency-biased).
    
    v2.0 Elite Floor: Stocks in top percentiles get a minimum trend score
    because "no movement at the top" is NOT failure — it's excellence.
    
    Elite floors:
      Top 5% (pct > 95)  → floor 70
      Top 10% (pct > 90) → floor 65
      Top 20% (pct > 80) → floor 58
      Top 30% (pct > 70) → floor 52
    """
    if n < 3:
        return 50.0

    x = np.arange(n, dtype=float)
    y = np.array(pcts, dtype=float)
    # Exponential weights favoring recent data
    weights = np.exp(0.12 * x)
    weights /= weights.sum()

    # Weighted least squares
    wx = (weights * x).sum()
    wy = (weights * y).sum()
    wxx = (weights * x * x).sum()
    wxy = (weights * x * y).sum()
    w_sum = weights.sum()

    denom = w_sum * wxx - wx * wx
    if abs(denom) < 1e-10:
        return 50.0

    slope = (w_sum * wxy - wx * wy) / denom

    # Normalize: +2 percentile/week = 100, -2 = 0
    raw_score = float(np.clip(slope / 2.0 * 50 + 50, 0, 100))
    
    # v2.0 Elite Floor: top-positioned stocks get a minimum trend score
    recent_avg_pct = np.mean(pcts[-min(4, n):])
    if recent_avg_pct > 95:
        raw_score = max(raw_score, 70)
    elif recent_avg_pct > 90:
        raw_score = max(raw_score, 65)
    elif recent_avg_pct > 80:
        raw_score = max(raw_score, 58)
    elif recent_avg_pct > 70:
        raw_score = max(raw_score, 52)
    
    return raw_score


def _calc_velocity_adaptive(pcts: List[float], n: int, window: int = 4) -> float:
    """
    Position-Relative Velocity.
    
    KEY INSIGHT: Moving from rank 5 → 3 is ASTRONOMICALLY harder than
    moving from rank 500 → 300. The velocity score must reflect this.
    
    A stock at 98th percentile that stays at 98th → velocity = 65 (good!)
    A stock at 50th percentile that stays at 50th → velocity = 50 (neutral)
    A stock at 98th that drops to 95th → velocity = 55 (small dip, not disaster)
    A stock at 50th that drops to 45th → velocity = 35 (real decline)
    
    Formula: raw_velocity + position_bonus
    Position bonus = scaled by how HARD it is to move at current level
    """
    if n < 2:
        return 50.0

    w = min(window, n - 1)
    change = pcts[-1] - pcts[-w - 1]
    current_pct = pcts[-1]

    # Raw velocity (same as before)
    raw_velocity = float(np.clip(change / 10.0 * 50 + 50, 0, 100))
    
    # Position-relative adjustment:
    # At high percentiles, holding steady is an ACHIEVEMENT
    # Difficulty multiplier: higher position = harder to improve = bonus for holding
    if current_pct >= 95:
        # Top 5%: holding = good (bonus 15), small dip forgiven
        hold_bonus = 15.0
        change_sensitivity = 0.6  # Less sensitive to small changes
    elif current_pct >= 85:
        hold_bonus = 10.0
        change_sensitivity = 0.75
    elif current_pct >= 70:
        hold_bonus = 5.0
        change_sensitivity = 0.9
    else:
        hold_bonus = 0.0
        change_sensitivity = 1.0
    
    # Apply: dampen negative changes for top stocks, amplify positive for bottom
    if change < 0:
        adjusted = 50 + (change / 10.0 * 50) * change_sensitivity
    else:
        adjusted = 50 + (change / 10.0 * 50)
    
    # Add hold bonus (being at the top and not falling is GOOD)
    adjusted += hold_bonus
    
    return float(np.clip(adjusted, 0, 100))


def _calc_acceleration(pcts: List[float], n: int, window: int = 3) -> float:
    """Is the rate of improvement increasing?"""
    if n < 2 * window + 1:
        return 50.0

    # Recent velocity
    recent_vel = (pcts[-1] - pcts[-window - 1]) / window
    # Prior velocity
    prior_vel = (pcts[-window - 1] - pcts[-2 * window - 1]) / window

    accel = recent_vel - prior_vel

    # Normalize: +2 pct/week² = 100, -2 = 0
    return float(np.clip(accel / 2.0 * 50 + 50, 0, 100))


def _calc_consistency_adaptive(pcts: List[float], n: int) -> float:
    """
    Position-Aware Consistency + Information Ratio (v3.0).
    
    UPGRADE: Blends position-aware band consistency with Information Ratio,
    the quant-finance gold standard for risk-adjusted consistency.
    IR = mean(excess_return) / std(excess_return)
    High IR = reliably outperforming. Low IR = noisy, unreliable.
    """
    if n < 3:
        return 50.0

    changes = np.diff(pcts)
    std = np.std(changes)
    positive_ratio = np.sum(changes > 0) / len(changes)
    avg_pct = np.mean(pcts)
    current_pct = pcts[-1]

    # === INFORMATION RATIO COMPONENT ===
    ir_cfg = INFO_RATIO_CONFIG
    benchmark = ir_cfg['benchmark_growth']  # Expected pct growth/week
    excess = [c - benchmark for c in changes]  # Excess over benchmark
    excess_mean = float(np.mean(excess))
    excess_std = float(np.std(excess)) if len(excess) > 1 else 1.0
    ir = excess_mean / max(excess_std, 0.01)  # Information Ratio

    # Convert IR to 0-100 score
    if ir >= ir_cfg['excellent_ir']:
        ir_score = 85 + min((ir - ir_cfg['excellent_ir']) * 20, 15)  # 85-100
    elif ir >= ir_cfg['good_ir']:
        t = (ir - ir_cfg['good_ir']) / (ir_cfg['excellent_ir'] - ir_cfg['good_ir'])
        ir_score = 55 + t * 30  # 55-85
    elif ir >= ir_cfg['poor_ir']:
        t = (ir - ir_cfg['poor_ir']) / (ir_cfg['good_ir'] - ir_cfg['poor_ir'])
        ir_score = 30 + t * 25  # 30-55
    else:
        ir_score = max(10, 30 + (ir - ir_cfg['poor_ir']) * 20)  # 10-30
    ir_score = float(np.clip(ir_score, 0, 100))
    
    # === POSITION-RELATIVE CONSISTENCY ===
    if avg_pct >= 85:
        pct_range = max(pcts) - min(pcts)
        band_score = float(np.clip(100 - pct_range * 1.67, 0, 100))
        time_at_top = sum(1 for p in pcts if p >= 80) / len(pcts) * 100
        vol_bonus = float(np.clip(100 - std * 5, 0, 100))
        # Elite: 30% band, 25% time-at-top, 20% low-vol, 25% IR
        return 0.30 * band_score + 0.25 * time_at_top + 0.20 * vol_bonus + 0.25 * ir_score
    
    elif avg_pct >= 60:
        stability = float(np.clip(100 - std * 2, 0, 100))
        direction = positive_ratio * 100
        trajectory_lift = min((pcts[-1] - pcts[0]) / 20.0 * 10, 15)
        # Strong: 35% stability, 30% direction, 35% IR
        base = 0.35 * stability + 0.30 * direction + 0.35 * ir_score
        return float(np.clip(base + trajectory_lift, 0, 100))
    
    else:
        stability = float(np.clip(100 - std * 2, 0, 100))
        direction = positive_ratio * 100
        # Lower: 35% stability, 30% direction, 35% IR
        return 0.35 * stability + 0.30 * direction + 0.35 * ir_score


def _calc_resilience(pcts: List[float], n: int) -> float:
    """Recovery ability from percentile drawdowns"""
    if n < 4:
        return 50.0

    arr = np.array(pcts)
    peak = np.maximum.accumulate(arr)
    drawdowns = peak - arr
    max_dd = np.max(drawdowns)
    current_dd = drawdowns[-1]

    if max_dd < 1.0:
        return 100.0  # No meaningful drawdown

    recovery_ratio = 1.0 - safe_div(current_dd, max_dd, 1.0)
    dd_penalty = min(max_dd / 50, 0.5)  # Max 50% penalty for huge drawdowns

    return float(np.clip(recovery_ratio * 100 * (1 - dd_penalty), 0, 100))


# ── Pattern Detection (v3.0 — 13 patterns, 0 ghosts) ──

def _detect_pattern(ranks, totals, pcts, positional, trend, velocity, acceleration, consistency) -> str:
    """Classify trajectory into one of 13 patterns using a priority cascade.
    
    v3.0 improvements over v2.0:
    - Added: crash, topping_out, consolidating, momentum_building
    - Removed 3 ghost patterns (price_confirmed, price_divergent, decay_warning)
      — those are signal tags, not trajectory patterns
    - At Peak now requires positive acceleration to avoid masking topping stocks
    - Breakout works with n>=3 and uses adaptive threshold
    - 'stagnant' replaced with 'neutral' for average stocks
    - Crash catches severe collapses that 'fading' was too gentle for
    """
    n = len(ranks)
    if n < MIN_WEEKS_DEFAULT:
        return 'new_entry'

    current_pct = pcts[-1]
    avg_pct = float(np.mean(pcts))
    current_rank = ranks[-1]
    best_rank = min(ranks)
    pct_diffs = np.diff(pcts)

    # ── TIER 1: CRITICAL SIGNALS (check first — safety & dominance) ──

    # 💥 Crash — Severe rapid collapse (must be checked EARLY to warn users)
    if n >= 3:
        recent_drop = pcts[-1] - pcts[-3] if n >= 4 else pcts[-1] - pcts[-2]
        if recent_drop < -25 and velocity < 30:
            return 'crash'
        # Also catch sustained multi-week collapse
        if n >= 4 and all(d < -3 for d in pct_diffs[-3:]) and current_pct < 30:
            return 'crash'

    # 🎯 Stable Elite — Consistently top-ranked (MUST come before Rocket)
    if positional > 88 and consistency > 60 and current_pct > 85:
        return 'stable_elite'

    # ── TIER 2: STRONG POSITIVE PATTERNS ──

    # 🚀 Rocket — Strong improvement across all dimensions
    if trend > 78 and velocity > 72 and acceleration > 55:
        return 'rocket'

    # ⚡ Breakout — Sudden jump beyond normal variance (works with n>=3)
    # v3.0: requires the LATEST change to be sharp, not just cumulative
    if n >= 3:
        lookback = min(3, n - 1)
        recent_change = pcts[-1] - pcts[-(lookback + 1)]
        avg_abs_change = float(np.mean(np.abs(pct_diffs)))
        latest_change = pcts[-1] - pcts[-2]
        prev_avg_change = float(np.mean(np.abs(pct_diffs[:-1]))) if len(pct_diffs) > 1 else avg_abs_change
        # Adaptive threshold: 2.5× for stable stocks, 2.0× for volatile
        breakout_mult = 2.5 if consistency > 50 else 2.0
        if (avg_abs_change > 0.5 and recent_change > 0 and
                recent_change > breakout_mult * avg_abs_change and
                latest_change > 1.5 * prev_avg_change):  # Latest must be SHARP, not gradual
            return 'breakout'

    # 🔥 Momentum Building — Acceleration surging but trend/velocity haven't caught up yet
    if acceleration > 68 and velocity > 50 and trend < 70 and current_pct > avg_pct:
        return 'momentum_building'

    # ── TIER 3: POSITIONAL PATTERNS (where are they now?) ──

    # ⛰️ Topping Out — Near peak but momentum fading (MUST be checked BEFORE At Peak)
    if best_rank > 0 and current_rank <= best_rank * 1.15 and current_pct > 75:
        if acceleration < 40 or velocity < 38:
            return 'topping_out'

    # 🏔️ At Peak — Near best rank with sustained strength
    if best_rank > 0 and current_rank <= best_rank * 1.12 and current_pct > 78:
        if acceleration >= 40 and velocity >= 38:  # Confirm momentum is healthy
            return 'at_peak'

    # ── TIER 4: DIRECTIONAL PATTERNS ──

    # 📈 Steady Climber — Gradual consistent improvement
    if trend > 58 and consistency > 58 and velocity > 48:
        return 'steady_climber'

    # 🔄 Recovery — Bouncing back from deterioration
    if velocity > 62 and current_pct > avg_pct and trend < 55:
        return 'recovery'

    # 📉 Fading — Deteriorating (but not crashing)
    if velocity < 35 and trend < 40:
        return 'fading'

    # ── TIER 5: STRUCTURAL PATTERNS ──

    # ⏳ Consolidating — Tight range, low movement (potential breakout setup)
    if consistency > 65 and abs(trend - 50) < 12 and abs(velocity - 50) < 12:
        return 'consolidating'

    # 🌊 Volatile — Wild swings
    if consistency < 32:
        return 'volatile'

    # ➖ Neutral — Average stock, no strong signal
    return 'neutral'


# ============================================
# TOP MOVERS CALCULATION
# ============================================

def get_top_movers(histories: dict, n: int = 10, weeks: int = 1,
                   tickers: set = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get biggest rank gainers and decliners over *weeks* weeks.

    Args:
        histories: ticker → history dict (must have 'ranks' list).
        n:         number of top movers to return per side.
        weeks:     look-back window in weeks (1 = last week vs now,
                   2 = 2 weeks ago vs now, etc.).
        tickers:   optional set of tickers to restrict to (sidebar filters).
    """
    movers = []
    for ticker, h in histories.items():
        if tickers is not None and ticker not in tickers:
            continue
        rk = h['ranks']
        if len(rk) < weeks + 1:
            continue
        prev = int(rk[-(weeks + 1)])
        curr = int(rk[-1])
        change = prev - curr            # Positive = improved
        movers.append({
            'ticker': ticker,
            'company_name': h['company_name'],
            'category': h['category'],
            'prev_rank': prev,
            'current_rank': curr,
            'rank_change': change,
        })

    mover_df = pd.DataFrame(movers)
    if mover_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    gainers   = mover_df.nlargest(n, 'rank_change')
    decliners = mover_df.nsmallest(n, 'rank_change')
    return gainers, decliners


# ============================================
# 3-STAGE SELECTION FUNNEL ENGINE
# ============================================

def run_funnel(traj_df: pd.DataFrame, histories: dict, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the 3-Stage Selection Funnel.
    
    Stage 1: Discovery  — Trajectory Score ≥ threshold OR pattern match
    Stage 2: Validation — 5 Wave Engine rules, must pass min_rules/5
    Stage 3: Final      — TQ≥70, Leader patterns, no DOWNTREND
    
    Returns: (stage1_df, stage2_df, stage3_df) — each with pass/fail annotations
    """
    # ── STAGE 1: DISCOVERY ──
    s1_score = config.get('stage1_score', 70)
    s1_patterns = config.get('stage1_patterns', ['rocket', 'breakout'])
    
    s1_mask = (traj_df['trajectory_score'] >= s1_score)
    if s1_patterns:
        s1_mask = s1_mask | (traj_df['pattern_key'].isin(s1_patterns))
    stage1 = traj_df[s1_mask].copy()
    stage1 = stage1.sort_values('trajectory_score', ascending=False).reset_index(drop=True)
    
    if stage1.empty:
        return stage1, pd.DataFrame(), pd.DataFrame()
    
    # ── STAGE 2: VALIDATION (5 Wave Engine Rules) ──
    s2_tq_min = config.get('stage2_tq', 60)
    s2_ms_min = config.get('stage2_master_score', 50)
    s2_min_rules = config.get('stage2_min_rules', 4)
    
    stage2_rows = []
    for _, row in stage1.iterrows():
        h = histories.get(row['ticker'], {})
        if not h:
            continue
        
        rules_passed = 0
        rules_detail = []
        
        # Rule 1: Trend Quality ≥ threshold (from latest CSV)
        latest_tq = h['trend_qualities'][-1] if h.get('trend_qualities') else 0
        if latest_tq >= s2_tq_min:
            rules_passed += 1
            rules_detail.append(f'✅ TQ={latest_tq:.0f}')
        else:
            rules_detail.append(f'❌ TQ={latest_tq:.0f}')
        
        # Rule 2: Market State NOT DOWNTREND/STRONG_DOWNTREND
        latest_ms = h['market_states'][-1] if h.get('market_states') else ''
        if latest_ms not in ['DOWNTREND', 'STRONG_DOWNTREND']:
            rules_passed += 1
            rules_detail.append(f'✅ {latest_ms or "N/A"}')
        else:
            rules_detail.append(f'❌ {latest_ms}')
        
        # Rule 3: Master Score ≥ threshold
        latest_score = h['scores'][-1] if h.get('scores') else 0
        if latest_score >= s2_ms_min:
            rules_passed += 1
            rules_detail.append(f'✅ MS={latest_score:.0f}')
        else:
            rules_detail.append(f'❌ MS={latest_score:.0f}')
        
        # Rule 4: Recent rank not crashing (last week Δ ≥ -20)
        if len(h['ranks']) >= 2:
            recent_delta = h['ranks'][-2] - h['ranks'][-1]  # positive = improved
            if recent_delta >= -20:
                rules_passed += 1
                rules_detail.append(f'✅ Δ={recent_delta:+.0f}')
            else:
                rules_detail.append(f'❌ Δ={recent_delta:+.0f}')
        else:
            rules_detail.append('⚠️ No Δ data')
        
        # Rule 5: Volume confirmation in patterns
        latest_pats = h['pattern_history'][-1] if h.get('pattern_history') else ''
        vol_keywords = ['VOL EXPLOSION', 'LIQUID LEADER', 'INSTITUTIONAL']
        has_vol = any(kw in latest_pats for kw in vol_keywords)
        if has_vol:
            rules_passed += 1
            rules_detail.append('✅ Vol✓')
        else:
            rules_detail.append('❌ No Vol')
        
        row_dict = row.to_dict()
        row_dict['rules_passed'] = rules_passed
        row_dict['rules_detail'] = ' | '.join(rules_detail)
        row_dict['s2_pass'] = rules_passed >= s2_min_rules
        row_dict['latest_tq'] = latest_tq
        row_dict['latest_ms'] = latest_ms
        row_dict['latest_pats'] = latest_pats
        stage2_rows.append(row_dict)
    
    stage2 = pd.DataFrame(stage2_rows)
    if stage2.empty:
        return stage1, stage2, pd.DataFrame()
    
    stage2 = stage2.sort_values(['s2_pass', 'trajectory_score'], ascending=[False, False]).reset_index(drop=True)
    stage2_passed = stage2[stage2['s2_pass']].copy()
    
    if stage2_passed.empty:
        return stage1, stage2, pd.DataFrame()
    
    # ── STAGE 3: FINAL FILTER ──
    s3_tq_min = config.get('stage3_tq', 70)
    s3_require_leader = config.get('stage3_require_leader', True)
    s3_dt_weeks = config.get('stage3_no_downtrend_weeks', 4)
    
    stage3_rows = []
    for _, row in stage2_passed.iterrows():
        h = histories.get(row['ticker'], {})
        if not h:
            continue
        
        latest_tq = row.get('latest_tq', 0)
        latest_pats = row.get('latest_pats', '')
        
        # Check 1: TQ ≥ strict threshold
        tq_pass = latest_tq >= s3_tq_min
        
        # Check 2: Has CAT LEADER or MARKET LEADER
        has_leader = 'CAT LEADER' in latest_pats or 'MARKET LEADER' in latest_pats
        leader_pass = has_leader if s3_require_leader else True
        
        # Check 3: No DOWNTREND in last N weeks of market_state history
        recent_states = h['market_states'][-s3_dt_weeks:] if h.get('market_states') else []
        no_downtrend = not any('DOWNTREND' in s for s in recent_states)
        
        # All 3 must pass
        final_pass = tq_pass and leader_pass and no_downtrend
        
        row_dict = dict(row)
        row_dict['tq_pass'] = tq_pass
        row_dict['leader_pass'] = leader_pass
        row_dict['no_downtrend'] = no_downtrend
        row_dict['final_pass'] = final_pass
        
        # Build Stage 3 detail
        s3_details = []
        s3_details.append(f"{'✅' if tq_pass else '❌'} TQ≥{s3_tq_min} ({latest_tq:.0f})")
        s3_details.append(f"{'✅' if leader_pass else '❌'} Leader Pattern")
        s3_details.append(f"{'✅' if no_downtrend else '❌'} No Downtrend ({s3_dt_weeks}w)")
        row_dict['s3_detail'] = ' | '.join(s3_details)
        
        stage3_rows.append(row_dict)
    
    stage3 = pd.DataFrame(stage3_rows)
    if stage3.empty:
        return stage1, stage2, stage3
    
    stage3 = stage3.sort_values(['final_pass', 'trajectory_score'], ascending=[False, False]).reset_index(drop=True)
    
    return stage1, stage2, stage3


# ============================================
# UI: SIDEBAR
# ============================================

def render_sidebar(metadata: dict, traj_df: pd.DataFrame):
    """Render sidebar with data info and global filters"""
    with st.sidebar:
        st.markdown("---")

        # Data status
        st.markdown("#### 📂 Data Status")
        st.markdown(f"**Weeks Loaded:** {metadata['total_weeks']}")
        st.markdown(f"**Date Range:** {metadata['date_range']}")
        st.markdown(f"**Total Tickers:** {metadata['total_tickers']:,}")
        st.markdown(f"**Avg Stocks/Week:** {metadata['avg_stocks_per_week']:,}")
        st.markdown("---")

        # Filters
        st.markdown("#### ⚙️ Filters")

        # Category filter
        categories = ['All'] + sorted(traj_df['category'].dropna().unique().tolist())
        selected_cats = st.multiselect("Category", categories, default=['All'], key='sb_cat')

        # Sector filter (top sectors by count)
        sector_counts = traj_df['sector'].value_counts()
        top_sectors = sector_counts[sector_counts >= 3].index.tolist()
        sectors = ['All'] + sorted(top_sectors)
        selected_sectors = st.multiselect("Sector", sectors, default=['All'], key='sb_sector')

        # Price Alignment filter
        pa_options = ['All', '💰 Confirmed', '⚠️ Divergent', '➖ Neutral']
        selected_pa = st.selectbox("Price Alignment", pa_options, index=0, key='sb_pa')

        # Momentum Decay filter
        md_options = ['All', '✅ No Decay', '🔻 High Decay', '⚡ Moderate Decay', '~ Mild Decay']
        selected_md = st.selectbox("Momentum Decay", md_options, index=0, key='sb_md')

        # Min weeks
        min_weeks = st.slider("Min Weeks of Data", 2, metadata['total_weeks'], MIN_WEEKS_DEFAULT, key='sb_weeks')

        # Min T-Score
        min_score = st.slider("Min Trajectory Score", 0, 100, 0, key='sb_score')

        st.markdown("---")
        st.markdown("#### 📋 Quick Filters")
        quick_filter = st.radio("Preset", ['None', '🚀 Rockets Only', '🎯 Elite Only',
                                           '📈 Climbers', '⚡ Breakouts', '🏔️ At Peak',
                                           '🔥 Momentum', '💥 Crashes', '⛰️ Topping',
                                           '⏳ Consolidating', 'TMI > 70', 'Positional > 80'],
                                index=0, key='sb_quick')

        st.markdown("---")
        st.caption("v2.3.0 | Return-Based + Decay + Sector Alpha")

    return {
        'categories': selected_cats,
        'sectors': selected_sectors,
        'price_alignment': selected_pa,
        'momentum_decay': selected_md,
        'min_weeks': min_weeks,
        'min_score': min_score,
        'quick_filter': quick_filter
    }


def apply_filters(traj_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filters to trajectory DataFrame"""
    df = traj_df.copy()

    # Category
    if 'All' not in filters['categories']:
        df = df[df['category'].isin(filters['categories'])]

    # Sector
    if 'All' not in filters['sectors']:
        df = df[df['sector'].isin(filters['sectors'])]

    # Price Alignment
    pa = filters.get('price_alignment', 'All')
    if pa == '💰 Confirmed':
        df = df[df['price_label'] == 'PRICE_CONFIRMED']
    elif pa == '⚠️ Divergent':
        df = df[df['price_label'] == 'PRICE_DIVERGENT']
    elif pa == '➖ Neutral':
        df = df[df['price_label'] == 'NEUTRAL']

    # Momentum Decay
    md = filters.get('momentum_decay', 'All')
    if md == '🔻 High Decay':
        df = df[df['decay_label'] == 'DECAY_HIGH']
    elif md == '⚡ Moderate Decay':
        df = df[df['decay_label'] == 'DECAY_MODERATE']
    elif md == '~ Mild Decay':
        df = df[df['decay_label'] == 'DECAY_MILD']
    elif md == '✅ No Decay':
        df = df[~df['decay_label'].isin(['DECAY_HIGH', 'DECAY_MODERATE', 'DECAY_MILD'])]

    # Min weeks
    df = df[df['weeks'] >= filters['min_weeks']]

    # Min score
    df = df[df['trajectory_score'] >= filters['min_score']]

    # Quick filters
    qf = filters['quick_filter']
    if qf == '🚀 Rockets Only':
        df = df[df['pattern_key'] == 'rocket']
    elif qf == '🎯 Elite Only':
        df = df[df['pattern_key'] == 'stable_elite']
    elif qf == '📈 Climbers':
        df = df[df['pattern_key'] == 'steady_climber']
    elif qf == '⚡ Breakouts':
        df = df[df['pattern_key'] == 'breakout']
    elif qf == '🏔️ At Peak':
        df = df[df['pattern_key'] == 'at_peak']
    elif qf == '🔥 Momentum':
        df = df[df['pattern_key'] == 'momentum_building']
    elif qf == '💥 Crashes':
        df = df[df['pattern_key'] == 'crash']
    elif qf == '⛰️ Topping':
        df = df[df['pattern_key'] == 'topping_out']
    elif qf == '⏳ Consolidating':
        df = df[df['pattern_key'] == 'consolidating']
    elif qf == 'TMI > 70':
        df = df[df['tmi'] > 70]
    elif qf == 'Positional > 80':
        df = df[df['positional'] > 80]

    # Re-rank after filtering (no display_n limit — applied per-tab where needed)
    df = df.reset_index(drop=True)
    df['t_rank'] = range(1, len(df) + 1)

    return df


# ============================================
# UI: RANKINGS TAB — ALL TIME BEST (v3.0)
# ============================================

def render_rankings_tab(filtered_df: pd.DataFrame, all_df: pd.DataFrame,
                        histories: dict, metadata: dict):
    """Render the main rankings tab — Clean, minimal, maximum signal density."""

    # ── Ensure all v2.3 columns exist (defensive) ──
    for col, default in [('price_tag', ''), ('signal_tags', ''), ('decay_tag', ''),
                         ('decay_label', ''), ('decay_score', 0), ('decay_multiplier', 1.0),
                         ('sector_alpha_tag', 'NEUTRAL'), ('sector_alpha_value', 0),
                         ('price_label', 'NEUTRAL'), ('price_alignment', 50),
                         ('price_multiplier', 1.0), ('pre_price_score', 0),
                         ('pre_decay_score', 0), ('grade_emoji', '📉'),
                         ('pattern_key', 'neutral'), ('pattern', '➖ Neutral'),
                         ('sector', ''),
                         ('company_name', ''), ('category', ''), ('industry', '')]:
        if col not in all_df.columns:
            all_df[col] = default
        if col not in filtered_df.columns:
            filtered_df[col] = default

    # ── Compute metrics once ──
    total = len(all_df)
    shown = len(filtered_df)
    avg_score = filtered_df['trajectory_score'].mean() if shown > 0 else 0
    rockets = int((filtered_df['pattern_key'] == 'rocket').sum())
    elites = int((filtered_df['pattern_key'] == 'stable_elite').sum())
    confirmed = int((filtered_df['price_label'] == 'PRICE_CONFIRMED').sum())
    divergent = int((filtered_df['price_label'] == 'PRICE_DIVERGENT').sum())
    decay_high = int((filtered_df['decay_label'] == 'DECAY_HIGH').sum())
    decay_mod = int((filtered_df['decay_label'] == 'DECAY_MODERATE').sum())
    decay_mild = int((filtered_df['decay_label'] == 'DECAY_MILD').sum())
    decay_any = decay_high + decay_mod + decay_mild
    clean_pct = round((1 - decay_any / max(shown, 1)) * 100, 1)
    sect_leaders = int((filtered_df['sector_alpha_tag'] == 'SECTOR_LEADER').sum())
    grade_s = int((filtered_df['grade'] == 'S').sum())
    grade_a = int((filtered_df['grade'] == 'A').sum())
    clean_n = shown - decay_any

    # ════════════════════════════════════════════
    # SECTION 1 — METRIC STRIP (compact, 8 chips)
    # ════════════════════════════════════════════
    def _chip(val, lbl, cls=''):
        return f'<div class="m-chip {cls}"><div class="m-val">{val}</div><div class="m-lbl">{lbl}</div></div>'

    sc_cls = 'm-green' if avg_score >= 55 else 'm-orange' if avg_score >= 40 else 'm-red'
    chips = ''.join([
        _chip(f'{shown:,}', 'Stocks'),
        _chip(f'{avg_score:.1f}', 'Avg Score', sc_cls),
        _chip(f'{grade_s + grade_a}', 'S + A Grade', 'm-green'),
        _chip(f'{rockets}', '🚀 Rockets'),
        _chip(f'{confirmed}', '💰 Confirmed', 'm-green'),
        _chip(f'{decay_high}', '🔻 Traps', 'm-red' if decay_high > 0 else ''),
        _chip(f'{sect_leaders}', '👑 Alpha', 'm-gold'),
        _chip(f'{metadata["total_weeks"]}', 'Weeks'),
    ])
    st.markdown(f'<div class="m-strip">{chips}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # SECTION 2 — CONTROL PANEL + SMART TABLE
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">📋 Trajectory Rankings</div>', unsafe_allow_html=True)

    # ── Control row: Show Top | Sort | View | Export ──
    ctl0, ctl1, ctl2, ctl3 = st.columns([0.8, 1.3, 1.3, 1.0])
    with ctl0:
        display_n = st.selectbox("Show Top", [10, 20, 50, 100, 200, 500],
                                  index=3, key='rank_topn')
    with ctl1:
        sort_by = st.selectbox("Sort by", [
            'Trajectory Score', 'Current Rank', 'Rank Change', 'TMI',
            'Positional Quality', 'Best Rank', 'Streak', 'Trend', 'Velocity',
            'Consistency', 'Price Alignment', 'Decay Score', 'Sector Alpha'
        ], key='rank_sort', label_visibility='collapsed')
    with ctl2:
        view_mode = st.selectbox("View", [
            'Standard', 'Compact', 'Signals', 'Complete', 'Custom'
        ], key='rank_view', label_visibility='collapsed')
    with ctl3:
        export_btn = st.button("📥 Export CSV", key='rank_export', use_container_width=True)

    # ── Apply Show Top limit ──
    filtered_df = filtered_df.head(display_n).copy()
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df['t_rank'] = range(1, len(filtered_df) + 1)
    shown = len(filtered_df)

    # ── Pro Rank = rank within full universe (stable, never changes with filters) ──
    pro_rank_sorted = all_df.sort_values('trajectory_score', ascending=False).reset_index(drop=True)
    pro_rank_map = {t: i + 1 for i, t in enumerate(pro_rank_sorted['ticker'])}
    filtered_df['pro_rank'] = filtered_df['ticker'].map(pro_rank_map).fillna(0).astype(int)

    sort_map = {
        'Trajectory Score': ('trajectory_score', False),
        'Positional Quality': ('positional', False),
        'TMI': ('tmi', False),
        'Current Rank': ('current_rank', True),
        'Rank Change': ('rank_change', False),
        'Best Rank': ('best_rank', True),
        'Streak': ('streak', False),
        'Trend': ('trend', False),
        'Velocity': ('velocity', False),
        'Consistency': ('consistency', False),
        'Price Alignment': ('price_alignment', False),
        'Decay Score': ('decay_score', True),
        'Sector Alpha': ('sector_alpha_value', False),
    }
    col_name, ascending = sort_map.get(sort_by, ('trajectory_score', False))
    display_df = filtered_df.sort_values(col_name, ascending=ascending).reset_index(drop=True)
    display_df['t_rank'] = range(1, len(display_df) + 1)

    # ── Add latest price from histories ──
    display_df['latest_price'] = display_df['ticker'].apply(
        lambda t: round(histories.get(t, {}).get('prices', [0])[-1], 2)
        if histories.get(t, {}).get('prices') else 0
    )

    # ── Column definitions for each view ──
    # Each: (df_col, display_name, tooltip, column_config_or_None)
    COL_DEFS = {
        'Pro Rank': ('pro_rank', 'Pro Rank', 'Rank in full universe (all stocks, unfiltered)',
                     st.column_config.NumberColumn(width="small")),
        'Ticker':   ('ticker', 'Ticker', 'NSE ticker symbol', None),
        'Company':  ('company_name', 'Company', 'Company name (truncated)', None),
        'Sector':   ('sector', 'Sector', 'Business sector', None),
        'Category': ('category', 'Category', 'Large/Mid/Small Cap', None),
        '₹ Price':  ('latest_price', '₹ Price', 'Latest closing price (₹)',
                     st.column_config.NumberColumn(format="₹%.2f")),
        'T-Score':  ('trajectory_score', 'T-Score', 'Composite trajectory score (0-100)',
                     st.column_config.ProgressColumn('T-Score', min_value=0, max_value=100, format="%.1f")),
        'Grade':    ('grade', 'Grade', 'S/A/B/C/D/F based on T-Score', None),
        'Pattern':  ('pattern', 'Pattern', 'Trajectory pattern classification', None),
        'Signals':  ('signal_tags', 'Signals', 'All signal tags combined', None),
        'TMI':      ('tmi', 'TMI', 'Trajectory Momentum Index (0-100)',
                     st.column_config.ProgressColumn('TMI', min_value=0, max_value=100, format="%.0f")),
        'Best':     ('best_rank', 'Best', 'Best rank ever achieved', None),
        'Δ Total':  ('rank_change', 'Δ Total', 'Total rank change (first → now)',
                     st.column_config.NumberColumn(format="%+d")),
        'Δ Week':   ('last_week_change', 'Δ Week', 'Rank change this week',
                     st.column_config.NumberColumn(format="%+d")),
        'Streak':   ('streak', 'Streak', 'Consecutive weeks improving',
                     st.column_config.NumberColumn(format="%d 🔥")),
        'Wks':      ('weeks', 'Wks', 'Total weeks tracked', None),
        'Trend':    ('trend', 'Trend', 'Trend component score', None),
        'Velocity': ('velocity', 'Velocity', 'Velocity component score', None),
        'Consistency': ('consistency', 'Consistency', 'Consistency component score', None),
        'Positional': ('positional', 'Positional', 'Positional quality score', None),
        'Price Signal': ('price_label', 'Price Signal', 'Price alignment: CONFIRMED/DIVERGENT/NEUTRAL', None),
        'Decay':    ('decay_label', 'Decay', 'Momentum decay level: HIGH/MODERATE/MILD/CLEAN', None),
        'Alpha':    ('sector_alpha_tag', 'Alpha', 'Sector alpha classification', None),
        'Trajectory': ('sparkline', 'Trajectory', 'Score trajectory over time',
                       st.column_config.LineChartColumn('Trajectory', y_min=0, y_max=100, width="medium")),
    }

    VIEW_PRESETS = {
        'Compact':  ['Pro Rank', 'Ticker', '₹ Price', 'T-Score', 'Grade', 'Pattern',
                     'Δ Total', 'Streak', 'Trajectory'],
        'Standard': ['Pro Rank', 'Ticker', 'Company', 'Sector', '₹ Price', 'T-Score', 'Grade',
                     'Pattern', 'Signals', 'TMI', 'Best', 'Δ Total', 'Δ Week', 'Streak', 'Wks', 'Trajectory'],
        'Signals':  ['Pro Rank', 'Ticker', 'Company', 'Sector', '₹ Price', 'T-Score', 'Grade',
                     'Pattern', 'Signals', 'Price Signal', 'Decay', 'Alpha', 'Trajectory'],
        'Complete': ['Pro Rank', 'Ticker', 'Company', 'Sector', 'Category', '₹ Price', 'T-Score',
                     'Grade', 'Pattern', 'Signals', 'TMI', 'Best', 'Δ Total', 'Δ Week', 'Streak', 'Wks',
                     'Trend', 'Velocity', 'Consistency', 'Positional', 'Price Signal', 'Decay', 'Alpha', 'Trajectory'],
    }

    # ── Custom view: user picks columns ──
    if view_mode == 'Custom':
        all_col_names = list(COL_DEFS.keys())
        default_custom = VIEW_PRESETS['Standard']
        selected_cols = st.multiselect(
            "Select columns to display",
            all_col_names,
            default=default_custom,
            key='custom_cols'
        )
        if not selected_cols:
            selected_cols = ['#', 'Ticker', 'T-Score', 'Grade']
    else:
        selected_cols = VIEW_PRESETS.get(view_mode, VIEW_PRESETS['Standard'])

    # ── Build table_df and column_config from selected columns ──
    df_cols = []
    disp_names = []
    col_config = {}
    for col_key in selected_cols:
        cdef = COL_DEFS.get(col_key)
        if cdef is None:
            continue
        src_col, disp_name, tooltip, cfg = cdef
        if src_col not in display_df.columns:
            continue
        df_cols.append(src_col)
        disp_names.append(disp_name)
        if cfg is not None:
            col_config[disp_name] = cfg

    table_df = display_df[df_cols].copy()
    table_df.columns = disp_names

    # ── Truncate long text columns ──
    if 'Company' in table_df.columns:
        table_df['Company'] = table_df['Company'].astype(str).str[:24]
    if 'Sector' in table_df.columns:
        table_df['Sector'] = table_df['Sector'].astype(str).str[:18]
    if 'Category' in table_df.columns:
        table_df['Category'] = table_df['Category'].astype(str).str[:12]

    # ── Dynamic height ──
    tbl_height = min(750, max(180, len(table_df) * 35 + 60))

    st.dataframe(
        table_df, column_config=col_config,
        hide_index=True, use_container_width=True, height=tbl_height
    )

    # ── Export CSV ──
    if export_btn:
        csv_data = table_df.drop(columns=['Trajectory'], errors='ignore').to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV", data=csv_data,
            file_name=f"trajectory_rankings_{metadata.get('latest_date','export')}.csv",
            mime='text/csv', key='csv_download'
        )

    # ════════════════════════════════════════════
    # SECTION 6 — INTELLIGENCE DASHBOARD (tabs)
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">📊 Intelligence</div>', unsafe_allow_html=True)

    tab_health, tab_sectors, tab_patterns, tab_grades = st.tabs([
        "🫀 Health", "🏢 Sectors", "🔮 Patterns", "📊 Grades & Alpha"
    ])

    # ─── TAB: Health ───
    with tab_health:
        h1, h2 = st.columns(2)
        with h1:
            fig_health = go.Figure(data=[go.Pie(
                labels=['Clean', 'Mild', 'Moderate', 'Severe'],
                values=[clean_n, decay_mild, decay_mod, decay_high],
                marker_colors=['#238636', '#d29922', '#da3633', '#f85149'],
                hole=0.62, textinfo='label+value', textfont_size=11, sort=False,
                textfont_color='#e6edf3'
            )])
            fig_health.update_layout(
                title=dict(text="Momentum Health", font=dict(size=13, color='#e6edf3')),
                height=340, template='plotly_dark', showlegend=False,
                margin=dict(t=45, b=15, l=15, r=15),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(text=f'{clean_pct}%<br><span style="font-size:10px;color:#8b949e">Clean</span>',
                                   x=0.5, y=0.5, font_size=18, showarrow=False, font_color='#3fb950')]
            )
            st.plotly_chart(fig_health, use_container_width=True)

        with h2:
            above70 = int((filtered_df['pre_price_score'] >= 70).sum()) if 'pre_price_score' in filtered_df.columns else 0
            price_boosted = int((filtered_df['price_multiplier'] > 1.01).sum())
            price_pen = int((filtered_df['price_multiplier'] < 0.99).sum())
            decay_pen = int((filtered_df['decay_multiplier'] < 0.99).sum())

            fig_pipe = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute", "absolute", "relative", "relative", "relative", "absolute"],
                x=["Universe", "Pre≥70", "💰 Boost", "⚠️ Price↓", "🔻 Decay↓", "S+A Final"],
                y=[shown, above70, price_boosted, -price_pen, -decay_pen, grade_s + grade_a],
                connector={"line": {"color": "#30363d"}},
                increasing={"marker": {"color": "#238636"}},
                decreasing={"marker": {"color": "#da3633"}},
                totals={"marker": {"color": "#FF6B35"}},
                text=[shown, above70, f"+{price_boosted}", f"−{price_pen}", f"−{decay_pen}", grade_s + grade_a],
                textposition="outside", textfont_color='#e6edf3'
            ))
            fig_pipe.update_layout(
                title=dict(text="Score Pipeline", font=dict(size=13, color='#e6edf3')),
                height=340, template='plotly_dark', showlegend=False,
                margin=dict(t=45, b=25, l=30, r=15),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='#21262d'),
            )
            st.plotly_chart(fig_pipe, use_container_width=True)

        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("Clean Stocks", f"{clean_n:,}", f"{clean_pct}%")
        hm2.metric("Price Confirmed", confirmed, f"{round(confirmed/max(shown,1)*100,1)}%")
        hm3.metric("Decay Warnings", decay_any, f"−{round(decay_any/max(shown,1)*100,1)}%", delta_color="inverse")
        hm4.metric("Divergent", divergent, f"{round(divergent/max(shown,1)*100,1)}%", delta_color="inverse")

    # ─── TAB: Sectors ───
    with tab_sectors:
        qualified = filtered_df[filtered_df['weeks'] >= 3].copy()
        if not qualified.empty:
            sect_agg = qualified.groupby('sector').agg(
                avg_score=('trajectory_score', 'mean'),
                count=('trajectory_score', 'count'),
                leaders=('sector_alpha_tag', lambda x: (x == 'SECTOR_LEADER').sum()),
                decay_n=('decay_label', lambda x: x.isin(['DECAY_HIGH', 'DECAY_MODERATE']).sum()),
            ).reset_index()
            sect_agg = sect_agg[sect_agg['count'] >= 3].sort_values('avg_score', ascending=False).head(20)

            if not sect_agg.empty:
                max_s, min_s = sect_agg['avg_score'].max(), sect_agg['avg_score'].min()
                rng = max(max_s - min_s, 1)
                colors = []
                for s in sect_agg['avg_score']:
                    r = (s - min_s) / rng
                    colors.append('#238636' if r > 0.7 else '#d29922' if r > 0.35 else '#da3633')

                bar_labels = [
                    f"{row['sector'][:22]}  ·  {int(row['count'])} stk  ·  👑{int(row['leaders'])}"
                    for _, row in sect_agg.iterrows()
                ]
                fig_sec = go.Figure(data=[go.Bar(
                    x=sect_agg['avg_score'].values, y=bar_labels,
                    orientation='h', marker_color=colors,
                    text=[f"{v:.1f}" for v in sect_agg['avg_score']],
                    textposition='auto', textfont_color='#e6edf3'
                )])
                fig_sec.update_layout(
                    title=dict(text="Sectors by Avg Trajectory Score", font=dict(size=13, color='#e6edf3')),
                    height=min(480, 70 + len(sect_agg) * 24), template='plotly_dark',
                    xaxis_title='Avg T-Score', yaxis=dict(autorange='reversed'),
                    margin=dict(t=45, b=35, l=250, r=15),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#21262d'),
                )
                st.plotly_chart(fig_sec, use_container_width=True)
        else:
            st.info("Need 3+ weeks of data for sector analysis")

    # ─── TAB: Patterns ───
    with tab_patterns:
        p1, p2 = st.columns([3, 2])
        with p1:
            pattern_counts = filtered_df['pattern_key'].value_counts()
            p_labels = [PATTERN_DEFS.get(k, ('', k, ''))[1] for k in pattern_counts.index]
            p_colors = [PATTERN_COLORS.get(k, '#8b949e') for k in pattern_counts.index]
            fig_pat = go.Figure(data=[go.Pie(
                labels=p_labels, values=pattern_counts.values,
                marker_colors=p_colors, hole=0.5,
                textinfo='label+percent', textfont_size=10,
                textfont_color='#e6edf3'
            )])
            fig_pat.update_layout(
                title=dict(text="Pattern Distribution", font=dict(size=13, color='#e6edf3')),
                height=380, template='plotly_dark', showlegend=False,
                margin=dict(t=45, b=15, l=15, r=15),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_pat, use_container_width=True)

        with p2:
            st.markdown("**Pattern Breakdown**")
            for pk in pattern_counts.index[:10]:
                emoji, name, _ = PATTERN_DEFS.get(pk, ('•', pk, ''))
                cnt = int(pattern_counts[pk])
                pct = round(cnt / max(shown, 1) * 100, 1)
                bar_w = min(pct * 3, 100)
                bar_color = PATTERN_COLORS.get(pk, '#8b949e')
                st.markdown(f"""<div style="margin-bottom:5px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#e6edf3;">
                        <span>{emoji} {name}</span><span style="color:#8b949e;">{cnt} ({pct}%)</span>
                    </div>
                    <div class="pbar-wrap"><div class="pbar" style="background:{bar_color};width:{bar_w}%;"></div></div>
                </div>""", unsafe_allow_html=True)

    # ─── TAB: Grades & Alpha ───
    with tab_grades:
        ga1, ga2 = st.columns(2)
        with ga1:
            grade_counts = filtered_df['grade'].value_counts().reindex(['S', 'A', 'B', 'C', 'D', 'F']).fillna(0)
            fig_gr = go.Figure(data=[go.Bar(
                x=grade_counts.index, y=grade_counts.values,
                marker_color=[GRADE_COLORS.get(g, '#8b949e') for g in grade_counts.index],
                text=[int(v) for v in grade_counts.values],
                textposition='outside', width=0.6,
                textfont_color='#e6edf3'
            )])
            fig_gr.update_layout(
                title=dict(text="Grade Distribution", font=dict(size=13, color='#e6edf3')),
                height=320, template='plotly_dark',
                margin=dict(t=45, b=35, l=35, r=15),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='#21262d'),
            )
            st.plotly_chart(fig_gr, use_container_width=True)

        with ga2:
            alpha_order = ['SECTOR_LEADER', 'SECTOR_OUTPERFORM', 'SECTOR_ALIGNED',
                           'SECTOR_BETA', 'SECTOR_LAGGARD', 'NEUTRAL']
            alpha_cmap = {'SECTOR_LEADER': '#d29922', 'SECTOR_OUTPERFORM': '#238636',
                          'SECTOR_ALIGNED': '#8b949e', 'SECTOR_BETA': '#da3633',
                          'SECTOR_LAGGARD': '#f85149', 'NEUTRAL': '#484f58'}
            alpha_counts = filtered_df['sector_alpha_tag'].value_counts()
            al, av, ac = [], [], []
            for a in alpha_order:
                if a in alpha_counts.index:
                    al.append(a.replace('SECTOR_', ''))
                    av.append(int(alpha_counts[a]))
                    ac.append(alpha_cmap.get(a, '#8b949e'))
            if al:
                fig_al = go.Figure(data=[go.Bar(
                    x=al, y=av, marker_color=ac,
                    text=av, textposition='outside', width=0.5,
                    textfont_color='#e6edf3'
                )])
                fig_al.update_layout(
                    title=dict(text="Sector Alpha Classification", font=dict(size=13, color='#e6edf3')),
                    height=320, template='plotly_dark',
                    margin=dict(t=45, b=35, l=35, r=15),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='#21262d'),
                )
                st.plotly_chart(fig_al, use_container_width=True)


# ============================================
# UI: SEARCH TAB
# ============================================

def render_search_tab(filtered_df: pd.DataFrame, traj_df: pd.DataFrame, histories: dict, dates_iso: list):
    """Search & Analyse — v3.0 (Price Trajectory, Pro Rank, Latest Price, Clean UI)

    Args:
        filtered_df: Category/sector-filtered stocks (for dropdown).
        traj_df:     Full unfiltered data (for Pro Rank against universe).
    """

    # ── Search Input — dropdown shows only filtered stocks ──
    label_map = {}
    for _, row in filtered_df.iterrows():
        label_map[f"{row['ticker']} — {row['company_name'][:35]}"] = row['ticker']
    labels = sorted(label_map.keys())

    # Clear stale selection if it no longer exists in filtered labels
    if 'search_select' in st.session_state and st.session_state['search_select'] not in labels:
        st.session_state['search_select'] = None

    selected_label = st.selectbox("🔍 Search Stock",
                                   labels, index=None,
                                   placeholder="Type ticker or company name...",
                                   key='search_select')

    if selected_label is None:
        st.info("👆 Select a stock to view detailed trajectory analysis")
        return

    ticker = label_map[selected_label]
    matches = filtered_df[filtered_df['ticker'] == ticker]
    if matches.empty:
        st.warning("Stock not found in current filter selection")
        return
    row = matches.iloc[0]
    h = histories.get(ticker, {})
    if not h:
        st.warning("No history data available for this ticker")
        return

    # ── Derived Data ──
    latest_price = h['prices'][-1] if h.get('prices') else 0
    pcts = ranks_to_percentiles(h['ranks'], h['total_per_week'])
    total_stocks = h['total_per_week'][-1] if h.get('total_per_week') else 0
    # Pro Rank = rank within full universe (all stocks, not filtered)
    sorted_df = traj_df.sort_values('trajectory_score', ascending=False).reset_index(drop=True)
    pro_rank_idx = sorted_df[sorted_df['ticker'] == ticker].index
    pro_rank = int(pro_rank_idx[0]) + 1 if len(pro_rank_idx) > 0 else 0

    pattern_key = row.get('pattern_key', 'neutral')
    p_emoji, p_name, p_desc = PATTERN_DEFS.get(pattern_key, ('➖', 'Neutral', ''))
    p_color = PATTERN_COLORS.get(pattern_key, '#8b949e')
    grade_color = {'S': '#FFD700', 'A': '#3fb950', 'B': '#58a6ff', 'C': '#d29922', 'D': '#FF5722', 'F': '#f85149'}.get(row['grade'], '#888')

    # ── Header Card ──
    st.markdown(f"""
    <div style="background:#0d1117; border-radius:14px; padding:20px 24px; margin-bottom:16px; border:1px solid #30363d;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
            <div style="flex:1; min-width:200px;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
                    <span style="font-size:1.6rem; font-weight:800; color:#fff;">{ticker}</span>
                    <span style="background:{p_color}22; color:{p_color}; padding:3px 10px; border-radius:12px; font-size:0.75rem; border:1px solid {p_color}44;">{p_emoji} {p_name}</span>
                </div>
                <div style="color:#8b949e; font-size:0.95rem; margin-bottom:2px;">{row['company_name']}</div>
                <div style="color:#484f58; font-size:0.8rem;">{row['category']} • {row.get('sector', '')} • {row.get('industry', '')}</div>
            </div>
            <div style="display:flex; gap:20px; align-items:center;">
                <div style="text-align:center;">
                    <div style="font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">Pro Rank</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#58a6ff;">#{pro_rank}</div>
                    <div style="font-size:0.65rem; color:#484f58;">of {total_stocks}</div>
                </div>
                <div style="width:1px; height:50px; background:#30363d;"></div>
                <div style="text-align:center;">
                    <div style="font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">T-Score</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#FF6B35;">{row['trajectory_score']:.1f}</div>
                    <div style="font-size:0.65rem; color:#484f58;">/ 100</div>
                </div>
                <div style="width:1px; height:50px; background:#30363d;"></div>
                <div style="text-align:center;">
                    <div style="font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">Grade</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{grade_color};">{row['grade']}</div>
                    <div style="font-size:0.65rem; color:#484f58;">{row['grade_emoji']}</div>
                </div>
                <div style="width:1px; height:50px; background:#30363d;"></div>
                <div style="text-align:center;">
                    <div style="font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">Price</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#e6edf3;">₹{latest_price:,.1f}</div>
                    <div style="font-size:0.65rem; color:#484f58;">Latest</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Compact KPI Strip ──
    price_label_display = row.get('price_label', 'NEUTRAL')
    decay_lbl = row.get('decay_label', '')
    sa_tag = row.get('sector_alpha_tag', 'NEUTRAL')
    sa_icons = {'SECTOR_LEADER': '👑', 'SECTOR_OUTPERFORM': '⬆️', 'SECTOR_ALIGNED': '➖', 'SECTOR_BETA': '🏷️', 'SECTOR_LAGGARD': '📉'}

    kpi_items = [
        ('CSV Rank', f"#{row['current_rank']}", f"{row['last_week_change']:+d}w"),
        ('Best Rank', f"#{row['best_rank']}", ''),
        ('Total Δ', f"{row['rank_change']:+d}", 'pos' if row['rank_change'] > 0 else ''),
        ('TMI', f"{row['tmi']:.0f}", ''),
        ('Streak', f"{row['streak']}w", ''),
        ('Price Align', f"{'💰' if price_label_display == 'PRICE_CONFIRMED' else '⚠️' if price_label_display == 'PRICE_DIVERGENT' else '➖'} {row.get('price_alignment', 50):.0f}", ''),
        ('Decay', f"{'🔻' if decay_lbl == 'DECAY_HIGH' else '⚡' if decay_lbl == 'DECAY_MODERATE' else '✅'} {row.get('decay_score', 0)}", ''),
        ('Sector', f"{sa_icons.get(sa_tag, '➖')}", sa_tag.split('_')[-1].title() if sa_tag != 'NEUTRAL' else 'Neutral'),
    ]
    kpi_html = ''.join([
        f'<div class="m-chip"><div style="font-size:0.62rem;color:#8b949e;text-transform:uppercase;">{label}</div>'
        f'<div style="font-size:0.95rem;font-weight:700;color:#e6edf3;">{val}</div>'
        f'<div style="font-size:0.6rem;color:#6e7681;">{sub}</div></div>'
        for label, val, sub in kpi_items
    ])
    st.markdown(f'<div class="m-strip">{kpi_html}</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Row 1: Rank Trajectory + Radar ──
    chart_c1, chart_c2 = st.columns([3, 2])

    with chart_c1:
        st.markdown('<div class="sec-head">📊 Rank Trajectory</div>', unsafe_allow_html=True)
        _render_rank_chart(h, ticker)

    with chart_c2:
        st.markdown('<div class="sec-head">🎯 Component Breakdown</div>', unsafe_allow_html=True)
        _render_radar_chart(row)

    # ── Row 2: Price Trajectory (full width) ──
    st.markdown('<div class="sec-head">💰 Price Trajectory</div>', unsafe_allow_html=True)
    _render_price_chart(h, ticker)

    # ── Score Pipeline Detail ──
    st.markdown('<div class="sec-head">🔬 Score Pipeline</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        # Component Scores Table
        avg_pct_val = float(np.mean(h.get('ranks', [500])))
        total_wk = h.get('total_per_week', [2000])
        avg_total = float(np.mean(total_wk)) if total_wk else 2000
        stock_avg_pct = (1 - avg_pct_val / max(avg_total, 1)) * 100
        adp_w = _get_adaptive_weights(stock_avg_pct)
        comp_data = {
            'Component': ['Positional', 'Trend', 'Velocity', 'Acceleration', 'Consistency', 'Resilience'],
            'Wt': [f"{adp_w[k]*100:.0f}%" for k in ['positional','trend','velocity','acceleration','consistency','resilience']],
            'Score': [row['positional'], row['trend'], row['velocity'], row['acceleration'], row['consistency'], row['resilience']],
            'Contrib': [round(row[k] * adp_w[k], 1) for k in ['positional','trend','velocity','acceleration','consistency','resilience']]
        }
        st.dataframe(pd.DataFrame(comp_data), column_config={
            'Score': st.column_config.ProgressColumn('Score', min_value=0, max_value=100, format="%.1f")
        }, hide_index=True, use_container_width=True, height=248)

    with sc2:
        # Price Alignment + Momentum Decay
        pa_label = row.get('price_label', 'NEUTRAL')
        pa_color = '#00E676' if pa_label == 'PRICE_CONFIRMED' else '#FF1744' if pa_label == 'PRICE_DIVERGENT' else '#484f58'
        d_label = row.get('decay_label', '')
        d_color = '#FF1744' if d_label == 'DECAY_HIGH' else '#FF9800' if d_label == 'DECAY_MODERATE' else '#FFD700' if d_label == 'DECAY_MILD' else '#3fb950'
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d; margin-bottom:10px;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">💰 Price Alignment</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Score</span>
                <span style="color:{pa_color}; font-weight:700;">{row.get('price_alignment', 50):.0f}/100</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Multiplier</span>
                <span style="color:{pa_color}; font-weight:600;">×{row.get('price_multiplier', 1.0):.3f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Status</span>
                <span style="color:{pa_color}; font-weight:700;">{pa_label.replace('_', ' ')}</span>
            </div>
            <div style="margin-top:6px; color:#484f58; font-size:0.7rem;">Pre: {row.get('pre_price_score', row['trajectory_score']):.1f} → Post: {row.get('pre_decay_score', row['trajectory_score']):.1f}</div>
        </div>
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">🔻 Momentum Decay</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Score</span>
                <span style="color:{d_color}; font-weight:700;">{row.get('decay_score', 0)}/100</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Multiplier</span>
                <span style="color:{d_color}; font-weight:600;">×{row.get('decay_multiplier', 1.0):.3f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Status</span>
                <span style="color:{d_color}; font-weight:700;">{d_label if d_label else 'CLEAN ✅'}</span>
            </div>
            <div style="margin-top:6px; color:#484f58; font-size:0.7rem;">Pre: {row.get('pre_decay_score', row['trajectory_score']):.1f} → Final: {row['trajectory_score']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with sc3:
        # Sector Alpha + Trajectory Stats
        sa_val = row.get('sector_alpha_value', 0)
        sa_colors_map = {'SECTOR_LEADER': '#FFD700', 'SECTOR_OUTPERFORM': '#3fb950', 'SECTOR_ALIGNED': '#484f58', 'SECTOR_BETA': '#FF9800', 'SECTOR_LAGGARD': '#FF1744'}
        sa_color = sa_colors_map.get(sa_tag, '#484f58')
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d; margin-bottom:10px;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">🏛️ Sector Alpha</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Sector</span>
                <span style="color:#e6edf3; font-weight:600;">{row.get('sector', 'N/A')}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Alpha</span>
                <span style="color:{sa_color}; font-weight:700;">{sa_val:+.1f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Tag</span>
                <span style="color:{sa_color}; font-weight:700;">{sa_icons.get(sa_tag, '➖')} {sa_tag.replace('SECTOR_', '').title()}</span>
            </div>
        </div>
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">📊 Trajectory Stats</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Worst Rank</span>
                <span style="color:#e6edf3;">#{row['worst_rank']}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Avg Rank</span>
                <span style="color:#e6edf3;">#{row['avg_rank']}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Rank Volatility</span>
                <span style="color:#e6edf3;">{row['rank_volatility']:.1f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Weeks</span>
                <span style="color:#e6edf3;">{row['weeks']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Latest Wave Detection Patterns
    if row.get('latest_patterns', ''):
        st.markdown(f'<div style="margin-top:8px;"><span style="color:#8b949e; font-size:0.72rem; text-transform:uppercase;">Wave Patterns:</span> <span class="pattern-tag">{row["latest_patterns"]}</span></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Week-by-Week History ──
    with st.expander("📅 Week-by-Week History", expanded=False):
        week_data = {
            'Date': h['dates'],
            'Rank': [int(r) for r in h['ranks']],
            'Pctl': [round(p, 1) for p in pcts],
            'Price ₹': [round(p, 1) for p in h['prices']],
            'M.Score': [round(s, 1) for s in h['scores']],
            'Stocks': h['total_per_week'],
        }
        wk_changes = [0] + [int(h['ranks'][i - 1] - h['ranks'][i]) for i in range(1, len(h['ranks']))]
        week_data['Δ Rank'] = wk_changes
        # Price changes
        price_changes = [0] + [round(h['prices'][i] - h['prices'][i-1], 1) for i in range(1, len(h['prices']))]
        week_data['Δ Price'] = price_changes

        def _safe_round(lst, decimals=1):
            return [round(v, decimals) if v is not None and not np.isnan(v) else None for v in lst]
        if h.get('ret_7d'):
            week_data['Ret 7d%'] = _safe_round(h['ret_7d'])
        if h.get('ret_30d'):
            week_data['Ret 30d%'] = _safe_round(h['ret_30d'])

        wk_df = pd.DataFrame(week_data).iloc[::-1]  # Latest first
        wk_col_config = {
            'Δ Rank': st.column_config.NumberColumn(format="%+d"),
            'Pctl': st.column_config.ProgressColumn('Pctl', min_value=0, max_value=100, format="%.1f"),
            'Price ₹': st.column_config.NumberColumn(format="₹%.1f"),
            'Δ Price': st.column_config.NumberColumn(format="%+.1f"),
        }
        if 'Ret 7d%' in wk_df.columns:
            wk_col_config['Ret 7d%'] = st.column_config.NumberColumn(format="%.1f%%")
        if 'Ret 30d%' in wk_df.columns:
            wk_col_config['Ret 30d%'] = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(wk_df, column_config=wk_col_config, hide_index=True, use_container_width=True)

    # ── Compare ──
    with st.expander("⚖️ Compare Stocks", expanded=False):
        compare_labels = [l for l in labels if l != selected_label]
        compare_selections = st.multiselect("Select up to 4 stocks",
                                             compare_labels, max_selections=4,
                                             key='compare_select')
        if compare_selections:
            compare_tickers = [label_map[l] for l in compare_selections]
            _render_comparison_chart(ticker, compare_tickers, histories, traj_df)


def _render_rank_chart(h: dict, ticker: str):
    """Rank + Master Score dual-axis trajectory chart"""
    dates = h['dates']
    ranks = h['ranks']
    scores = h['scores']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Rank (primary y — inverted so lower rank = higher on chart)
    fig.add_trace(go.Scatter(
        x=dates, y=ranks,
        mode='lines+markers',
        name='Rank',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(255,107,53,0.08)',
        hovertemplate='%{x}<br>Rank: #%{y}<extra></extra>'
    ), secondary_y=False)

    # Master Score (secondary y)
    fig.add_trace(go.Scatter(
        x=dates, y=scores,
        mode='lines+markers',
        name='Master Score',
        line=dict(color='#58a6ff', width=2, dash='dot'),
        marker=dict(size=4),
        opacity=0.85,
        hovertemplate='%{x}<br>M.Score: %{y:.1f}<extra></extra>'
    ), secondary_y=True)

    # Best rank annotation
    best_idx = int(np.argmin(ranks))
    fig.add_annotation(
        x=dates[best_idx], y=ranks[best_idx],
        text=f"Best: #{int(ranks[best_idx])}",
        showarrow=True, arrowhead=2,
        font=dict(color='#FFD700', size=10),
        bgcolor='rgba(0,0,0,0.7)', bordercolor='#FFD700'
    )

    fig.update_layout(
        height=340,
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
        margin=dict(t=10, b=55, l=50, r=50),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_yaxes(title_text="Rank", autorange="reversed",
                     secondary_y=False, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text="Master Score",
                     secondary_y=True, gridcolor='rgba(255,255,255,0.02)')

    st.plotly_chart(fig, use_container_width=True)


def _render_price_chart(h: dict, ticker: str):
    """Price trajectory chart with min/max annotations"""
    dates = h['dates']
    prices = h['prices']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode='lines+markers',
        name='Price ₹',
        line=dict(color='#00C853', width=2.5),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor='rgba(0,200,83,0.06)',
        hovertemplate='%{x}<br>₹%{y:,.1f}<extra></extra>'
    ))

    # High / Low annotations
    hi_idx = int(np.argmax(prices))
    lo_idx = int(np.argmin(prices))
    fig.add_annotation(
        x=dates[hi_idx], y=prices[hi_idx],
        text=f"High: ₹{prices[hi_idx]:,.1f}",
        showarrow=True, arrowhead=2,
        font=dict(color='#00E676', size=10),
        bgcolor='rgba(0,0,0,0.7)', bordercolor='#00E676'
    )
    if hi_idx != lo_idx:
        fig.add_annotation(
            x=dates[lo_idx], y=prices[lo_idx],
            text=f"Low: ₹{prices[lo_idx]:,.1f}",
            showarrow=True, arrowhead=2, ay=30,
            font=dict(color='#FF5252', size=10),
            bgcolor='rgba(0,0,0,0.7)', bordercolor='#FF5252'
        )

    fig.update_layout(
        height=280,
        template='plotly_dark',
        hovermode='x unified',
        margin=dict(t=10, b=50, l=50, r=20),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(title='Price ₹', gridcolor='rgba(255,255,255,0.04)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_radar_chart(row):
    """Render radar/spider chart for component scores (6 components)"""
    categories = ['Positional', 'Trend', 'Velocity', 'Acceleration', 'Consistency', 'Resilience']
    values = [row['positional'], row['trend'], row['velocity'], row['acceleration'],
              row['consistency'], row['resilience']]
    values_closed = values + [values[0]]  # Close the polygon
    cats_closed = categories + [categories[0]]

    fig = go.Figure()

    # Reference circle at 50
    fig.add_trace(go.Scatterpolar(
        r=[50] * 6,
        theta=cats_closed,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Score polygon
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill='toself',
        fillcolor='rgba(255,107,53,0.25)',
        line=dict(color='#FF6B35', width=2),
        name='Scores',
        hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                           tickfont=dict(size=9), gridcolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        height=380,
        template='plotly_dark',
        showlegend=False,
        margin=dict(t=20, b=20, l=60, r=60)
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_comparison_chart(main_ticker: str, compare_tickers: list,
                              histories: dict, traj_df: pd.DataFrame):
    """Render comparison chart for multiple stocks"""
    colors = ['#FF6B35', '#00C853', '#2196F3', '#FFD700', '#E040FB']
    all_tickers = [main_ticker] + compare_tickers

    fig = go.Figure()
    for i, ticker in enumerate(all_tickers):
        h = histories.get(ticker, {})
        if not h:
            continue
        # Convert to percentiles for fair comparison
        pcts = ranks_to_percentiles(h['ranks'], h['total_per_week'])
        name_row = traj_df[traj_df['ticker'] == ticker]
        label = f"{ticker}"
        if not name_row.empty:
            label = f"{ticker} (T:{name_row.iloc[0]['trajectory_score']:.0f})"

        fig.add_trace(go.Scatter(
            x=h['dates'], y=pcts,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[i % len(colors)], width=3 if i == 0 else 2),
            marker=dict(size=6 if i == 0 else 4),
            opacity=1.0 if i == 0 else 0.8
        ))

    fig.update_layout(
        title="Rank Percentile Comparison (higher = better rank)",
        height=400,
        template='plotly_dark',
        hovermode='x unified',
        yaxis=dict(title='Rank Percentile', range=[0, 100]),
        xaxis=dict(title='Week'),
        legend=dict(orientation='h', y=-0.2),
        margin=dict(t=40, b=60, l=60, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    comp_rows = []
    for ticker in all_tickers:
        r = traj_df[traj_df['ticker'] == ticker]
        if r.empty:
            continue
        r = r.iloc[0]
        comp_rows.append({
            'Ticker': ticker,
            'Company': r['company_name'][:25],
            'T-Score': r['trajectory_score'],
            'Grade': f"{r['grade_emoji']} {r['grade']}",
            'Pattern': r['pattern'],
            'TMI': r['tmi'],
            'Rank Now': r['current_rank'],
            'Best': r['best_rank'],
            'Δ Rank': r['rank_change'],
            'Streak': r['streak']
        })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)


# ============================================
# UI: FUNNEL TAB (3-Stage Selection System)
# ============================================

def render_funnel_tab(filtered_df: pd.DataFrame, traj_df: pd.DataFrame, histories: dict, metadata: dict):
    """3-Stage Selection Funnel — v3.0 (Clean, Minimal, Smart)

    Args:
        filtered_df: Category/sector-filtered stocks (funnel input).
        traj_df:     Full unfiltered data (for total universe count).
    """

    # ── Header ──
    st.markdown("""
    <div style="background:#0d1117; border-radius:14px; padding:18px 24px; margin-bottom:16px; border:1px solid #30363d;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div>
                <span style="font-size:1.4rem; font-weight:800; color:#fff;">🎯 Selection Funnel</span>
                <div style="color:#8b949e; font-size:0.85rem; margin-top:2px;">Systematic filtering: Discovery → Validation → Final Buys</div>
            </div>
            <div style="display:flex; gap:6px;">
                <span style="background:#58a6ff22; color:#58a6ff; padding:4px 10px; border-radius:8px; font-size:0.72rem; border:1px solid #58a6ff44;">Stage 1: Score + Pattern</span>
                <span style="background:#d2992222; color:#d29922; padding:4px 10px; border-radius:8px; font-size:0.72rem; border:1px solid #d2992244;">Stage 2: 5-Rule Engine</span>
                <span style="background:#3fb95022; color:#3fb950; padding:4px 10px; border-radius:8px; font-size:0.72rem; border:1px solid #3fb95044;">Stage 3: Final Filter</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Configuration (compact inline) ──
    with st.expander("⚙️ Funnel Configuration", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown('<div style="color:#58a6ff; font-weight:700; font-size:0.85rem; margin-bottom:4px;">Stage 1 — Discovery</div>', unsafe_allow_html=True)
            s1_score = st.number_input("Min T-Score", 30, 100, FUNNEL_DEFAULTS['stage1_score'], key='f_s1')
            s1_patterns = st.multiselect(
                "Include Patterns",
                ['rocket', 'breakout', 'momentum_building', 'stable_elite', 'at_peak', 'steady_climber', 'recovery'],
                default=FUNNEL_DEFAULTS['stage1_patterns'], key='f_s1_pat'
            )
        with fc2:
            st.markdown('<div style="color:#d29922; font-weight:700; font-size:0.85rem; margin-bottom:4px;">Stage 2 — Validation</div>', unsafe_allow_html=True)
            s2_tq = st.number_input("Min Trend Quality", 30, 100, FUNNEL_DEFAULTS['stage2_tq'], key='f_s2_tq')
            s2_ms = st.number_input("Min Master Score", 20, 100, FUNNEL_DEFAULTS['stage2_master_score'], key='f_s2_ms')
            s2_rules = st.number_input("Min Rules (of 5)", 2, 5, FUNNEL_DEFAULTS['stage2_min_rules'], key='f_s2_r')
        with fc3:
            st.markdown('<div style="color:#3fb950; font-weight:700; font-size:0.85rem; margin-bottom:4px;">Stage 3 — Final</div>', unsafe_allow_html=True)
            s3_tq = st.number_input("Min TQ (strict)", 50, 100, FUNNEL_DEFAULTS['stage3_tq'], key='f_s3_tq')
            s3_leader = st.checkbox("Require Leader Pattern", FUNNEL_DEFAULTS['stage3_require_leader'], key='f_s3_l')
            s3_dt = st.number_input("No DOWNTREND (weeks)", 1, 10, FUNNEL_DEFAULTS['stage3_no_downtrend_weeks'], key='f_s3_dt')

    funnel_config = {
        'stage1_score': s1_score, 'stage1_patterns': s1_patterns,
        'stage2_tq': s2_tq, 'stage2_master_score': s2_ms, 'stage2_min_rules': s2_rules,
        'stage3_tq': s3_tq, 'stage3_require_leader': s3_leader, 'stage3_no_downtrend_weeks': s3_dt
    }

    # ── Execute Funnel (on filtered stocks) ──
    stage1, stage2, stage3 = run_funnel(filtered_df, histories, funnel_config)

    total = len(filtered_df)
    s1_count = len(stage1)
    s2_pass = len(stage2[stage2['s2_pass']]) if not stage2.empty and 's2_pass' in stage2.columns else 0
    s3_pass = len(stage3[stage3['final_pass']]) if not stage3.empty and 'final_pass' in stage3.columns else 0

    # ── Pipeline Metrics Strip ──
    s1_pct = round(s1_count / max(total, 1) * 100, 1)
    s2_pct = round(s2_pass / max(s1_count, 1) * 100, 1)
    s3_pct = round(s3_pass / max(s2_pass, 1) * 100, 1)
    overall_pct = round(s3_pass / max(total, 1) * 100, 2)

    pipeline_items = [
        ('📊 Universe', f'{total:,}', '100%', '#8b949e'),
        ('🔍 Discovery', f'{s1_count}', f'{s1_pct}%', '#58a6ff'),
        ('✅ Validated', f'{s2_pass}', f'{s2_pct}% pass', '#d29922'),
        ('🏆 Final Buys', f'{s3_pass}', f'{s3_pct}% pass', '#3fb950'),
        ('📌 Selection', f'{overall_pct}%', 'of universe', '#FFD700'),
    ]
    pipe_html = ''.join([
        f'<div class="m-chip">'
        f'<div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;">{label}</div>'
        f'<div style="font-size:1.3rem;font-weight:800;color:{color};">{val}</div>'
        f'<div style="font-size:0.6rem;color:#6e7681;">{sub}</div>'
        f'</div>'
        for label, val, sub, color in pipeline_items
    ])
    st.markdown(f'<div class="m-strip">{pipe_html}</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Visual Funnel Diagram ──
    fc_left, fc_right = st.columns([2, 3])

    with fc_left:
        fig_funnel = go.Figure(go.Funnel(
            y=['Universe', 'Stage 1', 'Stage 2', 'Final Buys'],
            x=[total, s1_count, s2_pass, s3_pass],
            textinfo='value+percent initial',
            textposition='inside',
            textfont=dict(size=13),
            marker=dict(
                color=['#30363d', '#58a6ff', '#d29922', '#3fb950'],
                line=dict(width=1, color='#21262d')
            ),
            connector=dict(line=dict(color='#30363d', width=1))
        ))
        fig_funnel.update_layout(
            height=300,
            template='plotly_dark',
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            showlegend=False,
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with fc_right:
        # Stage flow breakdown — single self-contained HTML block
        pat_list = ', '.join(s1_patterns[:3]) + ('...' if len(s1_patterns) > 3 else '')
        s1_bar_w = min(s1_pct, 100)
        s2_bar_w = min(s2_pct, 100)
        s3_bar_w = min(s3_pct, 100)
        leader_text = 'Leader required' if s3_leader else 'Leader optional'

        flow_html = f"""
        <div style="background:#161b22; border-radius:12px; padding:16px; border:1px solid #30363d;">
        <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:12px;">Pipeline Flow</div>
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <div style="width:8px; height:8px; border-radius:50%; background:#58a6ff; flex-shrink:0;"></div>
            <div style="flex:1;">
                <div style="display:flex; justify-content:space-between;"><span style="color:#e6edf3; font-size:0.85rem; font-weight:600;">Stage 1 — Discovery</span><span style="color:#58a6ff; font-weight:700;">{s1_count} stocks</span></div>
                <div style="color:#6e7681; font-size:0.75rem; margin-top:2px;">T-Score ≥ {s1_score} OR pattern in [{pat_list}]</div>
                <div style="background:#21262d; border-radius:4px; height:6px; margin-top:4px;"><div style="background:#58a6ff; height:6px; border-radius:4px; width:{s1_bar_w}%;"></div></div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <div style="width:8px; height:8px; border-radius:50%; background:#d29922; flex-shrink:0;"></div>
            <div style="flex:1;">
                <div style="display:flex; justify-content:space-between;"><span style="color:#e6edf3; font-size:0.85rem; font-weight:600;">Stage 2 — Validation</span><span style="color:#d29922; font-weight:700;">{s2_pass} passed</span></div>
                <div style="color:#6e7681; font-size:0.75rem; margin-top:2px;">5 rules: TQ≥{s2_tq} | No Downtrend | MS≥{s2_ms} | Δ≥-20 | Vol — need {s2_rules}/5</div>
                <div style="background:#21262d; border-radius:4px; height:6px; margin-top:4px;"><div style="background:#d29922; height:6px; border-radius:4px; width:{s2_bar_w}%;"></div></div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:8px; height:8px; border-radius:50%; background:#3fb950; flex-shrink:0;"></div>
            <div style="flex:1;">
                <div style="display:flex; justify-content:space-between;"><span style="color:#e6edf3; font-size:0.85rem; font-weight:600;">Stage 3 — Final Buys</span><span style="color:#3fb950; font-weight:700;">{s3_pass} selected</span></div>
                <div style="color:#6e7681; font-size:0.75rem; margin-top:2px;">TQ≥{s3_tq} | {leader_text} | No downtrend last {s3_dt}w</div>
                <div style="background:#21262d; border-radius:4px; height:6px; margin-top:4px;"><div style="background:#3fb950; height:6px; border-radius:4px; width:{s3_bar_w}%;"></div></div>
            </div>
        </div>
        </div>
        """
        st.markdown(flow_html, unsafe_allow_html=True)

    st.markdown("")

    # ══════════════════════════════════════════
    # FINAL BUYS — Most Important, Shown First
    # ══════════════════════════════════════════
    st.markdown('<div class="sec-head">🏆 Final Buys</div>', unsafe_allow_html=True)

    if s3_pass > 0:
        final_buys = stage3[stage3['final_pass']].copy().reset_index(drop=True)

        # Precompute Pro Rank map (T-Score sorted)
        pro_rank_sorted = traj_df.sort_values('trajectory_score', ascending=False).reset_index(drop=True)
        pro_rank_map = {t: i + 1 for i, t in enumerate(pro_rank_sorted['ticker'])}
        total_stocks = len(pro_rank_sorted)

        # Card grid — 2 per row
        for i in range(0, len(final_buys), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j >= len(final_buys):
                    break
                r = final_buys.iloc[i + j]
                h = histories.get(r['ticker'], {})
                latest_price = h['prices'][-1] if h.get('prices') else 0
                p_key = r.get('pattern_key', 'neutral')
                p_emoji, p_name, _ = PATTERN_DEFS.get(p_key, ('➖', 'Neutral', ''))
                p_color = PATTERN_COLORS.get(p_key, '#8b949e')
                grade_color = {'S': '#FFD700', 'A': '#3fb950', 'B': '#58a6ff', 'C': '#d29922', 'D': '#FF5722', 'F': '#f85149'}.get(r['grade'], '#888')
                pro_rank = pro_rank_map.get(r['ticker'], 0)

                with col:
                    st.markdown(f"""
                    <div class="final-buy-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <div>
                                <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                                    <span style="font-size:1.15rem; font-weight:800; color:#3fb950;">{r['ticker']}</span>
                                    <span style="background:{p_color}22; color:{p_color}; padding:2px 8px; border-radius:10px; font-size:0.68rem; border:1px solid {p_color}44;">{p_emoji} {p_name}</span>
                                </div>
                                <div style="color:#8b949e; font-size:0.8rem;">{r.get('company_name', '')[:35]}</div>
                                <div style="color:#484f58; font-size:0.7rem;">{r.get('category', '')} • {r.get('sector', '')}</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-size:1.5rem; font-weight:800; color:#FF6B35;">{r['trajectory_score']:.1f}</div>
                                <div style="font-size:0.6rem; color:#8b949e;">T-SCORE</div>
                            </div>
                        </div>
                        <div style="display:flex; gap:12px; margin-top:10px; padding-top:8px; border-top:1px solid #21262d; flex-wrap:wrap;">
                            <div><span style="color:#6e7681; font-size:0.65rem;">Pro Rank</span><br><span style="color:#58a6ff; font-weight:700;">#{pro_rank}</span></div>
                            <div><span style="color:#6e7681; font-size:0.65rem;">Rank</span><br><span style="color:#e6edf3; font-weight:700;">#{r['current_rank']}</span></div>
                            <div><span style="color:#6e7681; font-size:0.65rem;">Grade</span><br><span style="color:{grade_color}; font-weight:700;">{r['grade_emoji']} {r['grade']}</span></div>
                            <div><span style="color:#6e7681; font-size:0.65rem;">TMI</span><br><span style="color:#e6edf3; font-weight:700;">{r['tmi']:.0f}</span></div>
                            <div><span style="color:#6e7681; font-size:0.65rem;">TQ</span><br><span style="color:#e6edf3; font-weight:700;">{r.get('latest_tq', 0):.0f}</span></div>
                            <div><span style="color:#6e7681; font-size:0.65rem;">Rules</span><br><span style="color:#3fb950; font-weight:700;">{r.get('rules_passed', 0)}/5</span></div>
                            <div><span style="color:#6e7681; font-size:0.65rem;">Price</span><br><span style="color:#e6edf3; font-weight:700;">₹{latest_price:,.1f}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Summary table
        st.markdown("")
        fb_cols = ['ticker', 'company_name', 'category', 'trajectory_score', 'grade',
                   'pattern', 'tmi', 'current_rank', 'best_rank', 'rank_change',
                   'latest_tq', 'rules_passed']
        fb_display = final_buys[[c for c in fb_cols if c in final_buys.columns]].copy()
        fb_display.columns = [c.replace('_', ' ').title() for c in fb_display.columns]
        st.dataframe(fb_display, column_config={
            'Trajectory Score': st.column_config.ProgressColumn('T-Score', min_value=0, max_value=100, format="%.1f"),
        }, hide_index=True, use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#161b22; border-radius:10px; padding:24px; text-align:center; border:1px solid #30363d;">
            <div style="font-size:1.3rem; margin-bottom:6px;">🔍</div>
            <div style="color:#8b949e; font-size:0.9rem;">No stocks passed all 3 stages</div>
            <div style="color:#484f58; font-size:0.8rem; margin-top:4px;">Try relaxing Stage 3 criteria in the config above</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ══════════════════════════════════════════
    # STAGE 2: Validation Detail
    # ══════════════════════════════════════════
    with st.expander(f"✅ Stage 2 — Validation ({s2_pass} passed / {len(stage2) if not stage2.empty else 0} tested)", expanded=False):
        if not stage2.empty:
            st.markdown(f'<div style="color:#6e7681; font-size:0.8rem; margin-bottom:8px;">5 Rules: TQ≥{s2_tq} | Not DOWNTREND | MS≥{s2_ms} | Δ≥-20 | Volume — need {s2_rules}/5</div>', unsafe_allow_html=True)
            s2_display_cols = ['ticker', 'company_name', 'trajectory_score', 'rules_passed',
                               's2_pass', 'rules_detail', 'pattern', 'current_rank']
            s2_display = stage2[[c for c in s2_display_cols if c in stage2.columns]].copy()
            s2_display.columns = ['Ticker', 'Company', 'T-Score', 'Rules', 'Pass', 'Detail', 'Pattern', 'Rank']
            s2_display['Company'] = s2_display['Company'].str[:30]
            st.dataframe(s2_display, column_config={
                'Pass': st.column_config.CheckboxColumn('Pass'),
                'T-Score': st.column_config.ProgressColumn('T-Score', min_value=0, max_value=100, format="%.1f"),
            }, hide_index=True, use_container_width=True, height=400)
        else:
            st.caption("No stocks entered Stage 2")

    # ══════════════════════════════════════════
    # STAGE 1: Discovery
    # ══════════════════════════════════════════
    with st.expander(f"🔍 Stage 1 — Discovery ({s1_count} candidates)", expanded=False):
        if not stage1.empty:
            s1_cols = ['ticker', 'company_name', 'category', 'trajectory_score',
                       'grade', 'pattern', 'tmi', 'current_rank']
            s1_display = stage1[[c for c in s1_cols if c in stage1.columns]].head(100).copy()
            s1_display.columns = [c.replace('_', ' ').title() for c in s1_display.columns]
            st.dataframe(s1_display, column_config={
                'Trajectory Score': st.column_config.ProgressColumn('T-Score', min_value=0, max_value=100, format="%.1f"),
                'Tmi': st.column_config.ProgressColumn('TMI', min_value=0, max_value=100, format="%.0f"),
            }, hide_index=True, use_container_width=True, height=400)
        else:
            st.caption("No stocks passed Stage 1")

    # ══════════════════════════════════════════
    # NEAR MISSES
    # ══════════════════════════════════════════
    if not stage3.empty:
        near_misses = stage3[~stage3['final_pass']].copy()
        if len(near_misses) > 0:
            with st.expander(f"📋 Near Misses — {len(near_misses)} stocks passed Stage 2 but failed Stage 3", expanded=False):
                nm_cols = ['ticker', 'company_name', 'trajectory_score', 's3_detail', 'latest_tq', 'current_rank']
                nm_display = near_misses[[c for c in nm_cols if c in near_misses.columns]].copy()
                nm_display.columns = ['Ticker', 'Company', 'T-Score', 'S3 Detail', 'TQ', 'Rank']
                nm_display['Company'] = nm_display['Company'].str[:30]
                st.dataframe(nm_display, column_config={
                    'T-Score': st.column_config.ProgressColumn('T-Score', min_value=0, max_value=100, format="%.1f"),
                }, hide_index=True, use_container_width=True)


# ============================================
# UI: ALERTS TAB
# ============================================

def render_alerts_tab(filtered_df: pd.DataFrame, histories: dict):
    """Alerts Tab v5.0 — Ultimate Edition.

    - Readable font sizes (no sub-0.7 rem)
    - Batch CSS-grid HTML per section
    - Top Movers: 50 per side, multi-week filter (1w / 2w / 4w / 8w / All)
    """

    # ── Ensure columns ──────────────────────────────────────────────────
    _DEFAULTS = [
        ('decay_label', ''), ('decay_multiplier', 1.0),
        ('price_label', 'NEUTRAL'), ('sector_alpha_tag', 'NEUTRAL'),
        ('grade', 'F'), ('grade_emoji', '📉'),
        ('pattern_key', 'neutral'), ('pattern', '➖ Neutral'),
        ('company_name', ''), ('sector', ''), ('weeks', 0),
    ]
    for col, default in _DEFAULTS:
        if col not in filtered_df.columns:
            filtered_df[col] = default

    # ── Helpers ──────────────────────────────────────────────────────────
    def _safe_ret(h: dict, key: str) -> str:
        vals = [v for v in h.get(key, [])
                if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return f"{vals[-1]:+.1f}%" if vals else '—'

    def _last_price(h: dict) -> float:
        return h['prices'][-1] if h.get('prices') else 0

    def _metric(label: str, value: str, color: str = '#e6edf3') -> str:
        return (f'<div style="min-width:44px;">'
                f'<div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.3px;">{label}</div>'
                f'<div style="color:{color};font-weight:700;font-size:0.88rem;margin-top:2px;">{value}</div></div>')

    def _score_bar(pct: float, color: str) -> str:
        w = min(max(pct, 0), 100)
        return (f'<div style="background:#21262d;border-radius:3px;height:4px;margin-top:10px;">'
                f'<div style="width:{w}%;height:4px;border-radius:3px;'
                f'background:linear-gradient(90deg,{color},{color}88);"></div></div>')

    # ── Compute alert data ──────────────────────────────────────────────
    decay_high = int((filtered_df['decay_label'] == 'DECAY_HIGH').sum())
    decay_mod  = int((filtered_df['decay_label'] == 'DECAY_MODERATE').sum())
    conv_mask  = (
        (filtered_df['grade'].isin(['S', 'A'])) &
        (~filtered_df['decay_label'].isin(['DECAY_HIGH', 'DECAY_MODERATE'])) &
        (filtered_df['weeks'] >= 4)
    )
    conv_count = int(conv_mask.sum())
    confirmed  = int((filtered_df['price_label'] == 'PRICE_CONFIRMED').sum())
    divergent  = int((filtered_df['price_label'] == 'PRICE_DIVERGENT').sum())
    _filtered_tickers = set(filtered_df['ticker'].tolist())
    gainers_1w, _ = get_top_movers(histories, n=1, weeks=1, tickers=_filtered_tickers)
    top_delta  = int(gainers_1w.iloc[0]['rank_change']) if not gainers_1w.empty else 0
    trap_total = decay_high + decay_mod

    # ── Header Card ─────────────────────────────────────────────────────
    trap_bg = '#f8514918' if trap_total > 0 else '#21262d'
    trap_fg = '#f85149'   if trap_total > 0 else '#484f58'
    div_bg  = '#d2992218' if divergent > 0 else '#21262d'
    div_fg  = '#d29922'   if divergent > 0 else '#484f58'
    st.markdown(f"""
    <div style="background:#0d1117;border-radius:14px;padding:18px 24px;margin-bottom:16px;border:1px solid #30363d;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div>
          <span style="font-size:1.4rem;font-weight:800;color:#fff;">🚨 Alerts &amp; Signals</span>
          <div style="color:#8b949e;font-size:0.88rem;margin-top:2px;">Real-time warnings · conviction picks · market movers</div>
        </div>
        <div style="display:flex;gap:6px;align-items:center;">
          <span style="background:{trap_bg};color:{trap_fg};padding:4px 10px;border-radius:8px;font-size:0.78rem;font-weight:600;">{trap_total} Traps</span>
          <span style="background:#3fb95018;color:#3fb950;padding:4px 10px;border-radius:8px;font-size:0.78rem;font-weight:600;">{conv_count} Conviction</span>
          <span style="background:{div_bg};color:{div_fg};padding:4px 10px;border-radius:8px;font-size:0.78rem;font-weight:600;">{divergent} Divergent</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Summary Strip ───────────────────────────────────────────────────
    chips = [
        ('🔻 Severe',    str(decay_high),   'traps', '#f85149'),
        ('⚠️ Moderate',  str(decay_mod),    'decay', '#d29922'),
        ('🏆 Conviction', str(conv_count),  'picks', '#3fb950'),
        ('💰 Confirmed', str(confirmed),    'price', '#3fb950'),
        ('📉 Divergent', str(divergent),    'price', '#f85149' if divergent > 0 else '#484f58'),
        ('🔥 Top Move',  f'+{top_delta}',   'ranks', '#58a6ff'),
    ]
    chip_html = ''.join(
        f'<div class="m-chip">'
        f'<div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;">{lbl}</div>'
        f'<div style="font-size:1.2rem;font-weight:800;color:{clr};">{val}</div>'
        f'<div style="font-size:0.68rem;color:#6e7681;">{sub}</div></div>'
        for lbl, val, sub, clr in chips
    )
    st.markdown(f'<div class="m-strip">{chip_html}</div>', unsafe_allow_html=True)
    st.markdown('')

    # ════════════════════════════════════════════════════════════════════
    # § 1  MOMENTUM DECAY TRAPS
    # ════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head">🚨 Momentum Decay Traps</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-cap">Strong trajectory but deteriorating price momentum — rank correction may follow</div>', unsafe_allow_html=True)

    high_traps = filtered_df[
        (filtered_df['decay_label'].isin(['DECAY_HIGH', 'DECAY_MODERATE'])) &
        (filtered_df['trajectory_score'] >= 40)
    ].sort_values('trajectory_score', ascending=False).head(12)

    if high_traps.empty:
        st.markdown(
            '<div style="background:#161b22;border-radius:10px;padding:18px;text-align:center;'
            'border:1px solid #30363d;"><span style="color:#3fb950;font-weight:600;font-size:0.9rem;">'
            '✅ No momentum decay traps detected</span></div>', unsafe_allow_html=True)
    else:
        cards = []
        for _, tr in high_traps.iterrows():
            th = histories.get(tr['ticker'], {})
            price = _last_price(th)
            lr7  = _safe_ret(th, 'ret_7d')
            lr30 = _safe_ret(th, 'ret_30d')
            sev  = tr.get('decay_label', '')
            is_severe = sev == 'DECAY_HIGH'
            sc = '#f85149' if is_severe else '#d29922'
            tag = 'SEVERE' if is_severe else 'MODERATE'
            dm  = tr.get('decay_multiplier', 1.0)
            score = tr['trajectory_score']

            cards.append(
                f'<div style="background:rgba(248,81,73,0.03);border:1px solid rgba(248,81,73,'
                f'{"0.45" if is_severe else "0.25"});border-radius:12px;padding:14px;'
                f'border-left:3px solid {sc};">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><span style="font-weight:700;font-size:0.95rem;color:#e6edf3;">{tr["ticker"]}</span>'
                f'<span style="color:#8b949e;font-size:0.78rem;margin-left:8px;">{str(tr.get("company_name",""))[:22]}</span></div>'
                f'<span style="background:{sc}18;color:{sc};padding:2px 8px;border-radius:8px;'
                f'font-size:0.72rem;font-weight:700;">{tag}</span></div>'
                f'<div style="display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;">'
                f'{_metric("Rank", f"#{int(tr["current_rank"])}")}'
                f'{_metric("Score", f"{score:.0f}", "#FF6B35")}'
                f'{_metric("7d", lr7, sc)}'
                f'{_metric("30d", lr30, sc)}'
                f'{_metric("Decay", f"x{dm:.3f}", sc)}'
                f'{_metric("Price", f"₹{price:,.0f}")}'
                f'</div>{_score_bar(score, sc)}</div>'
            )

        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">'
            f'{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # § 2  CONVICTION PICKS
    # ════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head">🏆 Conviction Picks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-cap">Grade S/A · Clean momentum · 4+ weeks · Price-confirmed preferred</div>', unsafe_allow_html=True)

    conv_df = filtered_df[
        conv_mask & (filtered_df['price_label'] == 'PRICE_CONFIRMED')
    ].sort_values('trajectory_score', ascending=False).head(8)
    if len(conv_df) < 3:
        conv_df = filtered_df[conv_mask].sort_values('trajectory_score', ascending=False).head(8)

    if conv_df.empty:
        st.markdown(
            '<div style="background:#161b22;border-radius:10px;padding:18px;text-align:center;'
            'border:1px solid #30363d;"><span style="color:#8b949e;font-size:0.9rem;">'
            'No conviction picks with current filters</span></div>', unsafe_allow_html=True)
    else:
        cards = []
        for _, cr in conv_df.iterrows():
            gc = GRADE_COLORS.get(cr['grade'], '#8b949e')
            ch = histories.get(cr['ticker'], {})
            price = _last_price(ch)
            lr7  = _safe_ret(ch, 'ret_7d')
            p_key = cr.get('pattern_key', 'neutral')
            p_emoji, p_name, _ = PATTERN_DEFS.get(p_key, ('➖', 'Neutral', ''))
            p_color = PATTERN_COLORS.get(p_key, '#8b949e')
            score = cr['trajectory_score']
            wks = int(cr.get('weeks', 0))

            pills = ''
            if cr.get('sector_alpha_tag') == 'SECTOR_LEADER':
                pills += '<span class="pill p-gld">👑 Leader</span> '
            elif cr.get('sector_alpha_tag') == 'SECTOR_OUTPERFORM':
                pills += '<span class="pill p-grn">Outperform</span> '
            if cr.get('price_label') == 'PRICE_CONFIRMED':
                pills += '<span class="pill p-grn">💰 Confirmed</span> '

            cards.append(
                f'<div style="background:rgba(63,185,80,0.03);border:1px solid rgba(63,185,80,0.35);'
                f'border-radius:12px;padding:14px;border-left:3px solid #3fb950;">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
                f'<div><span style="font-weight:700;font-size:0.95rem;color:#e6edf3;">{cr["ticker"]}</span>'
                f'<div style="color:#8b949e;font-size:0.78rem;margin-top:1px;">{str(cr.get("company_name",""))[:24]}</div></div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:1.3rem;font-weight:800;color:{gc};">{score:.0f}</div>'
                f'<div style="font-size:0.72rem;color:#8b949e;">{cr.get("grade_emoji","")} {cr["grade"]}</div></div></div>'
                f'<div style="display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;">'
                f'{_metric("Rank", f"#{int(cr["current_rank"])}")}'
                f'{_metric("Pattern", f"{p_emoji} {p_name[:12]}", p_color)}'
                f'{_metric("7d", lr7, "#3fb950")}'
                f'{_metric("Price", f"₹{price:,.0f}")}'
                f'{_metric("Weeks", str(wks), "#58a6ff")}'
                f'</div>{_score_bar(score, "#3fb950")}'
                f'<div style="margin-top:6px;">{pills}</div></div>'
            )

        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;">'
            f'{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # § 3  PRICE DIVERGENT
    # ════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head">📉 Price Divergent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-cap">Trajectory rising but price returns negative — potential rank correction ahead</div>', unsafe_allow_html=True)

    div_stocks = filtered_df[
        filtered_df['price_label'] == 'PRICE_DIVERGENT'
    ].sort_values('trajectory_score', ascending=False).head(12)

    if div_stocks.empty:
        st.markdown(
            '<div style="background:#161b22;border-radius:10px;padding:18px;text-align:center;'
            'border:1px solid #30363d;"><span style="color:#3fb950;font-weight:600;font-size:0.9rem;">'
            '✅ No price-divergent stocks detected</span></div>', unsafe_allow_html=True)
    else:
        cards = []
        for _, dv in div_stocks.iterrows():
            gc = GRADE_COLORS.get(dv['grade'], '#8b949e')
            dh = histories.get(dv['ticker'], {})
            price = _last_price(dh)
            lr7  = _safe_ret(dh, 'ret_7d')
            lr30 = _safe_ret(dh, 'ret_30d')
            score = dv['trajectory_score']

            cards.append(
                f'<div style="background:rgba(210,153,34,0.03);border:1px solid rgba(210,153,34,0.30);'
                f'border-radius:12px;padding:14px;border-left:3px solid #d29922;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><span style="font-weight:700;font-size:0.95rem;color:#e6edf3;">{dv["ticker"]}</span>'
                f'<span style="color:#8b949e;font-size:0.78rem;margin-left:8px;">{str(dv.get("company_name",""))[:22]}</span></div>'
                f'<span style="color:{gc};font-weight:700;font-size:0.95rem;">{dv.get("grade_emoji","")} {score:.0f}</span></div>'
                f'<div style="display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;">'
                f'{_metric("Rank", f"#{int(dv["current_rank"])}")}'
                f'{_metric("7d", lr7, "#d29922")}'
                f'{_metric("30d", lr30, "#d29922")}'
                f'{_metric("Price", f"₹{price:,.0f}")}'
                f'<div style="min-width:44px;"><div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.3px;">Status</div>'
                f'<div style="margin-top:2px;"><span class="pill p-red" style="font-size:0.72rem;">Divergent</span></div></div>'
                f'</div>{_score_bar(score, "#d29922")}</div>'
            )

        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">'
            f'{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # § 4  TOP MOVERS — 50 per side, multi-week filter
    # ════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head">🔥 Top Movers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-cap">Biggest rank changes — filter by time window</div>', unsafe_allow_html=True)

    # -- Determine max available weeks from histories --
    max_hist_len = max((len(h['ranks']) for h in histories.values()), default=2) - 1
    week_options = [w for w in [1, 2, 4, 8, 12] if w <= max_hist_len]
    if not week_options:
        week_options = [1]
    week_labels = {1: '1 Week', 2: '2 Weeks', 4: '4 Weeks', 8: '8 Weeks', 12: '12 Weeks'}

    sel_col, info_col = st.columns([1, 3])
    with sel_col:
        mv_weeks = st.selectbox(
            'Time Window',
            options=week_options,
            format_func=lambda x: week_labels.get(x, f'{x} Weeks'),
            index=0,
            key='alert_mover_weeks',
        )
    with info_col:
        st.markdown(f"""
        <div style="background:#161b22;border-radius:10px;padding:10px 16px;margin-top:6px;border:1px solid #30363d;">
            <span style="color:#8b949e;font-size:0.82rem;">Showing rank change over </span>
            <span style="color:#58a6ff;font-weight:700;font-size:0.88rem;">{week_labels.get(mv_weeks, f"{mv_weeks}w")}</span>
            <span style="color:#8b949e;font-size:0.82rem;"> · Top 50 climbers &amp; 50 decliners</span>
        </div>""", unsafe_allow_html=True)

    gainers, decliners = get_top_movers(histories, n=50, weeks=mv_weeks, tickers=_filtered_tickers)

    def _mover_table_html(df_mv: pd.DataFrame, accent: str, icon: str, label: str) -> str:
        """Build one mover panel as a single HTML string — fully styled."""
        count = len(df_mv)
        hdr = (f'<div style="background:#161b22;border-radius:10px 10px 0 0;padding:12px 16px;'
               f'border:1px solid #30363d;border-bottom:2px solid {accent};display:flex;'
               f'justify-content:space-between;align-items:center;">'
               f'<span style="font-size:0.88rem;font-weight:700;color:{accent};">{icon} {label}</span>'
               f'<span style="color:#6e7681;font-size:0.78rem;">{count} stocks</span></div>')

        if df_mv.empty:
            return (hdr + '<div style="background:#0d1117;border-radius:0 0 10px 10px;padding:18px;'
                    'border:1px solid #30363d;border-top:0;text-align:center;color:#6e7681;font-size:0.85rem;'
                    '">No movers detected</div>')

        enriched = df_mv.merge(
            filtered_df[['ticker', 'trajectory_score', 'grade']].drop_duplicates('ticker'),
            on='ticker', how='left')

        # Column header row
        col_hdr = (
            '<div style="display:flex;align-items:center;padding:6px 14px;gap:8px;'
            'background:#161b22;border-bottom:1px solid #30363d;font-size:0.72rem;color:#6e7681;'
            'text-transform:uppercase;letter-spacing:0.5px;">'
            '<span style="min-width:44px;text-align:right;">Chg</span>'
            '<span style="flex:1;">Stock</span>'
            '<span style="min-width:80px;text-align:center;">Prev → Now</span>'
            '<span style="min-width:36px;text-align:center;">Grd</span>'
            '<span style="min-width:36px;text-align:right;">Score</span></div>'
        )

        rows_html = [col_hdr]
        for i, (_, m) in enumerate(enriched.iterrows()):
            rc = int(m['rank_change'])
            ts = m.get('trajectory_score', 0)
            ts = 0 if pd.isna(ts) else ts
            gr = m.get('grade', '—')
            gr = '—' if pd.isna(gr) else gr
            gc = GRADE_COLORS.get(gr, '#8b949e')
            stripe = 'rgba(22,27,34,0.5)' if i % 2 else 'transparent'
            chg_c = '#3fb950' if rc > 0 else '#f85149'
            chg_sign = '+' if rc > 0 else ''

            rows_html.append(
                f'<div style="display:flex;align-items:center;padding:6px 14px;gap:8px;background:{stripe};'
                f'border-bottom:1px solid #21262d;">'
                f'<span style="color:{chg_c};font-weight:800;font-size:0.88rem;min-width:44px;text-align:right;'
                f'font-variant-numeric:tabular-nums;">{chg_sign}{rc}</span>'
                f'<div style="flex:1;overflow:hidden;white-space:nowrap;">'
                f'<span style="color:#e6edf3;font-weight:600;font-size:0.85rem;">{m["ticker"]}</span>'
                f'<span style="color:#8b949e;font-size:0.75rem;margin-left:6px;">'
                f'{str(m.get("company_name",""))[:20]}</span></div>'
                f'<span style="color:#8b949e;font-size:0.8rem;min-width:80px;text-align:center;'
                f'font-variant-numeric:tabular-nums;">{int(m["prev_rank"])} → {int(m["current_rank"])}</span>'
                f'<span style="color:{gc};font-weight:700;font-size:0.82rem;min-width:36px;text-align:center;">{gr}</span>'
                f'<span style="color:#FF6B35;font-weight:600;font-size:0.82rem;min-width:36px;text-align:right;'
                f'font-variant-numeric:tabular-nums;">{ts:.0f}</span></div>')

        body = (f'<div style="background:#0d1117;border-radius:0 0 10px 10px;border:1px solid #30363d;'
                f'border-top:0;overflow:hidden;max-height:580px;overflow-y:auto;">{"".join(rows_html)}</div>')
        return hdr + body

    g_html = _mover_table_html(gainers,   '#3fb950', '⬆️', 'Biggest Climbers')
    d_html = _mover_table_html(decliners, '#f85149', '⬇️', 'Biggest Decliners')
    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">'
        f'<div>{g_html}</div><div>{d_html}</div></div>', unsafe_allow_html=True)


# ============================================
# UI: EXPORT TAB
# ============================================

def render_export_tab(filtered_df: pd.DataFrame, all_df: pd.DataFrame, histories: dict):
    """Render export options tab"""

    st.markdown("##### 📤 Export Trajectory Data")
    st.markdown("Download trajectory rankings and analysis data in various formats.")

    # Export scope
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        scope = st.radio("Export Scope", ['Filtered Rankings (as shown)', 'All Rankings', 'Custom Selection'],
                          key='exp_scope')
    with exp_c2:
        detail = st.radio("Detail Level", ['Compact', 'Standard', 'Full Detail'],
                           key='exp_detail')

    # Determine data to export
    if scope == 'Filtered Rankings (as shown)':
        export_df = filtered_df.copy()
    elif scope == 'All Rankings':
        export_df = all_df.copy()
    else:
        # Custom selection
        available = sorted(all_df['ticker'].tolist())
        selected = st.multiselect("Select tickers to export", available, key='exp_tickers')
        if selected:
            export_df = all_df[all_df['ticker'].isin(selected)].copy()
        else:
            st.info("Select tickers above to export")
            return

    # Column selection based on detail level
    compact_cols = ['t_rank', 'ticker', 'company_name', 'category', 'trajectory_score',
                    'grade', 'pattern', 'tmi', 'positional', 'current_rank', 'best_rank', 'rank_change', 'weeks']
    standard_cols = compact_cols + ['trend', 'velocity', 'acceleration', 'consistency',
                                     'resilience', 'sector', 'industry', 'streak',
                                     'last_week_change', 'avg_rank', 'rank_volatility']
    full_cols = standard_cols + ['worst_rank', 'market_state', 'latest_patterns',
                                  'grade_emoji', 'pattern_key',
                                  'price_alignment', 'price_multiplier', 'price_label',
                                  'decay_score', 'decay_multiplier', 'decay_label',
                                  'sector_alpha_tag', 'sector_alpha_value', 'signal_tags']

    if detail == 'Compact':
        cols = [c for c in compact_cols if c in export_df.columns]
    elif detail == 'Standard':
        cols = [c for c in standard_cols if c in export_df.columns]
    else:
        cols = [c for c in full_cols if c in export_df.columns]

    export_data = export_df[cols].copy()
    # Remove sparkline from export (it's a list column)
    if 'sparkline' in export_data.columns:
        export_data = export_data.drop(columns=['sparkline'])

    # Preview
    st.markdown(f"##### Preview ({len(export_data)} stocks, {len(cols)} columns)")
    st.dataframe(export_data.head(20), hide_index=True, use_container_width=True)

    # Download buttons
    dl_c1, dl_c2, dl_c3 = st.columns(3)

    with dl_c1:
        csv_data = export_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"trajectory_rankings_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv'
        )

    with dl_c2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_data.to_excel(writer, index=False, sheet_name='Trajectory Rankings')
        buffer.seek(0)
        st.download_button(
            label="📥 Download Excel",
            data=buffer.getvalue(),
            file_name=f"trajectory_rankings_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    with dl_c3:
        json_data = export_data.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"trajectory_rankings_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime='application/json'
        )

    # ── Export individual stock trajectory ──
    st.markdown("---")
    st.markdown("##### 📈 Export Individual Stock History")
    ticker_for_export = st.selectbox("Select ticker", sorted(histories.keys()), key='exp_single')
    if ticker_for_export and ticker_for_export in histories:
        h = histories[ticker_for_export]
        hist_df = pd.DataFrame({
            'date': h['dates'],
            'rank': [int(r) for r in h['ranks']],
            'master_score': [round(s, 2) for s in h['scores']],
            'price': [round(p, 2) for p in h['prices']],
            'total_stocks': h['total_per_week'],
            'percentile': [round((1 - r / max(t, 1)) * 100, 2) for r, t in zip(h['ranks'], h['total_per_week'])]
        })
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
        csv_hist = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {ticker_for_export} History (CSV)",
            data=csv_hist,
            file_name=f"{ticker_for_export}_trajectory_history.csv",
            mime='text/csv'
        )


# ============================================
# UI: ABOUT TAB
# ============================================

def render_about_tab():
    """Render about/documentation tab"""

    st.markdown("""
    ## 📊 Rank Trajectory Engine v2.3 — Return-Based Intelligence + Momentum Decay + Sector Alpha

    The **ALL TIME BEST** stock rank trajectory analysis system with **adaptive weight intelligence**,
    **return-based alignment**, **momentum decay warning**, and **sector alpha detection** — the
    cleanest, smartest, most comprehensive scoring engine for the Wave Detection ecosystem.

    ---

    ### 🧠 The Architecture: 5-Layer Pipeline

    ```
    6-Component Adaptive Scoring → Elite Dominance Bonus → Return-Based Alignment
        → Momentum Decay Penalty → Sector Alpha Tag
    ```

    | Layer | What It Does | Impact |
    |---|---|---|
    | **Adaptive Weights** | Weight profile shifts by position tier | Elite: Position=45%. Bottom: Velocity=25% |
    | **Elite Bonus** | Sustained top-tier → guaranteed score floor | Top 3% for 60% weeks → floor 88 |
    | **Return Alignment** | CSV ret_7d/ret_30d confirms or flags trajectory | ×0.92 (divergent) to ×1.08 (confirmed) |
    | **Momentum Decay** | Catches stocks with good rank but negative returns | ×0.93 (severe) to ×1.0 (clean) |
    | **Sector Alpha** | Separates leaders from sector-beta riders | Tag: LEADER / BETA / LAGGARD |

    ---

    ### 💰 Return-Based Price Alignment (v2.3 UPGRADE)

    **v2.2 Problem:** Raw price alignment required complex split detection logic.
    MCX showed ₹11,050→₹2,219 (5:1 split) — needed manual detection.

    **v2.3 Solution:** Uses `ret_7d` and `ret_30d` from CSV — **already split-adjusted
    by the data provider**. No split detection needed. Cleaner, more accurate.

    #### How It Works

    | Signal | Weight | What It Measures |
    |--------|--------|------------------|
    | **Return-Rank Direction** | 55% | Does ret_7d sign match rank percentile change? |
    | **Return Quality** | 45% | Are recent ret_30d values positive for high-ranked stocks? |

    #### Multiplier Range

    | Alignment Score | Multiplier | Label | Meaning |
    |----------------|-----------|-------|---------|
    | **72-100** | ×1.03 — ×1.08 | 💰 CONFIRMED | Returns validate rank trajectory |
    | **50-72** | ×1.00 — ×1.03 | NEUTRAL | Inconclusive |
    | **35-50** | ×0.97 — ×1.00 | NEUTRAL | Mild concern |
    | **0-35** | ×0.92 — ×0.97 | ⚠️ DIVERGENT | Returns contradict rank |

    ---

    ### 🔻 Momentum Decay Warning (v2.3 NEW)

    **The Problem:** Deep audit found **11.4% of top-10% stocks have negative 30-day returns**.
    These are TRAP stocks — ranked well based on PAST momentum that has now faded.
    The rank hasn't dropped yet because ranking lags reality.

    **Example:** Stock at rank 151 (top 7%) with ret_30d = -22.49% → **DECAY_HIGH** (×0.93 penalty)

    #### 4 Decay Signals

    | Signal | What It Checks | Max Points |
    |--------|---------------|------------|
    | **Weekly Return** | ret_7d < -5% → 30 pts, < -2% → 15 pts | 30 |
    | **30-Day Return** | Top stock + ret_30d < -15% → 40 pts (THE TRAP!) | 40 |
    | **From High** | from_high_pct < -20% on ranked stock → 20 pts | 20 |
    | **Consecutive Negative** | 3+ weeks of ret_7d < -1% → 15 pts | 15 |

    #### Decay Penalty

    | Decay Score | Multiplier | Label |
    |------------|-----------|-------|
    | **≥ 60** | ×0.93 | 🔻 DECAY_HIGH |
    | **35-59** | ×0.96 | ⚡ DECAY_MODERATE |
    | **15-34** | ×0.98 | ~ DECAY_MILD |
    | **< 15** | ×1.00 | ✅ CLEAN |

    ---

    ### 🏛️ Sector Alpha Check (v2.3 NEW)

    **The Problem:** 46% of top-50 stocks came from just 2 sectors (Capital Goods 26% + Metals 20%).
    When a sector rotates out, ALL these stocks drop together. A stock with score 75 in a sector
    averaging 72 is just riding the sector wave — NOT genuine alpha.

    #### How It Works
    - Compares each stock's trajectory score to its sector's mean and standard deviation
    - Calculates Z-score: `(stock_score - sector_mean) / sector_std`

    | Z-Score | Classification | Icon | Meaning |
    |---------|---------------|------|---------|
    | **> 1.5** | SECTOR_LEADER | 👑 | Genuine alpha — outperforms sector significantly |
    | **0.5 - 1.5** | SECTOR_OUTPERFORM | ⬆️ | Above sector average |
    | **-0.5 - 0.5** | SECTOR_ALIGNED | ➖ | Moving with sector |
    | **-1.0 - -0.5** (hot sector) | SECTOR_BETA | 🏷️ | Riding sector wave, not genuine alpha |
    | **< -1.0** | SECTOR_LAGGARD | 📉 | Below sector average |

    ---

    ### 🏗️ Adaptive Weight System

    Weights **dynamically shift** based on the stock's average percentile:

    | Tier | Avg Percentile | Positional | Trend | Velocity | Accel | Consistency | Resilience |
    |------|---------------|------------|-------|----------|-------|-------------|------------|
    | **Elite** | > 90% | **45%** | 12% | 8% | 5% | 18% | 12% |
    | **Strong** | 70-90% | 32% | 18% | 12% | 8% | 16% | 14% |
    | **Mid** | 40-70% | 18% | 22% | **20%** | 12% | 14% | 14% |
    | **Bottom** | < 40% | 10% | 20% | **25%** | **18%** | 12% | 15% |

    *Smooth interpolation between tiers — no hard cutoffs.*

    ---

    ### 🛡️ Elite Dominance Bonus

    | Tier | Percentile | Required Duration | Score Floor |
    |------|-----------|------------------|-------------|
    | Top 3% | > 97th | ≥ 60% of weeks | **88** |
    | Top 5% | > 95th | ≥ 60% of weeks | **82** |
    | Top 10% | > 90th | ≥ 55% of weeks | **75** |
    | Top 20% | > 80th | ≥ 50% of weeks | **68** |

    ---

    ### 🎯 3-Stage Selection Funnel

    #### Stage 1: Discovery
    - **Filter:** Trajectory Score ≥ 70 **OR** Rocket/Breakout pattern
    - **Output:** ~50-100 candidates

    #### Stage 2: Validation (5 Rules, must pass 4/5)
    | # | Rule | Threshold | Why |
    |---|------|-----------|-----|
    | 1 | Trend Quality (TQ) | ≥ 60 | Confirms Wave Detection quality |
    | 2 | Market State | ≠ DOWNTREND | 10.1x higher loser ratio! |
    | 3 | Master Score | ≥ 50 | Minimum quality floor |
    | 4 | Recent Rank Δ | ≥ -20 | Not in freefall |
    | 5 | Volume Pattern | VOL / LIQUID / INSTITUTIONAL | Volume confirms conviction |

    #### Stage 3: Final Filter (ALL must pass)
    - TQ ≥ 70 | Leader Pattern required | No DOWNTREND in last 4 weeks
    - **Output:** ~5-10 FINAL BUYS

    ---

    ### 📊 TMI (Trajectory Momentum Index)

    `TMI = 100 - (100 / (1 + RS))` where `RS = Avg Rank Gain / Avg Rank Loss`

    | TMI Range | Interpretation |
    |-----------|----------------|
    | **70-100** | Strong momentum — rank consistently improving |
    | **50-70** | Moderate momentum |
    | **30-50** | Weak momentum — mixed signals |
    | **0-30** | Deteriorating — rank consistently worsening |

    ---

    ### 🏷️ Trajectory Patterns
    """)

    # Pattern table
    pattern_rows = []
    for key, (emoji, name, desc) in PATTERN_DEFS.items():
        pattern_rows.append({'Pattern': f"{emoji} {name}", 'Description': desc})
    st.table(pd.DataFrame(pattern_rows))

    st.markdown("""
    ---

    ### 📡 Signal Tags (v2.3)

    The **Signals** column in rankings combines multiple indicators:

    | Icon | Signal | Meaning |
    |------|--------|---------|
    | 💰 | Price Confirmed | Returns validate rank trajectory |
    | ⚠️ | Price Divergent | Returns contradict rank |
    | 🔻 | Decay High | Good rank but severely negative returns — TRAP! |
    | ⚡ | Decay Moderate | Moderate momentum decay warning |
    | 👑 | Sector Leader | Genuine alpha — significantly above sector average |
    | 🏷️ | Sector Beta | Riding hot sector, not genuine alpha |
    | 📉 | Sector Laggard | Below sector average |

    ---

    ### 🎓 Grades

    | Grade | Score Range | Meaning |
    |-------|------------|---------|
    | 🏆 **S** | 85 — 100 | Elite — sustained top position or explosive improvement |
    | 🥇 **A** | 70 — 84 | Excellent — strong position + positive trajectory |
    | 🥈 **B** | 55 — 69 | Good — above average, potential emerging |
    | 🥉 **C** | 40 — 54 | Average — mixed signals, watch list |
    | 📊 **D** | 25 — 39 | Below average — weak or deteriorating |
    | 📉 **F** | 0 — 24 | Poor — declining or insufficient data |

    ---

    ### ⚙️ Technical Details

    - **Adaptive Weights**: Smooth interpolation between 4 tier profiles (elite/strong/mid/bottom)
    - **Positional Quality**: Sigmoid-boosted percentile — non-linear scaling for top positions
    - **Elite Trend Floor**: Top 5% → floor 70, Top 10% → 65, Top 20% → 58
    - **Position-Relative Velocity**: Hold bonus (15 for top 5%), dampened dip sensitivity
    - **Elite Consistency**: Band-based (40%) + time-at-top (35%) + low-vol bonus (25%)
    - **Elite Dominance Bonus**: Sustained top-tier → guaranteed score floor (82-88)
    - **Return-Based Alignment**: Uses ret_7d/ret_30d from CSV (split-adjusted by provider)
    - **Momentum Decay**: 4-signal trap detection (ret_7d, ret_30d, from_high, consecutive)
    - **Sector Alpha**: Z-score based sector-relative performance classification
    - **Recency Weighting**: Exponential decay (λ=0.12) for trend regression

    ---

    *Built for the Wave Detection ecosystem • v2.3.0-ULTIMATE • February 2026*
    """)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point"""

    # ── Sidebar: Upload CSVs ──
    with st.sidebar:
        st.markdown("### 📊 Rank Trajectory Engine")
        uploaded_files = st.file_uploader(
            "📂 Upload Weekly CSV Snapshots",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload your Wave Detection weekly CSV exports (Stocks_Weekly_YYYY-MM-DD_*.csv)"
        )

    # Header
    st.markdown('<div class="main-header">📊 RANK TRAJECTORY ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Stock Rank Trajectory Analysis • Multi-Week Momentum Intelligence</div>',
                unsafe_allow_html=True)

    if not uploaded_files:
        st.info("👈 Upload your weekly CSV snapshots from the sidebar to begin trajectory analysis")
        st.markdown("""
        **How to use:**
        1. Open the **sidebar** (click `>` at the top-left if collapsed)
        2. Click **Browse files** or drag-and-drop your Wave Detection weekly CSV exports
        3. Upload multiple weeks at once (select all CSVs together)
        4. Files should be named: `Stocks_Weekly_YYYY-MM-DD_Month_Year.csv`
        5. Minimum **3 weeks** recommended for meaningful trajectory analysis
        """)
        return

    st.caption(f"📁 {len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''} uploaded")

    # ── Session-state caching (recompute only when files change) ──
    cache_key = tuple(sorted((f.name, f.size) for f in uploaded_files))
    if st.session_state.get('_traj_key') != cache_key:
        with st.spinner("📊 Computing trajectories across all weeks..."):
            result = load_and_compute(uploaded_files)
        st.session_state['_traj_key'] = cache_key
        st.session_state['_traj_result'] = result

    result = st.session_state['_traj_result']

    if result[0] is None:
        st.error("❌ No valid data found in uploaded files. Ensure CSVs contain `rank` and `ticker` columns.")
        return

    traj_df, histories, dates_iso, metadata = result

    if traj_df.empty:
        st.warning("No stocks found with sufficient data for trajectory analysis.")
        return

    # ── Sidebar ──
    filters = render_sidebar(metadata, traj_df)

    # ── Apply Filters ──
    filtered_df = apply_filters(traj_df, filters)

    # ── Tabs ──
    tab_ranking, tab_search, tab_funnel, tab_alerts, tab_export, tab_about = st.tabs([
        "🏆 Rankings", "🔍 Search & Analyze", "🎯 Funnel", "🚨 Alerts", "📤 Export", "ℹ️ About"
    ])

    with tab_ranking:
        render_rankings_tab(filtered_df, traj_df, histories, metadata)

    with tab_search:
        render_search_tab(filtered_df, traj_df, histories, dates_iso)

    with tab_funnel:
        render_funnel_tab(filtered_df, traj_df, histories, metadata)

    with tab_alerts:
        render_alerts_tab(filtered_df, histories)

    with tab_export:
        render_export_tab(filtered_df, traj_df, histories)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
