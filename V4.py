"""
Rank Trajectory Engine v8.0 — Wave Signal Fusion
=======================================================
Professional Stock Rank Trajectory Analysis System
with Adaptive Weight Intelligence, Return Quality Component, Directional Price-Rank
Alignment, Momentum Decay Warning, Sector Alpha Detection, Market Regime Awareness,
Confidence Intervals, Z-Score Normalization, Conviction Score, Risk-Adjusted T-Score,
Exit Warning System, Hot Streak Detection, Volume Confirmation, Multi-Stage Selection Funnel,
and WAVE SIGNAL FUSION ENGINE — Deep integration of 18 WAVE Detection signals.

CORE ARCHITECTURE:
  7-Component Adaptive Scoring → Elite Dominance Bonus → Bayesian Shrinkage
    → Unified Multiplier (Hurst × Wave Fusion × Price Alignment × Momentum Decay)
    → Sector Alpha Tag

  Wave Signal Fusion: Cross-validates WAVE Detection scores with Trajectory calculations.
    4 Fusion Signals: Confluence (35%) + Institutional Flow (30%) + Momentum Harmony (20%)
    + Fundamental Quality (15%) → Fusion Multiplier ×0.92 to ×1.10

  Components: Positional, Trend, Velocity, Acceleration, Consistency,
              Resilience, ReturnQuality
  Weights shift dynamically by position tier (elite/strong/mid/bottom).

  SIGNAL ISOLATION PRINCIPLE:
    - Return data enters through exactly ONE component (ReturnQuality).
    - Price-Rank Alignment scores DIRECTION only (sign agreement), never magnitude.
    - Momentum Decay uses separate ret_6m for proven-winner exemption.
    - No signal leakage: each data source has exactly one scoring path.

Components: Adaptive Weights by Tier (7 components)
  Elite (>90pct):  Pos 40% | Trd 10% | Vel 7%  | Acc 4%  | Con 16% | Res 10% | Ret 13%
  Strong (70-90):  Pos 28% | Trd 16% | Vel 10% | Acc 7%  | Con 14% | Res 12% | Ret 13%
  Mid (40-70):     Pos 15% | Trd 19% | Vel 17% | Acc 10% | Con 12% | Res 12% | Ret 15%
  Bottom (<40):    Pos 8%  | Trd 17% | Vel 21% | Acc 15% | Con 10% | Res 12% | Ret 17%

3-STAGE FUNNEL:
  Stage 1: Discovery  — Trajectory Score ≥70 or Rocket/Breakout → 50-100 candidates
  Stage 2: Validation — 5 Wave Engine rules, must pass 4/5    → 20-30 stocks
  Stage 3: Final      — TQ≥70, Leader patterns, no DOWNTREND  → 5-10 FINAL BUYS

Version: 8.0.0
Last Updated: March 2026
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

# Adaptive weight profiles by percentile tier (v6.0: 7 components)
ADAPTIVE_WEIGHTS = {
    # Elite (avg pct > 90): Position IS the score. Returns confirm dominance.
    'elite': {
        'positional': 0.40, 'trend': 0.10, 'velocity': 0.07,
        'acceleration': 0.04, 'consistency': 0.16, 'resilience': 0.10,
        'return_quality': 0.13
    },
    # Strong (avg pct 70-90): Balanced — good position + returns should back it up
    'strong': {
        'positional': 0.28, 'trend': 0.16, 'velocity': 0.10,
        'acceleration': 0.07, 'consistency': 0.14, 'resilience': 0.12,
        'return_quality': 0.13
    },
    # Mid (avg pct 40-70): Movement + returns — prove trajectory with gains
    'mid': {
        'positional': 0.15, 'trend': 0.19, 'velocity': 0.17,
        'acceleration': 0.10, 'consistency': 0.12, 'resilience': 0.12,
        'return_quality': 0.15
    },
    # Bottom (avg pct < 40): Returns matter most — are gains materializing?
    'bottom': {
        'positional': 0.08, 'trend': 0.17, 'velocity': 0.21,
        'acceleration': 0.15, 'consistency': 0.10, 'resilience': 0.12,
        'return_quality': 0.17
    }
}

# Elite Dominance Bonus thresholds
ELITE_BONUS = {
    'top3_sustained': {'pct_threshold': 97, 'history_ratio': 0.60, 'floor': 88},
    'top5_sustained': {'pct_threshold': 95, 'history_ratio': 0.60, 'floor': 82},
    'top10_sustained': {'pct_threshold': 90, 'history_ratio': 0.60, 'floor': 73},
    'top20_sustained': {'pct_threshold': 85, 'history_ratio': 0.55, 'floor': 65}
}

# Directional Price-Rank Alignment Configuration (v6.1 — direction-only, no magnitude)
PRICE_ALIGNMENT = {
    'noise_band_stable': 2.0,        # Ignore rank moves < this for stable stocks
    'noise_band_normal': 1.0,        # Ignore rank moves < this for normal stocks
    'min_weeks': 4,                  # Minimum weeks needed for alignment calculation
    'multiplier_max_boost': 1.08,    # Maximum upward multiplier (v5.3: reduced from 1.12)
    'multiplier_max_penalty': 0.88,  # Maximum downward multiplier (v5.3: tightened from 0.85)
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

# ── Return Quality Component Configuration (v6.0) ──
# ARCHITECTURE CHANGE: Returns are now a dedicated 7th scoring component.
# Previously returns leaked through 5 doors (velocity floor, acceleration floor,
# resilience bonus, conviction multiplier, price alignment signal 3).
# Now one clean component: _calc_return_quality() → 0 to 100.
# 4 sub-signals: 3M return (30%), 6M return (30%), short-term momentum (20%),
# return health / cross-timeframe agreement (20%).
RETURN_QUALITY = {
    'min_weeks': 3,                  # Minimum data weeks for scoring
}

# Sector Alpha Configuration (v2.3)
SECTOR_ALPHA = {
    'min_sector_stocks': 3,          # Minimum stocks in sector for alpha calc
    'leader_z': 1.5,                 # Z-score above this = sector leader
    'outperform_z': 0.5,             # Z-score above this = outperforming sector
    'aligned_z': -0.5,              # Z-score above this = aligned with sector
    'beta_sector_min': 60,           # Sector avg must be > this for beta detection
}

# ── Wave Signal Fusion Configuration (v8.0) ──
# Cross-validates WAVE Detection's 18 ignored signals with Trajectory Engine calculations.
# When BOTH systems agree a stock is strong → high conviction.
# When they disagree → uncertainty flag.
WAVE_FUSION = {
    # Fusion signal weights (must sum to 1.0)
    'confluence_weight': 0.35,           # Agreement between WAVE scoring and Trajectory
    'flow_weight': 0.30,                 # Institutional money flow strength
    'harmony_weight': 0.20,              # WAVE's momentum harmony (0-4)
    'fundamental_weight': 0.15,          # EPS/PE quality gate

    # Multiplier range
    'max_boost': 1.10,                   # Maximum fusion boost (strong agreement)
    'max_penalty': 0.92,                 # Maximum fusion penalty (strong conflict)
    'neutral_zone_lo': 42,               # Below this → penalty
    'neutral_zone_hi': 58,               # Above this → boost

    # Confluence thresholds (WAVE vs Trajectory agreement)
    'strong_agree_pct_diff': 10,         # Percentile difference for "strong agreement"
    'agree_pct_diff': 25,                # Percentile difference for "agreement"
    'disagree_pct_diff': 45,             # Percentile difference for "disagreement"

    # Institutional flow scoring
    'money_flow_strong': 50.0,           # money_flow_mm above this = strong flow
    'money_flow_moderate': 10.0,         # money_flow_mm above this = moderate flow
    'vmi_strong': 65,                    # VMI above this = strong volume-momentum
    'vmi_moderate': 40,                  # VMI above this = moderate
    'rvol_hot': 2.0,                     # RVOL above this = hot volume
    'rvol_active': 1.2,                  # RVOL above this = active volume

    # Fundamental quality thresholds
    'eps_growth_strong': 20.0,           # EPS change % for "strong growth"
    'eps_growth_moderate': 5.0,          # EPS change % for "moderate growth"
    'pe_reasonable_max': 50.0,           # PE ratio ceiling for "reasonable"
    'pe_value_max': 20.0,                # PE ratio ceiling for "value"
    'pe_negative_penalty': True,         # Penalize negative PE (loss-making)
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
    'S': '#FFD700', 'A': '#3fb950', 'B': '#58a6ff',
    'C': '#d29922', 'D': '#FF5722', 'F': '#f85149'
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
                    'ret_6m': [],             # v3.0: 6-month returns (institutional horizon)
                    'from_high_pct': [],      # v2.3: Distance from 52w high
                    'momentum_score': [],     # v2.3: Wave engine momentum score
                    'volume_score': [],       # v2.3: Wave engine volume score
                    # ── v8.0: Wave Signal Fusion — 18 previously ignored columns ──
                    'position_score': [],     # WAVE's 52-week position scoring
                    'acceleration_score': [], # WAVE's momentum acceleration
                    'breakout_score': [],     # WAVE's breakout detection (4 components)
                    'rvol_score': [],         # WAVE's relative volume quality
                    'pe': [],                 # Price/Earnings ratio
                    'eps_current': [],        # Current EPS
                    'eps_change_pct': [],     # EPS change percentage
                    'from_low_pct': [],       # Distance from 52-week low
                    'ret_1d': [],             # 1-day return (intraday momentum)
                    'ret_1y': [],             # 1-year return (long-term trend)
                    'rvol': [],               # Raw relative volume
                    'vmi': [],                # Volume-Momentum Index
                    'money_flow_mm': [],      # Money flow in millions
                    'position_tension': [],   # Position tension metric
                    'momentum_harmony': [],   # Momentum harmony (0-4)
                    'eps_tier': [],           # EPS tier classification
                    'pe_tier': [],            # PE tier classification
                    'overall_market_strength': [],  # Market strength metric
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
                ('ret_6m', 'ret_6m'),
                ('from_high_pct', 'from_high_pct'), ('momentum_score', 'momentum_score'),
                ('volume_score', 'volume_score'),
                # v8.0: Wave Signal Fusion — 18 previously ignored columns
                ('position_score', 'position_score'), ('acceleration_score', 'acceleration_score'),
                ('breakout_score', 'breakout_score'), ('rvol_score', 'rvol_score'),
                ('pe', 'pe'), ('eps_current', 'eps_current'),
                ('eps_change_pct', 'eps_change_pct'), ('from_low_pct', 'from_low_pct'),
                ('ret_1d', 'ret_1d'), ('ret_1y', 'ret_1y'),
                ('rvol', 'rvol'), ('vmi', 'vmi'),
                ('money_flow_mm', 'money_flow_mm'), ('position_tension', 'position_tension'),
                ('momentum_harmony', 'momentum_harmony'),
                ('overall_market_strength', 'overall_market_strength'),
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

            # v8.0: Track tier classifications (string columns)
            for tier_col in ['eps_tier', 'pe_tier']:
                tier_val = row.get(tier_col, '')
                h[tier_col].append(str(tier_val).strip() if pd.notna(tier_val) else '')

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

    # Build DataFrame and sort (v6.2: Confidence-Aware Ranking with Tie-Breakers)
    # Primary: trajectory_score DESC
    # Tie-breaker 1: confidence DESC (proven performers beat lucky streaks)
    # Tie-breaker 2: consistency DESC (stable beats volatile at equal score+conf)
    traj_df = pd.DataFrame(results)
    traj_df = traj_df.sort_values(
        ['trajectory_score', 'confidence', 'consistency'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    traj_df.insert(0, 't_rank', range(1, len(traj_df) + 1))
    
    # Add T-Percentile: position as percentile of universe (higher = better)
    total_stocks = len(traj_df)
    traj_df['t_percentile'] = ((total_stocks - traj_df['t_rank']) / max(total_stocks - 1, 1) * 100).round(1)

    # ── Step 3b: Market Regime Awareness (v6.2) ──
    # Computes market-wide median trend/velocity to normalize for market moves.
    # In a bear market, a flat stock is relatively STRONG.
    # In a bull market, a flat stock is relatively WEAK.
    market_trend_median = traj_df['trend'].median()
    market_velocity_median = traj_df['velocity'].median()

    # Classify market regime based on median trend
    if market_trend_median > 58:
        market_regime = 'BULL'
        market_adj_factor = 1.0  # No adjustment in bull market (baseline)
    elif market_trend_median < 42:
        market_regime = 'BEAR'
        market_adj_factor = 1.03  # Slight boost in bear market (survival premium)
    else:
        market_regime = 'SIDEWAYS'
        market_adj_factor = 1.0

    traj_df['market_regime'] = market_regime
    traj_df['market_trend_median'] = round(market_trend_median, 1)

    # Compute market-adjusted score: relative performance vs market
    # A stock with trend=60 in BEAR market (median=35) is exceptional
    # A stock with trend=60 in BULL market (median=70) is lagging
    def _market_adj_score(row):
        trend_vs_market = row['trend'] - market_trend_median
        vel_vs_market = row['velocity'] - market_velocity_median
        relative_strength = (trend_vs_market + vel_vs_market) / 2  # Range: -50 to +50
        # Scale to 0-100 centered at 50
        mkt_adj = 50 + relative_strength
        return round(np.clip(mkt_adj, 0, 100), 1)

    traj_df['market_adj_score'] = traj_df.apply(_market_adj_score, axis=1)

    # ── Step 3c: Z-Score Normalization (v6.2) ──
    # Converts raw component scores to cross-sectional z-scores, then rescales
    # to 0-100. This ensures scores reflect relative performance vs population.
    # Benefits: prevents score clustering, improves differentiation, robust to outliers.
    z_components = ['positional', 'trend', 'velocity', 'acceleration',
                    'consistency', 'resilience', 'return_quality']

    for col in z_components:
        col_mean = traj_df[col].mean()
        col_std = max(traj_df[col].std(), 1.0)  # Avoid division by zero
        z_col = (traj_df[col] - col_mean) / col_std
        # Rescale z-score to 0-100: z=-3 → 0, z=0 → 50, z=+3 → 100
        normalized_col = 50 + (z_col * 16.67)  # 50/3 ≈ 16.67
        traj_df[f'{col}_zscore'] = z_col.round(2)
        traj_df[f'{col}_norm'] = normalized_col.clip(0, 100).round(1)

    # Compute composite normalized score (average of normalized components)
    norm_cols = [f'{c}_norm' for c in z_components]
    traj_df['normalized_score'] = traj_df[norm_cols].mean(axis=1).round(2)

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
# WAVE SIGNAL FUSION ENGINE (v8.0)
# ============================================
# Deep integration of 18 previously ignored WAVE Detection signals.
# Cross-validates WAVE's independent scoring with Trajectory Engine calculations.
# 4 Fusion Signals → Fusion Score → Fusion Multiplier (×0.92 to ×1.10)

def _latest_valid(lst, default=None):
    """Get the latest non-NaN value from a history list."""
    if not lst:
        return default
    for v in reversed(lst):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def _avg_recent(lst, window=4, default=None):
    """Average of last `window` valid values."""
    if not lst:
        return default
    valid = [v for v in lst[-window:] if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(valid)) if valid else default


def _calc_wave_confluence(h: dict, traj_components: dict) -> float:
    """
    Wave Confluence Score — Agreement between WAVE Detection and Trajectory Engine.

    CORE INSIGHT: WAVE Detection and Trajectory Engine independently analyze the same stock.
    WAVE scores the stock's CURRENT state using cross-sectional analysis (vs all stocks this week).
    Trajectory scores the stock's TRAJECTORY over time (how ranks change week-over-week).

    When BOTH systems agree a stock is strong → HIGH confidence, genuine quality.
    When they disagree → uncertain, one system may be lagging or wrong.

    4 Cross-Validation Channels:
      1. WAVE position_score vs Trajectory positional (40%)
      2. WAVE acceleration_score vs Trajectory acceleration (25%)
      3. WAVE breakout_score vs Trajectory velocity (20%)
      4. WAVE momentum_score vs Trajectory trend (15%)

    Each channel scores 0-100 based on agreement level.
    """
    cfg = WAVE_FUSION
    channels = []

    # ── Channel 1: Position Agreement (40%) ──
    # WAVE position_score: 52-week range analysis (0-100)
    # Trajectory positional: rank percentile quality over time (0-100)
    wave_pos = _latest_valid(h.get('position_score', []))
    traj_pos = traj_components.get('positional', 50)
    if wave_pos is not None:
        diff = abs(wave_pos - traj_pos)
        if diff <= cfg['strong_agree_pct_diff']:
            ch_score = 85 + (cfg['strong_agree_pct_diff'] - diff) / cfg['strong_agree_pct_diff'] * 15
        elif diff <= cfg['agree_pct_diff']:
            t = (diff - cfg['strong_agree_pct_diff']) / (cfg['agree_pct_diff'] - cfg['strong_agree_pct_diff'])
            ch_score = 60 + (1 - t) * 25
        elif diff <= cfg['disagree_pct_diff']:
            t = (diff - cfg['agree_pct_diff']) / (cfg['disagree_pct_diff'] - cfg['agree_pct_diff'])
            ch_score = 30 + (1 - t) * 30
        else:
            ch_score = max(5, 30 - (diff - cfg['disagree_pct_diff']) * 0.5)
        # Boost if BOTH are high (agreement at the top matters more)
        if wave_pos > 70 and traj_pos > 70:
            ch_score = min(100, ch_score + 8)
        channels.append(('position', ch_score, 0.40))

    # ── Channel 2: Acceleration Agreement (25%) ──
    # WAVE acceleration_score: momentum slope comparison (0-100)
    # Trajectory acceleration: rate-of-change of velocity (0-100)
    wave_accel = _latest_valid(h.get('acceleration_score', []))
    traj_accel = traj_components.get('acceleration', 50)
    if wave_accel is not None:
        diff = abs(wave_accel - traj_accel)
        if diff <= cfg['strong_agree_pct_diff']:
            ch_score = 85 + (cfg['strong_agree_pct_diff'] - diff) / cfg['strong_agree_pct_diff'] * 15
        elif diff <= cfg['agree_pct_diff']:
            t = (diff - cfg['strong_agree_pct_diff']) / (cfg['agree_pct_diff'] - cfg['strong_agree_pct_diff'])
            ch_score = 55 + (1 - t) * 30
        elif diff <= cfg['disagree_pct_diff']:
            t = (diff - cfg['agree_pct_diff']) / (cfg['disagree_pct_diff'] - cfg['agree_pct_diff'])
            ch_score = 25 + (1 - t) * 30
        else:
            ch_score = max(5, 25 - (diff - cfg['disagree_pct_diff']) * 0.5)
        channels.append(('acceleration', ch_score, 0.25))

    # ── Channel 3: Breakout/Velocity Agreement (20%) ──
    # WAVE breakout_score: 4-component breakout detection (0-100)
    # Trajectory velocity: position-relative rank velocity (0-100)
    wave_brk = _latest_valid(h.get('breakout_score', []))
    traj_vel = traj_components.get('velocity', 50)
    if wave_brk is not None:
        # Breakout and velocity aren't 1:1 comparable, but high breakout + high velocity = strong
        # Use harmonic-mean-like scoring: both need to be elevated for high score
        combined = (wave_brk * 0.55 + traj_vel * 0.45)
        if combined >= 75:
            ch_score = 80 + (combined - 75) / 25 * 20
        elif combined >= 55:
            ch_score = 50 + (combined - 55) / 20 * 30
        elif combined >= 35:
            ch_score = 25 + (combined - 35) / 20 * 25
        else:
            ch_score = max(5, combined * 0.7)
        channels.append(('breakout', ch_score, 0.20))

    # ── Channel 4: Momentum/Trend Agreement (15%) ──
    # WAVE momentum_score: 3-component momentum (raw sigmoid + consistency + quality)
    # Trajectory trend: weighted linear regression of percentile trajectory
    wave_mom = _latest_valid(h.get('momentum_score', []))
    traj_trend = traj_components.get('trend', 50)
    if wave_mom is not None:
        diff = abs(wave_mom - traj_trend)
        if diff <= cfg['strong_agree_pct_diff']:
            ch_score = 80 + (cfg['strong_agree_pct_diff'] - diff) / cfg['strong_agree_pct_diff'] * 20
        elif diff <= cfg['agree_pct_diff']:
            t = (diff - cfg['strong_agree_pct_diff']) / (cfg['agree_pct_diff'] - cfg['strong_agree_pct_diff'])
            ch_score = 50 + (1 - t) * 30
        else:
            ch_score = max(5, 50 - (diff - cfg['agree_pct_diff']) * 0.8)
        channels.append(('momentum', ch_score, 0.15))

    # ── Composite Confluence Score ──
    if not channels:
        return 50.0  # No WAVE data available → neutral

    total_w = sum(w for _, _, w in channels)
    confluence = sum(s * w for _, s, w in channels) / total_w
    return float(np.clip(confluence, 0, 100))


def _calc_institutional_flow(h: dict) -> float:
    """
    Institutional Flow Signal from WAVE Detection data.

    Combines 4 volume/flow metrics that WAVE Detection computes but
    RANK TRAJECTORY previously ignored. These signals indicate whether
    rank improvements have institutional money backing them.

    4 Components:
      1. Money Flow MM (40%) — price × volume × rvol / 1M → smart money indicator
      2. VMI (30%) — Volume-Momentum Index (acceleration + correlation + footprint)
      3. RVOL Score (20%) — WAVE's relative volume quality score
      4. Overall Market Strength (10%) — Market environment context

    Returns: 0-100 score (higher = stronger institutional backing)
    """
    cfg = WAVE_FUSION
    components = []

    # ── Component 1: Money Flow (40%) ──
    mf = _latest_valid(h.get('money_flow_mm', []))
    if mf is not None:
        if mf >= cfg['money_flow_strong']:
            mf_score = 75 + min((mf - cfg['money_flow_strong']) / cfg['money_flow_strong'] * 25, 25)
        elif mf >= cfg['money_flow_moderate']:
            t = (mf - cfg['money_flow_moderate']) / (cfg['money_flow_strong'] - cfg['money_flow_moderate'])
            mf_score = 40 + t * 35
        elif mf > 0:
            mf_score = max(10, mf / cfg['money_flow_moderate'] * 40)
        else:
            mf_score = 10.0
        components.append(('money_flow', mf_score, 0.40))

    # ── Component 2: VMI — Volume-Momentum Index (30%) ──
    vmi = _latest_valid(h.get('vmi', []))
    if vmi is not None:
        if vmi >= cfg['vmi_strong']:
            vmi_score = 75 + min((vmi - cfg['vmi_strong']) / 35 * 25, 25)
        elif vmi >= cfg['vmi_moderate']:
            t = (vmi - cfg['vmi_moderate']) / (cfg['vmi_strong'] - cfg['vmi_moderate'])
            vmi_score = 40 + t * 35
        elif vmi > 0:
            vmi_score = max(10, vmi / cfg['vmi_moderate'] * 40)
        else:
            vmi_score = 10.0
        components.append(('vmi', vmi_score, 0.30))

    # ── Component 3: RVOL Score (20%) ──
    rvol_s = _latest_valid(h.get('rvol_score', []))
    if rvol_s is not None:
        # WAVE rvol_score is already 0-100
        components.append(('rvol_score', float(np.clip(rvol_s, 0, 100)), 0.20))

    # ── Component 4: Overall Market Strength (10%) ──
    oms = _latest_valid(h.get('overall_market_strength', []))
    if oms is not None:
        # overall_market_strength is typically 0-100
        components.append(('market_strength', float(np.clip(oms, 0, 100)), 0.10))

    if not components:
        return 50.0  # No data → neutral

    total_w = sum(w for _, _, w in components)
    flow_score = sum(s * w for _, s, w in components) / total_w
    return float(np.clip(flow_score, 0, 100))


def _calc_fundamental_quality(h: dict) -> float:
    """
    Fundamental Quality Gate from WAVE Detection data.

    NOT a primary scoring driver — just a quality confirmation/penalty.
    This system is momentum-based; fundamentals serve as a sanity check.

    3 Components:
      1. EPS Growth (40%) — eps_change_pct trend: growing earnings = real strength
      2. PE Reasonableness (30%) — not too high, not negative
      3. EPS Tier (30%) — WAVE's tier classification quality

    Returns: 0-100 score (50 = neutral, >60 = quality confirmed, <40 = fundamental concern)
    """
    cfg = WAVE_FUSION
    components = []

    # ── Component 1: EPS Growth (40%) ──
    eps_chg = _latest_valid(h.get('eps_change_pct', []))
    if eps_chg is not None:
        if eps_chg >= cfg['eps_growth_strong']:
            eps_score = 80 + min((eps_chg - cfg['eps_growth_strong']) / 30 * 20, 20)
        elif eps_chg >= cfg['eps_growth_moderate']:
            t = (eps_chg - cfg['eps_growth_moderate']) / (cfg['eps_growth_strong'] - cfg['eps_growth_moderate'])
            eps_score = 55 + t * 25
        elif eps_chg >= 0:
            eps_score = 40 + eps_chg / cfg['eps_growth_moderate'] * 15
        elif eps_chg >= -10:
            eps_score = 25 + (eps_chg + 10) / 10 * 15
        else:
            eps_score = max(5, 25 + eps_chg * 0.5)  # Deeply negative
        components.append(('eps_growth', float(np.clip(eps_score, 0, 100)), 0.40))

    # ── Component 2: PE Reasonableness (30%) ──
    pe = _latest_valid(h.get('pe', []))
    if pe is not None:
        if pe < 0 and cfg['pe_negative_penalty']:
            pe_score = 20.0  # Loss-making company
        elif pe == 0:
            pe_score = 30.0  # No earnings data
        elif pe <= cfg['pe_value_max']:
            pe_score = 85.0  # Value territory
        elif pe <= cfg['pe_reasonable_max']:
            t = (pe - cfg['pe_value_max']) / (cfg['pe_reasonable_max'] - cfg['pe_value_max'])
            pe_score = 60 + (1 - t) * 25  # Reasonable
        elif pe <= 100:
            pe_score = 35 + (100 - pe) / 50 * 25  # Getting expensive
        else:
            pe_score = max(10, 35 - (pe - 100) * 0.1)  # Very expensive
        components.append(('pe', float(np.clip(pe_score, 0, 100)), 0.30))

    # ── Component 3: EPS Tier (30%) ──
    eps_tier = h.get('eps_tier', [])
    latest_tier = eps_tier[-1] if eps_tier else ''
    if latest_tier and isinstance(latest_tier, str) and latest_tier.strip():
        tier_map = {
            'STRONG_GROWTH': 90, 'GROWTH': 75, 'MODERATE': 60,
            'STABLE': 55, 'DECLINING': 30, 'NEGATIVE': 15, 'LOSS': 10
        }
        # Fuzzy match — tier names may vary
        tier_upper = latest_tier.upper().strip()
        tier_score = 50.0  # Default
        for key, val in tier_map.items():
            if key in tier_upper:
                tier_score = val
                break
        components.append(('eps_tier', tier_score, 0.30))

    if not components:
        return 50.0  # No fundamental data → neutral (don't penalize)

    total_w = sum(w for _, _, w in components)
    fund_score = sum(s * w for _, s, w in components) / total_w
    return float(np.clip(fund_score, 0, 100))


def _compute_wave_fusion(h: dict, traj_components: dict) -> dict:
    """
    Wave Signal Fusion Engine (v8.0) — Master fusion function.

    Cross-validates WAVE Detection's 18 signals with Trajectory Engine calculations.
    Produces a fusion multiplier applied to the trajectory score.

    Pipeline:
      1. Wave Confluence (35%) — agreement between WAVE and Trajectory scoring
      2. Institutional Flow (30%) — money flow + VMI + RVOL strength
      3. Momentum Harmony (20%) — WAVE's 5-check harmony score (0-4)
      4. Fundamental Quality (15%) — EPS growth + PE reasonableness

    Returns dict with all fusion metrics.
    """
    cfg = WAVE_FUSION

    # ── Signal 1: Wave Confluence (35%) ──
    confluence = _calc_wave_confluence(h, traj_components)

    # ── Signal 2: Institutional Flow (30%) ──
    inst_flow = _calc_institutional_flow(h)

    # ── Signal 3: Momentum Harmony (20%) ──
    harmony_raw = _latest_valid(h.get('momentum_harmony', []), 2.0)
    # WAVE momentum_harmony is 0-4 (0=full disagreement, 4=full harmony)
    harmony_score = float(np.clip(harmony_raw / 4.0 * 100, 0, 100))

    # ── Signal 4: Fundamental Quality (15%) ──
    fund_quality = _calc_fundamental_quality(h)

    # ── Composite Fusion Score ──
    fusion_score = (
        cfg['confluence_weight'] * confluence +
        cfg['flow_weight'] * inst_flow +
        cfg['harmony_weight'] * harmony_score +
        cfg['fundamental_weight'] * fund_quality
    )
    fusion_score = float(np.clip(fusion_score, 0, 100))

    # ── Convert to Multiplier (smooth curve) ──
    lo = cfg['neutral_zone_lo']
    hi = cfg['neutral_zone_hi']
    if fusion_score >= hi:
        t = (fusion_score - hi) / max(100 - hi, 1)
        multiplier = 1.0 + t * (cfg['max_boost'] - 1.0)
    elif fusion_score <= lo:
        t = fusion_score / max(lo, 1)
        multiplier = cfg['max_penalty'] + t * (1.0 - cfg['max_penalty'])
    else:
        multiplier = 1.0
    multiplier = float(np.clip(multiplier, cfg['max_penalty'], cfg['max_boost']))

    # ── Classification Label ──
    if fusion_score >= 75:
        label = 'WAVE_STRONG'
    elif fusion_score >= 60:
        label = 'WAVE_CONFIRMED'
    elif fusion_score >= 40:
        label = 'WAVE_NEUTRAL'
    elif fusion_score >= 25:
        label = 'WAVE_WEAK'
    else:
        label = 'WAVE_CONFLICT'

    # ── Supplementary signals for downstream use ──
    position_tension = _latest_valid(h.get('position_tension', []), 0)
    from_low = _latest_valid(h.get('from_low_pct', []))
    ret_1d = _latest_valid(h.get('ret_1d', []))
    ret_1y = _latest_valid(h.get('ret_1y', []))

    return {
        'wave_fusion_score': round(fusion_score, 1),
        'wave_fusion_multiplier': round(multiplier, 4),
        'wave_fusion_label': label,
        'wave_confluence': round(confluence, 1),
        'wave_inst_flow': round(inst_flow, 1),
        'wave_harmony': round(harmony_score, 1),
        'wave_harmony_raw': round(harmony_raw, 1) if harmony_raw is not None else 2.0,
        'wave_fundamental': round(fund_quality, 1),
        'wave_position_tension': round(position_tension, 2) if position_tension else 0,
        'wave_from_low': round(from_low, 1) if from_low is not None else None,
        'wave_ret_1d': round(ret_1d, 2) if ret_1d is not None else None,
        'wave_ret_1y': round(ret_1y, 2) if ret_1y is not None else None,
    }


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

    # ── 7-Component Scores (v6.0 — ReturnQuality added) ──
    positional = _calc_positional_quality(pcts, n)
    trend = _calc_trend(pcts, n)
    velocity = _calc_velocity_adaptive(pcts, n)  # Position-relative velocity
    acceleration = _calc_acceleration(pcts, n)
    consistency = _calc_consistency_adaptive(pcts, n)  # Position-aware consistency
    resilience = _calc_resilience(pcts, n)

    # ── v6.0: Return floors REMOVED — returns are now a dedicated component ──
    # Previously returns leaked through 5 doors: velocity floor, acceleration floor,
    # resilience bonus, conviction multiplier, price alignment signal 3.
    # Now: one clean component handles ALL return information.

    # Compute avg_pct — needed by adaptive weights and return quality
    avg_pct = float(np.mean(pcts))

    # ── Return Quality Component (v6.0 — Dedicated 7th Component) ──
    ret_7d = h.get('ret_7d', [])
    ret_30d = h.get('ret_30d', [])
    ret_3m = h.get('ret_3m', [])
    ret_6m = h.get('ret_6m', [])
    from_high = h.get('from_high_pct', [])
    return_quality = _calc_return_quality(ret_3m, ret_6m, ret_7d, ret_30d, from_high, avg_pct, n)

    # ── v6.2: Compute confidence EARLY (needed for weight selection) ──
    bc = BAYESIAN_CONFIDENCE
    confidence = min(1.0, max(bc['min_confidence'], n / bc['full_confidence_weeks']))

    # ── Select Adaptive Weights Based on Percentile Tier + Confidence (v6.2) ──
    weights = _get_adaptive_weights(avg_pct, current_pct=pcts[-1], confidence=confidence)

    # ── Composite Score (7-Component Adaptive Weighted) ──
    trajectory_score = (
        weights['positional'] * positional +
        weights['trend'] * trend +
        weights['velocity'] * velocity +
        weights['acceleration'] * acceleration +
        weights['consistency'] * consistency +
        weights['resilience'] * resilience +
        weights['return_quality'] * return_quality
    )

    # ── Elite Dominance Bonus ──
    # Sustained top-tier presence guarantees a minimum score floor
    trajectory_score = _apply_elite_bonus(trajectory_score, pcts, n)

    # ── Bayesian Confidence Shrinkage (v3.0) ──
    # Shrinks score toward population mean when data is insufficient.
    # 4-week lucky streak ≠ 16-week proven performer. This ensures that.
    # (confidence already computed above for weight selection)
    trajectory_score = confidence * trajectory_score + (1 - confidence) * bc['prior_mean']

    # ── Hurst Exponent Persistence Multiplier (v3.0) ──
    # Determines if the rank trend will PERSIST or REVERT.
    # H > 0.55: trending → boost. H < 0.42: mean-reverting → penalize uptrends.
    hurst_multiplier = _calc_hurst_multiplier(pcts, trend)

    # ── v8.0: Wave Signal Fusion Multiplier ──
    # Cross-validates WAVE Detection's 18 signals with Trajectory calculations.
    # Produces ×0.92 to ×1.10 based on agreement level.
    traj_components = {
        'positional': positional, 'trend': trend, 'velocity': velocity,
        'acceleration': acceleration, 'consistency': consistency,
        'resilience': resilience, 'return_quality': return_quality,
    }
    wave_fusion = _compute_wave_fusion(h, traj_components)
    wave_fusion_multiplier = wave_fusion['wave_fusion_multiplier']

    # ── Price-Rank Alignment Multiplier (v6.1 — purely directional) ──
    price_multiplier, price_label, price_alignment = _calc_price_alignment(ret_7d, ret_30d, pcts, avg_pct)

    # ── Momentum Decay Warning (v6.1) ──
    # Catches stocks with good rank but deteriorating returns
    decay_multiplier, decay_label, decay_score = _calc_momentum_decay(ret_7d, ret_30d, from_high, pcts, avg_pct, ret_6m)

    # ── v8.0: Apply ALL multipliers with UNIFIED CAP ──
    # All 4 multipliers combined into one capped product.
    # Theoretical range: hurst(0.94-1.06) × fusion(0.92-1.10) × price(0.88-1.08) × decay(0.93-1.0)
    # Without cap: 0.709 to 1.258.  Symmetric cap: ×0.78 to ×1.18
    pre_price_score = trajectory_score  # Save for diagnostics
    combined_mult = hurst_multiplier * wave_fusion_multiplier * price_multiplier * decay_multiplier
    combined_mult = float(np.clip(combined_mult, 0.78, 1.18))
    trajectory_score = float(np.clip(trajectory_score * combined_mult, 0, 100))
    pre_decay_score = trajectory_score  # Same as final after unified cap (kept for backward compat)

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
    if return_quality >= 75:
        signal_parts.append('🔥')       # Strong returns
    elif return_quality <= 30:
        signal_parts.append('💧')       # Weak returns
    if decay_tag:
        signal_parts.append(decay_tag)
    # v8.0: Wave fusion signal tag
    wf_label = wave_fusion.get('wave_fusion_label', 'WAVE_NEUTRAL')
    if wf_label == 'WAVE_STRONG':
        signal_parts.append('🌊')
    elif wf_label == 'WAVE_CONFLICT':
        signal_parts.append('🔇')
    signal_tags = ''.join(signal_parts)

    # ── Confidence Intervals (v6.2) ──
    # Wider margin when: low confidence (few weeks) OR high rank volatility
    # margin = (1 - confidence) * (base_uncertainty + volatility_factor)
    ci_base = 18  # Base uncertainty with low data
    ci_vol_factor = min(rank_vol / 3, 8)  # Cap volatility contribution at 8 pts
    ci_margin = (1 - confidence) * (ci_base + ci_vol_factor)
    confidence_lower = max(0, trajectory_score - ci_margin)
    confidence_upper = min(100, trajectory_score + ci_margin * 0.5)  # Asymmetric: upside capped more

    # ══════════════════════════════════════════════════════════════════════════
    # v6.3: ADVANCED TRADING SIGNALS — 5 New Features for Better Returns
    # ══════════════════════════════════════════════════════════════════════════

    # ── 1. CONVICTION SCORE (0-100) ──
    # Combines multiple bullish signals into single actionable metric.
    # Higher conviction = more confident BUY signal.
    # v8.0: Enhanced with Wave Fusion signals.
    conviction = 0
    # Signal 1: Price-Rank Alignment (20 pts max) — reduced from 25 to make room for wave signals
    if price_label == 'PRICE_CONFIRMED':
        conviction += 20
    elif price_label == 'NEUTRAL':
        conviction += 8
    # Signal 2: Return Quality (15 pts max)
    if return_quality >= 75:
        conviction += 15
    elif return_quality >= 60:
        conviction += 9
    elif return_quality >= 50:
        conviction += 4
    # Signal 3: Data Confidence (15 pts max)
    if confidence >= 0.85:
        conviction += 15
    elif confidence >= 0.6:
        conviction += 9
    elif confidence >= 0.4:
        conviction += 4
    # Signal 4: Momentum Quality (15 pts max)
    if tmi >= 70:
        conviction += 15
    elif tmi >= 60:
        conviction += 9
    elif tmi >= 50:
        conviction += 4
    # Signal 5: Positional Strength (10 pts max)
    current_pct = pcts[-1]
    if current_pct >= 90:
        conviction += 10
    elif current_pct >= 80:
        conviction += 7
    elif current_pct >= 70:
        conviction += 4
    # v8.0 Signal 6: Wave Confluence Agreement (15 pts max)
    wf_confluence = wave_fusion.get('wave_confluence', 50)
    if wf_confluence >= 75:
        conviction += 15
    elif wf_confluence >= 60:
        conviction += 10
    elif wf_confluence >= 45:
        conviction += 4
    # v8.0 Signal 7: Institutional Flow (10 pts max)
    wf_flow = wave_fusion.get('wave_inst_flow', 50)
    if wf_flow >= 70:
        conviction += 10
    elif wf_flow >= 55:
        conviction += 6
    elif wf_flow >= 40:
        conviction += 2
    conviction = min(100, conviction)

    # Conviction tags for UI
    if conviction >= 80:
        conviction_tag = 'VERY_HIGH'
        conviction_emoji = '🎯'
    elif conviction >= 65:
        conviction_tag = 'HIGH'
        conviction_emoji = '✅'
    elif conviction >= 45:
        conviction_tag = 'MODERATE'
        conviction_emoji = '⚡'
    elif conviction >= 25:
        conviction_tag = 'LOW'
        conviction_emoji = '⚠️'
    else:
        conviction_tag = 'VERY_LOW'
        conviction_emoji = '❌'

    # ── 2. RISK-ADJUSTED T-SCORE ──
    # Penalizes high-volatility stocks that may reverse.
    # Sharpe-like: score / (1 + rank_volatility × 0.05)
    # rank_vol=20 → 2.0x divisor → score halved. rank_vol=10 → 1.5x → mild penalty.
    vol_penalty = 1.0 + (rank_vol * 0.05)
    risk_adj_score = trajectory_score / vol_penalty
    risk_adj_score = round(risk_adj_score, 2)

    # ── 3. EXIT WARNING SYSTEM ──
    # Detects when to SELL existing holdings.
    # Multiple warning signals aggregated into exit_risk score (0-100).
    exit_risk = 0
    exit_signals = []

    # Exit Signal 1: TMI Collapse (momentum dying)
    if tmi < 40 and trajectory_score > 50:
        exit_risk += 25
        exit_signals.append('TMI_WEAK')
    elif tmi < 50 and trajectory_score > 60:
        exit_risk += 15
        exit_signals.append('TMI_FADING')

    # Exit Signal 2: Price Divergence (rank up but price down)
    if price_label == 'PRICE_DIVERGENT':
        exit_risk += 25
        exit_signals.append('PRICE_DIV')

    # Exit Signal 3: Momentum Decay (returns collapsing)
    if decay_label == 'DECAY_HIGH':
        exit_risk += 30
        exit_signals.append('DECAY_HIGH')
    elif decay_label == 'DECAY_MODERATE':
        exit_risk += 15
        exit_signals.append('DECAY_MOD')

    # Exit Signal 4: Negative Streak (consecutive rank drops)
    neg_streak = 0
    for i in range(len(ranks) - 1, 0, -1):
        if ranks[i] > ranks[i - 1]:  # rank increased = worsened
            neg_streak += 1
        else:
            break
    if neg_streak >= 3:
        exit_risk += 20
        exit_signals.append(f'DROP_{neg_streak}W')
    elif neg_streak >= 2:
        exit_risk += 10
        exit_signals.append('DROP_2W')

    # Exit Signal 5: Pattern Warning
    if pattern_key in ['fading', 'crash', 'topping_out']:
        exit_risk += 20
        exit_signals.append(f'PAT_{pattern_key.upper()}')

    # v8.0 Exit Signal 6: Position Tension (from WAVE Detection)
    # High position tension near top = likely reversal
    wf_tension = wave_fusion.get('wave_position_tension', 0)
    if wf_tension is not None and wf_tension > 0.7 and current_pct >= 70:
        exit_risk += 15
        exit_signals.append('HIGH_TENSION')
    elif wf_tension is not None and wf_tension > 0.5 and current_pct >= 80:
        exit_risk += 8
        exit_signals.append('MOD_TENSION')

    # v8.0 Exit Signal 7: Wave Fusion Conflict (systems disagree)
    if wf_label == 'WAVE_CONFLICT' and trajectory_score > 50:
        exit_risk += 12
        exit_signals.append('WAVE_CONFLICT')

    exit_risk = min(100, exit_risk)
    if exit_risk >= 60:
        exit_tag = 'EXIT_NOW'
        exit_emoji = '🚨'
    elif exit_risk >= 40:
        exit_tag = 'CAUTION'
        exit_emoji = '⚠️'
    elif exit_risk >= 20:
        exit_tag = 'WATCH'
        exit_emoji = '👀'
    else:
        exit_tag = 'HOLD'
        exit_emoji = '✅'

    # ── 4. HOT STREAK DETECTION ──
    # Consecutive weeks of percentile improvement with high current position.
    # Research shows 4+ week momentum streaks have 72% continuation rate.
    hot_streak = False
    hot_streak_weeks = streak  # Already calculated above
    if streak >= 4 and current_pct >= 70:
        hot_streak = True
    elif streak >= 3 and current_pct >= 80:
        hot_streak = True
    elif streak >= 5 and current_pct >= 60:
        hot_streak = True

    # ── 5. VOLUME CONFIRMATION ──
    # Uses volume_score data if available to validate rank moves.
    volume_scores = h.get('volume_score', [])
    latest_vol_score = None
    vol_confirmed = 'NEUTRAL'

    # Get latest valid volume score
    for v in reversed(volume_scores):
        if v is not None and not np.isnan(v):
            latest_vol_score = float(v)
            break

    if latest_vol_score is not None:
        if streak > 0 and latest_vol_score >= 70:
            vol_confirmed = 'STRONG'  # Rank improving + high volume = strong signal
        elif streak > 0 and latest_vol_score >= 50:
            vol_confirmed = 'MODERATE'
        elif streak > 0 and latest_vol_score < 30:
            vol_confirmed = 'WEAK'  # Rank improving but low volume = may reverse
        elif neg_streak >= 2 and latest_vol_score >= 70:
            vol_confirmed = 'DISTRIBUTION'  # Rank falling + high volume = selling pressure

    # v8.0: Cross-check with institutional flow for enhanced volume confirmation
    wf_inst = wave_fusion.get('wave_inst_flow', 50)
    if vol_confirmed == 'MODERATE' and wf_inst >= 70:
        vol_confirmed = 'STRONG'   # Institutional flow confirms moderate volume
    elif vol_confirmed == 'STRONG' and wf_inst < 30:
        vol_confirmed = 'MODERATE'  # Institutional flow contradicts volume score

    return {
        'trajectory_score': round(trajectory_score, 2),
        'positional': round(positional, 2),
        'trend': round(trend, 2),
        'velocity': round(velocity, 2),
        'acceleration': round(acceleration, 2),
        'consistency': round(consistency, 2),
        'resilience': round(resilience, 2),
        'return_quality': round(return_quality, 2),
        'hurst': round(_estimate_hurst(pcts), 3) if n >= HURST_CONFIG['min_weeks'] else 0.5,
        'confidence': round(confidence, 3),
        'confidence_lower': round(confidence_lower, 2),
        'confidence_upper': round(confidence_upper, 2),
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
        'combined_mult': round(combined_mult, 4),
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
        'sparkline': sparkline_data,
        # v6.3: Advanced Trading Signals
        'conviction': conviction,
        'conviction_tag': conviction_tag,
        'conviction_emoji': conviction_emoji,
        'risk_adj_score': risk_adj_score,
        'exit_risk': exit_risk,
        'exit_tag': exit_tag,
        'exit_emoji': exit_emoji,
        'exit_signals': ','.join(exit_signals) if exit_signals else '',
        'hot_streak': hot_streak,
        'hot_streak_weeks': hot_streak_weeks,
        'vol_confirmed': vol_confirmed,
        'latest_vol_score': round(latest_vol_score, 1) if latest_vol_score else None,
        # v8.0: Wave Signal Fusion
        'wave_fusion_score': wave_fusion.get('wave_fusion_score', 50),
        'wave_fusion_multiplier': wave_fusion.get('wave_fusion_multiplier', 1.0),
        'wave_fusion_label': wave_fusion.get('wave_fusion_label', 'WAVE_NEUTRAL'),
        'wave_confluence': wave_fusion.get('wave_confluence', 50),
        'wave_inst_flow': wave_fusion.get('wave_inst_flow', 50),
        'wave_harmony': wave_fusion.get('wave_harmony', 50),
        'wave_harmony_raw': wave_fusion.get('wave_harmony_raw', 2.0),
        'wave_fundamental': wave_fusion.get('wave_fundamental', 50),
        'wave_position_tension': wave_fusion.get('wave_position_tension', 0),
        'wave_from_low': wave_fusion.get('wave_from_low'),
        'wave_ret_1d': wave_fusion.get('wave_ret_1d'),
        'wave_ret_1y': wave_fusion.get('wave_ret_1y'),
    }


def _empty_trajectory(ranks, totals, pcts, n):
    """Return neutral trajectory for insufficient data"""
    return {
        'trajectory_score': 0, 'positional': 0, 'trend': 50, 'velocity': 50,
        'acceleration': 50, 'consistency': 50, 'resilience': 50,
        'return_quality': 50,
        'hurst': 0.5, 'confidence': BAYESIAN_CONFIDENCE['min_confidence'],
        'confidence_lower': 0, 'confidence_upper': 13.5,
        'grade': 'F', 'grade_emoji': '📉',
        'pattern_key': 'new_entry', 'pattern': '💎 New Entry',
        'price_alignment': 50.0, 'price_multiplier': 1.0,
        'price_label': 'NEUTRAL', 'price_tag': '',
        'pre_price_score': 0,
        'decay_score': 0, 'decay_multiplier': 1.0,
        'decay_label': '', 'decay_tag': '',
        'pre_decay_score': 0,
        'combined_mult': 1.0,
        'signal_tags': '',
        'sector_alpha_tag': 'NEUTRAL', 'sector_alpha_value': 0,
        'current_rank': int(ranks[-1]) if ranks else 0,
        'best_rank': int(min(ranks)) if ranks else 0,
        'worst_rank': int(max(ranks)) if ranks else 0,
        'avg_rank': round(np.mean(ranks), 1) if ranks else 0,
        'rank_change': 0, 'last_week_change': 0, 'streak': 0,
        'tmi': 50.0, 'weeks': n, 'rank_volatility': 0,
        'sparkline': [round(p, 1) for p in pcts] if pcts else [],
        # v6.3: Advanced Trading Signals (defaults)
        'conviction': 0, 'conviction_tag': 'VERY_LOW', 'conviction_emoji': '❌',
        'risk_adj_score': 0, 'exit_risk': 0, 'exit_tag': 'HOLD', 'exit_emoji': '✅',
        'exit_signals': '', 'hot_streak': False, 'hot_streak_weeks': 0,
        'vol_confirmed': 'NEUTRAL', 'latest_vol_score': None,
        # v8.0: Wave Signal Fusion (defaults)
        'wave_fusion_score': 50, 'wave_fusion_multiplier': 1.0,
        'wave_fusion_label': 'WAVE_NEUTRAL', 'wave_confluence': 50,
        'wave_inst_flow': 50, 'wave_harmony': 50, 'wave_harmony_raw': 2.0,
        'wave_fundamental': 50, 'wave_position_tension': 0,
        'wave_from_low': None, 'wave_ret_1d': None, 'wave_ret_1y': None,
    }


# ── Adaptive Weight Selection ──

def _get_adaptive_weights(avg_pct: float, current_pct: float = None, confidence: float = None) -> dict:
    """
    Select weight profile based on stock's effective percentile position.
    Uses smooth interpolation between tiers for continuous transitions.
    
    v5.2: If current_pct >> avg_pct (recent riser), use blended effective_pct
    so weight selection reflects current reality, not stale history.
    A stock at current_pct=72, avg_pct=35 gets effective=57 → 'mid' weights
    instead of 'bottom' weights which over-penalize with 25% velocity weight.
    
    v6.2: Confidence-Aware Dynamic Weight Shifting.
    Low-confidence stocks (few weeks): shift weight toward momentum signals
    (velocity, acceleration, trend) — recent behavior is all we know.
    High-confidence stocks (many weeks): shift weight toward stability signals
    (positional, consistency, resilience) — proven track record matters more.
    
    Shift magnitude: ±5% max per component (preserves architecture stability).
    """
    effective_pct = avg_pct
    if current_pct is not None and current_pct > avg_pct + 15:
        effective_pct = 0.4 * avg_pct + 0.6 * current_pct
    
    # Select base weights by percentile tier (unchanged logic)
    if effective_pct >= 90:
        base_weights = ADAPTIVE_WEIGHTS['elite'].copy()
    elif effective_pct >= 70:
        ratio = (effective_pct - 70) / 20
        strong = ADAPTIVE_WEIGHTS['strong']
        elite = ADAPTIVE_WEIGHTS['elite']
        base_weights = {k: strong[k] * (1 - ratio) + elite[k] * ratio for k in strong}
    elif effective_pct >= 40:
        ratio = (effective_pct - 40) / 30
        mid = ADAPTIVE_WEIGHTS['mid']
        strong = ADAPTIVE_WEIGHTS['strong']
        base_weights = {k: mid[k] * (1 - ratio) + strong[k] * ratio for k in mid}
    else:
        ratio = effective_pct / 40
        bottom = ADAPTIVE_WEIGHTS['bottom']
        mid = ADAPTIVE_WEIGHTS['mid']
        base_weights = {k: bottom[k] * (1 - ratio) + mid[k] * ratio for k in bottom}
    
    # ── v6.2: Confidence-Aware Micro-Adjustment ──
    # Skip if confidence not provided (backward compatible for UI calls)
    if confidence is None:
        return base_weights
    
    # Shift factor: -1 at confidence=0.25, 0 at confidence=0.625, +1 at confidence=1.0
    # Low confidence → negative shift → boost momentum signals
    # High confidence → positive shift → boost stability signals
    shift_factor = (confidence - 0.625) / 0.375  # Range: -1 to +1
    shift_factor = float(np.clip(shift_factor, -1.0, 1.0))
    
    # Define which components gain/lose weight based on confidence
    # Momentum signals: velocity, acceleration, trend (gain weight when LOW confidence)
    # Stability signals: positional, consistency, resilience (gain weight when HIGH confidence)
    # return_quality: neutral (no shift)
    
    # Max shift per component: 5% (0.05)
    max_shift = 0.05
    
    # Calculate shifts (positive shift_factor = boost stability, reduce momentum)
    shifts = {
        'positional': shift_factor * max_shift,       # +5% at high conf, -5% at low
        'trend': -shift_factor * max_shift * 0.6,     # -3% at high conf, +3% at low
        'velocity': -shift_factor * max_shift,        # -5% at high conf, +5% at low
        'acceleration': -shift_factor * max_shift * 0.6,  # -3% at high conf, +3% at low
        'consistency': shift_factor * max_shift * 0.8,    # +4% at high conf, -4% at low
        'resilience': shift_factor * max_shift * 0.6,     # +3% at high conf, -3% at low
        'return_quality': 0.0                             # Neutral — always relevant
    }
    
    # Apply shifts and ensure weights stay positive
    adjusted = {k: max(0.02, base_weights[k] + shifts[k]) for k in base_weights}
    
    # Renormalize to sum to 1.0 (critical for score integrity)
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}


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
            # v5.3: reduced bonus_extra from 8 to 4 to prevent over-inflation
            bonus_extra = (ratio - required_ratio) / (1.0 - required_ratio + 0.01) * 4
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


# ── Return Quality Component (v6.0) ──

def _calc_return_quality(ret_3m_list: List[float], ret_6m_list: List[float],
                         ret_7d_list: List[float], ret_30d_list: List[float],
                         from_high_list: List[float], avg_pct: float, n: int) -> float:
    """
    Return Quality — Dedicated 7th scoring component (v6.0).

    ARCHITECTURE: Replaces 5 scattered return touchpoints (velocity floor,
    acceleration floor, resilience bonus, conviction multiplier, price alignment
    signal 3) with a single clean component (0-100).

    4 SUB-SIGNALS:
      Signal 1 (30%): 3-Month Return Score — medium-term momentum quality
      Signal 2 (30%): 6-Month Return Score — institutional horizon confirmation
      Signal 3 (20%): Short-Term Momentum — ret_7d + ret_30d recency signal
      Signal 4 (20%): Return Health — cross-timeframe agreement + from-high distance

    Returns: float (0-100), where 50 = neutral, >70 = strong, <30 = weak.
    """
    if n < RETURN_QUALITY.get('min_weeks', 3):
        return 50.0

    def _latest_valid(lst):
        """Get latest non-None, non-NaN value from a list."""
        if not lst:
            return None
        for v in reversed(lst):
            if v is not None and not np.isnan(v):
                return float(v)
        return None

    r3m = _latest_valid(ret_3m_list)
    r6m = _latest_valid(ret_6m_list)
    r7d = _latest_valid(ret_7d_list)
    r30d = _latest_valid(ret_30d_list)
    fh = _latest_valid(from_high_list)

    # ═══════════════════════════════════════════════════════════════════
    # Signal 1: 3-Month Return Score (30%)
    # Captures medium-term momentum. Sigmoid-like mapping from return %
    # to 0-100. Position-aware: elite stocks SHOULD have gains.
    # ═══════════════════════════════════════════════════════════════════
    if r3m is not None:
        if r3m >= 40:
            s1 = 92.0 + min((r3m - 40) / 60 * 8, 8.0)     # 92-100
        elif r3m >= 25:
            s1 = 78.0 + (r3m - 25) / 15 * 14               # 78-92
        elif r3m >= 15:
            s1 = 65.0 + (r3m - 15) / 10 * 13               # 65-78
        elif r3m >= 5:
            s1 = 52.0 + (r3m - 5) / 10 * 13                # 52-65
        elif r3m >= 0:
            s1 = 42.0 + r3m / 5 * 10                       # 42-52
        elif r3m >= -10:
            s1 = 25.0 + (r3m + 10) / 10 * 17               # 25-42
        elif r3m >= -25:
            s1 = 10.0 + (r3m + 25) / 15 * 15               # 10-25
        else:
            s1 = max(5.0, 10.0 + (r3m + 25) / 25 * 5)      # 5-10
    else:
        s1 = 50.0

    # ═══════════════════════════════════════════════════════════════════
    # Signal 2: 6-Month Return Score (30%)
    # Captures institutional-horizon quality. Wider range since 6m returns
    # span -50% to +150% typically. Strong 6m = proven gainer.
    # ═══════════════════════════════════════════════════════════════════
    if r6m is not None:
        if r6m >= 80:
            s2 = 93.0 + min((r6m - 80) / 120 * 7, 7.0)     # 93-100
        elif r6m >= 50:
            s2 = 80.0 + (r6m - 50) / 30 * 13               # 80-93
        elif r6m >= 30:
            s2 = 65.0 + (r6m - 30) / 20 * 15               # 65-80
        elif r6m >= 10:
            s2 = 50.0 + (r6m - 10) / 20 * 15               # 50-65
        elif r6m >= 0:
            s2 = 40.0 + r6m / 10 * 10                       # 40-50
        elif r6m >= -15:
            s2 = 22.0 + (r6m + 15) / 15 * 18               # 22-40
        elif r6m >= -30:
            s2 = 10.0 + (r6m + 30) / 15 * 12               # 10-22
        else:
            s2 = max(5.0, 10.0 + (r6m + 30) / 30 * 5)      # 5-10
    else:
        s2 = 50.0

    # ═══════════════════════════════════════════════════════════════════
    # Signal 3: Short-Term Momentum (20%)
    # ret_7d (35%) + ret_30d (65%). Recent price action quality.
    # Higher weight on 30d because 7d is noisy.
    # ═══════════════════════════════════════════════════════════════════
    st_parts = []
    st_weights = []

    if r7d is not None:
        if r7d >= 5:
            st7 = 85.0 + min((r7d - 5) / 10 * 15, 15.0)
        elif r7d >= 2:
            st7 = 68.0 + (r7d - 2) / 3 * 17
        elif r7d >= 0:
            st7 = 52.0 + r7d / 2 * 16
        elif r7d >= -2:
            st7 = 38.0 + (r7d + 2) / 2 * 14
        elif r7d >= -5:
            st7 = 20.0 + (r7d + 5) / 3 * 18
        else:
            st7 = max(5.0, 20.0 + (r7d + 5) / 10 * 15)
        st_parts.append(st7)
        st_weights.append(0.35)

    if r30d is not None:
        if r30d >= 15:
            st30 = 85.0 + min((r30d - 15) / 20 * 15, 15.0)
        elif r30d >= 5:
            st30 = 65.0 + (r30d - 5) / 10 * 20
        elif r30d >= 0:
            st30 = 50.0 + r30d / 5 * 15
        elif r30d >= -5:
            st30 = 35.0 + (r30d + 5) / 5 * 15
        elif r30d >= -15:
            st30 = 15.0 + (r30d + 15) / 10 * 20
        else:
            st30 = max(5.0, 15.0 + (r30d + 15) / 20 * 10)
        st_parts.append(st30)
        st_weights.append(0.65)

    if st_parts:
        total_stw = sum(st_weights)
        s3 = sum(p * w for p, w in zip(st_parts, st_weights)) / total_stw
    else:
        s3 = 50.0

    # ═══════════════════════════════════════════════════════════════════
    # Signal 4: Return Health (20%)
    # 4a (55%): Cross-timeframe agreement — r3m and r6m same direction?
    # 4b (45%): Distance from 52-week high — how far has it corrected?
    # ═══════════════════════════════════════════════════════════════════
    health_parts = []

    # 4a: Cross-timeframe agreement
    if r3m is not None and r6m is not None:
        both_pos = r3m > 0 and r6m > 0
        both_neg = r3m < -5 and r6m < -5
        if both_pos:
            # Both positive — strength depends on magnitude
            agree = min(80.0 + (r3m + r6m) / 4, 100.0)
        elif both_neg:
            agree = max(10.0, 30.0 + (r3m + r6m) / 4)
        elif (r3m > 0) != (r6m > 0):
            # Disagreement: transition zone — 3m improving but 6m still negative, or vice versa
            agree = 35.0
        else:
            agree = 50.0
        health_parts.append(('agree', agree, 0.55))

    # 4b: Distance from 52-week high
    if fh is not None:
        if fh >= 0:
            fh_score = 90.0       # At or near high
        elif fh >= -5:
            fh_score = 78.0       # Very close to high
        elif fh >= -10:
            fh_score = 65.0       # Normal pullback
        elif fh >= -20:
            fh_score = 48.0       # Moderate correction
        elif fh >= -30:
            fh_score = 32.0       # Significant correction
        else:
            fh_score = max(10.0, 32.0 + (fh + 30) / 30 * 22)  # Deep correction
        health_parts.append(('fh', fh_score, 0.45))

    if health_parts:
        total_hw = sum(w for _, _, w in health_parts)
        s4 = sum(v * w for _, v, w in health_parts) / total_hw
    else:
        s4 = 50.0

    # ═══════════════════════════════════════════════════════════════════
    # Final Composite: weighted blend of 4 sub-signals
    # ═══════════════════════════════════════════════════════════════════
    score = 0.30 * s1 + 0.30 * s2 + 0.20 * s3 + 0.20 * s4
    return float(np.clip(score, 0, 100))


def _calc_price_alignment(ret_7d: List[float], ret_30d: List[float],
                          pcts: List[float], avg_pct: float) -> Tuple[float, str, float]:
    """
    Price-Rank Alignment Multiplier (v6.1 — purely directional).

    Measures whether SHORT-TERM return direction agrees with rank movement.
    Does NOT score return magnitude — that is handled by ReturnQuality.

    TWO SIGNALS:
      Signal 1 (55%): EMA-Smoothed Weekly Directional Agreement
                      Does sign(ret_7d) match sign(percentile_change)?
      Signal 2 (45%): Monthly Directional Agreement
                      Does sign(ret_30d) match sign(percentile_change)?

    Both signals score DIRECTION ONLY (positive/negative/flat), never magnitude.
    This ensures ret_30d enters exactly ONE door: ReturnQuality component.

    MULTIPLIER RANGE: ×0.88 (strong divergence) to ×1.08 (strong confirmation)

    Returns: (multiplier, label, alignment_score)
    """
    cfg = PRICE_ALIGNMENT
    n = len(pcts)

    # ── Guard: Need valid return data ──
    valid_ret7 = [r for r in ret_7d if r is not None and not np.isnan(r)]
    if len(valid_ret7) < cfg['min_weeks'] or n < cfg['min_weeks']:
        return 1.0, 'NEUTRAL', 50.0

    # ── EMA-smooth ret_7d to reduce noise ──
    ema_span = cfg.get('ema_span', 3)
    smoothed_r7 = _ema_smooth(ret_7d, span=ema_span)

    # Build aligned pairs: (ema_r7, r30, percentile)
    pairs = []
    for i in range(n):
        r7e = smoothed_r7[i] if i < len(smoothed_r7) else float('nan')
        r30 = ret_30d[i] if i < len(ret_30d) else float('nan')
        if r7e is not None and not np.isnan(r7e):
            pairs.append((r7e, r30, pcts[i]))

    if len(pairs) < cfg['min_weeks']:
        return 1.0, 'NEUTRAL', 50.0

    # ── Signal 1: EMA-Smoothed Weekly Directional Agreement (55%) ──
    agree = 0.0
    total_weight = 0.0
    noise_band = cfg['noise_band_stable'] if avg_pct > 80 else cfg['noise_band_normal']
    recency_window = cfg.get('recency_window', 4)
    np_ = len(pairs)

    for i in range(1, np_):
        r7e = pairs[i][0]
        r_chg = pairs[i][2] - pairs[i - 1][2]  # Percentile change

        # Skip noise — tiny rank moves for elite stocks
        if abs(r_chg) < noise_band and abs(r7e) < 1.0:
            continue

        # Recency weight: last `recency_window` weeks get 2×
        w = 2.0 if (np_ - 1 - i) < recency_window else 1.0
        total_weight += w

        # Direction-only scoring: does sign(ret_7d) match sign(rank_change)?
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
        dir_score = 55.0  # No significant rank moves — slight positive bias

    # ── Signal 2: Monthly Directional Agreement (45%) — DIRECTION ONLY ──
    # Does sign(ret_30d) agree with sign(percentile_change_over_recent_weeks)?
    # Never scores magnitude — that's return_quality's job.
    recent_window = min(6, len(pairs))
    recent_pairs = pairs[-recent_window:]
    dir_agree = 0.0
    dir_total = 0.0

    for i in range(1, len(recent_pairs)):
        r30 = recent_pairs[i][1]
        if r30 is None or np.isnan(r30):
            continue

        pct_chg = recent_pairs[i][2] - recent_pairs[i - 1][2]
        w = 1.5 if i >= len(recent_pairs) - 2 else 1.0  # Recency bias
        dir_total += w

        # Directional agreement: sign(r30) matches sign(pct_change)?
        if r30 > 1.0 and pct_chg > 0.5:
            dir_agree += 1.0 * w    # Both positive direction
        elif r30 < -1.0 and pct_chg < -0.5:
            dir_agree += 0.8 * w    # Both negative — consistent
        elif abs(r30) <= 1.0:
            dir_agree += 0.3 * w    # Return flat — not disagreeing
        elif (r30 > 1.0 and pct_chg < -2.0) or (r30 < -1.0 and pct_chg > 2.0):
            dir_agree -= 0.5 * w    # Strong divergence
        else:
            dir_agree -= 0.1 * w    # Mild divergence

    if dir_total > 0:
        monthly_dir_score = float(np.clip((dir_agree / dir_total) * 50 + 50, 0, 100))
    else:
        monthly_dir_score = 55.0

    # ── Signal Disagreement Penalty (2 signals) ──
    signal_spread = abs(dir_score - monthly_dir_score)
    if signal_spread > 50:
        disagreement_penalty = 5.0   # Strong contradiction
    elif signal_spread > 30:
        disagreement_penalty = 2.5   # Moderate contradiction
    else:
        disagreement_penalty = 0.0   # Signals agree — no penalty

    # ── Composite Alignment Score (v6.1: 2 directional signals) ──
    alignment = 0.55 * dir_score + 0.45 * monthly_dir_score - disagreement_penalty
    alignment = float(np.clip(alignment, 0, 100))

    # ── Convert to Multiplier (×0.88 to ×1.08) ──
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
                         avg_pct: float,
                         ret_6m: Optional[List[float]] = None) -> Tuple[float, str, int]:
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

    v6.1: Uses actual ret_6m for Proven Winner Exemption instead of avg_pct proxy.
    v6.1: Smooth continuous multiplier instead of step-function cliffs.

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
    latest_r6m = _get_latest(ret_6m) if ret_6m else None

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

    # v6.1: Proven Winner Exemption — uses ACTUAL ret_6m instead of avg_pct proxy
    # A stock with ret_6m=82% pulling back 10% recently is NOT a trap —
    # it's a proven winner going through normal consolidation.
    if latest_r6m is not None:
        if latest_r6m > 60:
            decay_score = int(decay_score * 0.45)   # 55% reduction — strong proven winner
        elif latest_r6m > 30:
            decay_score = int(decay_score * 0.60)   # 40% reduction — solid gainer
        elif latest_r6m > 15:
            decay_score = int(decay_score * 0.75)   # 25% reduction — moderate gainer
        # ret_6m <= 15: no exemption — stock hasn't proven itself over 6 months
    else:
        # Fallback: ret_6m unavailable — use avg_pct as rough proxy
        if avg_pct >= 80:
            decay_score = int(decay_score * 0.60)
        elif avg_pct >= 70:
            decay_score = int(decay_score * 0.75)

    # v6.1: Smooth continuous multiplier (no step-function cliffs)
    # Maps decay_score 0-100 smoothly to multiplier 1.0-0.93
    # Formula: mult = 1.0 - (decay_score / 100) * (1.0 - high_decay_mult)
    # Score 0 → ×1.00, Score 15 → ×0.9895, Score 35 → ×0.9755, Score 60 → ×0.958, Score 100 → ×0.93
    if decay_score > 0:
        max_penalty_depth = 1.0 - cfg['high_decay_multiplier']  # 0.07
        multiplier = 1.0 - (decay_score / 100.0) * max_penalty_depth
        multiplier = float(np.clip(multiplier, cfg['high_decay_multiplier'], 1.0))

        # Label based on thresholds (for UI display)
        if decay_score >= cfg['severe_threshold']:
            label = 'DECAY_HIGH'
        elif decay_score >= cfg['moderate_threshold']:
            label = 'DECAY_MODERATE'
        elif decay_score >= cfg['mild_threshold']:
            label = 'DECAY_MILD'
        else:
            label = ''
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
    
    # v4.0: Recency-biased positional scoring
    # Problem found: stocks that improved recently were penalized by early bad weeks.
    # e.g. stock at avg_pct=66 with current_pct=89 was dragged down by old data.
    # Solution: Reduce overall weight to 10%, boost current to 60%.
    # For late movers (recent >> overall), further reduce overall to 5%.
    is_late_mover = recent_avg > overall_avg + 15
    if is_late_mover:
        raw_pct = 0.60 * current_pct + 0.35 * recent_avg + 0.05 * overall_avg
    else:
        raw_pct = 0.60 * current_pct + 0.30 * recent_avg + 0.10 * overall_avg
    
    # v5.2 SURGE OVERRIDE: Stock surged from bottom to top recently.
    # PROBLEM: 522241 has avg_pct=35 but cur=72 → positional=53 crushes it.
    # A stock that moved from 35th to 72nd percentile in recent weeks PROVED
    # itself. Positional should reflect WHERE IT IS, not where it was months ago.
    if current_pct >= 65 and overall_avg < 50 and recent_avg >= 55:
        surge_pct = 0.75 * current_pct + 0.25 * recent_avg
        raw_pct = max(raw_pct, surge_pct)
    
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
    
    # v5.3: Reduced trend floors — were inflating ~30% of stocks.
    # Old floors (70/65/58/52) prevented top stocks from EVER showing weak trend.
    recent_avg_pct = np.mean(pcts[-min(4, n):])
    if recent_avg_pct > 95:
        raw_score = max(raw_score, 62)   # v5.3: was 70
    elif recent_avg_pct > 90:
        raw_score = max(raw_score, 57)   # v5.3: was 65
    elif recent_avg_pct > 80:
        raw_score = max(raw_score, 52)   # v5.3: was 58
    elif recent_avg_pct > 70:
        raw_score = max(raw_score, 48)   # v5.3: was 52
    
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
    # v5.3: Reduced hold_bonus — was inflating velocity for top stocks (B-grade had
    # HIGHER velocity than S-grade because mid-range movers scored 77.5 vs elite's 73.8).
    if current_pct >= 95:
        hold_bonus = 8.0           # v5.3: was 15 — too generous
        change_sensitivity = 0.7   # v5.3: was 0.6
    elif current_pct >= 85:
        hold_bonus = 5.0           # v5.3: was 10
        change_sensitivity = 0.8   # v5.3: was 0.75
    elif current_pct >= 70:
        hold_bonus = 2.0           # v5.3: was 5
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
    """
    Rate-of-change of velocity — is improvement ACCELERATING or DECELERATING?
    
    v5.0 UPGRADE: Rolling Velocity WLS (replaces naive 2-point difference).
    
    OLD approach: Just (recent_velocity - prior_velocity) using 2 endpoints.
    PROBLEM: Extremely noise-sensitive. A single bad week could flip the entire
    score because it only sampled 2 velocity measurements at fixed points.
    
    NEW approach: Same philosophy as _calc_trend (which works excellently).
    1. Compute velocity at each point using a rolling window
    2. Fit exponentially-weighted least squares on the velocity series
    3. The SLOPE of the velocity series = acceleration
    
    This gives a smooth, robust acceleration estimate that uses ALL available
    data with proper recency weighting.
    
    Also adds position-relative adjustment: at high percentiles, maintaining
    velocity is harder (same logic as velocity's hold_bonus).
    """
    if n < window + 2:
        return 50.0

    # Step 1: Compute rolling velocities at each point
    velocities = []
    for i in range(window, n):
        vel = (pcts[i] - pcts[i - window]) / window
        velocities.append(vel)

    n_vel = len(velocities)
    if n_vel < 3:
        # Fallback to simple 2-point for very short series
        recent_vel = velocities[-1] if velocities else 0
        prior_vel = velocities[0] if velocities else 0
        accel = recent_vel - prior_vel
        return float(np.clip(accel / 2.0 * 50 + 50, 0, 100))

    # Step 2: Exponentially-weighted least squares on velocity series
    x = np.arange(n_vel, dtype=float)
    y = np.array(velocities, dtype=float)
    weights = np.exp(0.15 * x)  # Slightly steeper than trend's 0.12
    weights /= weights.sum()

    wx = (weights * x).sum()
    wy = (weights * y).sum()
    wxx = (weights * x * x).sum()
    wxy = (weights * x * y).sum()
    w_sum = weights.sum()

    denom = w_sum * wxx - wx * wx
    if abs(denom) < 1e-10:
        return 50.0

    # Slope of velocity = acceleration
    accel_slope = (w_sum * wxy - wx * wy) / denom

    # Normalize: +1.5 pct/week² = 100, -1.5 = 0
    # (Tighter than old ±2 because WLS is smoother, so real accelerations are smaller)
    raw_score = float(np.clip(accel_slope / 1.5 * 50 + 50, 0, 100))

    # Step 3: Position-relative adjustment (v5.3: floors reduced)
    current_pct = pcts[-1]
    if current_pct >= 92:
        raw_score = max(raw_score, 42.0)   # v5.3: was 45
    elif current_pct >= 80:
        raw_score = max(raw_score, 40.0)   # v5.3: was 42

    return float(np.clip(raw_score, 0, 100))


def _calc_consistency_adaptive(pcts: List[float], n: int) -> float:
    """
    Position-Aware Consistency + Information Ratio (v4.0).
    
    v4.0 UPGRADE: Added Directional Volatility Discount.
    Problem found: 24/30 missed gainers were VOLATILE pattern (pct_range=86).
    These stocks swing rank 50→500→100 as they gain. The old consistency
    function crushed them with std-based penalty. But if the DIRECTION is
    improving (2nd-half avg < 1st-half avg ranks), the volatility is a
    FEATURE of rapid improvement, not a bug. Discount the penalty by 30%.
    
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

    # === v4.0: DIRECTIONAL VOLATILITY DISCOUNT ===
    # If recent performance > overall AND current is above average:
    # The stock is volatile but IMPROVING → reduce consistency penalty
    volatility_discount = 1.0  # default: no discount
    if n >= 6:
        half = n // 2
        first_half_avg = float(np.mean(pcts[:half]))
        second_half_avg = float(np.mean(pcts[half:]))
        is_improving = second_half_avg > first_half_avg + 5
        is_currently_high = current_pct > avg_pct
        
        if is_improving and is_currently_high:
            # Discount: reduce std-based penalty by 30%
            # This means std is treated as 70% of actual value
            volatility_discount = 0.70
        elif is_improving:
            # Improving but not currently at top — 15% discount
            volatility_discount = 0.85
    
    # Apply discount to std for consistency calculations
    effective_std = std * volatility_discount

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
    
    # === POSITION-RELATIVE CONSISTENCY (v5.0: Smooth Interpolation) ===
    # OLD: Hard cutoffs at avg_pct=85 and 60 with completely different formulas.
    # A stock at 84.9 got a totally different calculation than 85.1.
    # NEW: Compute all tier scores, then smoothly blend using interpolation.
    
    # Elite tier score (relevant when avg_pct ≥ 75+)
    pct_range = max(pcts) - min(pcts)
    band_score = float(np.clip(100 - pct_range * 1.67, 0, 100))
    time_at_top = sum(1 for p in pcts if p >= 80) / len(pcts) * 100
    vol_bonus = float(np.clip(100 - effective_std * 5, 0, 100))
    elite_score = 0.30 * band_score + 0.25 * time_at_top + 0.20 * vol_bonus + 0.25 * ir_score
    
    # Strong tier score (relevant when avg_pct 50-85)
    stability = float(np.clip(100 - effective_std * 2, 0, 100))
    direction = positive_ratio * 100
    trajectory_lift = min((pcts[-1] - pcts[0]) / 20.0 * 10, 15)
    strong_score = float(np.clip(0.35 * stability + 0.30 * direction + 0.35 * ir_score + trajectory_lift, 0, 100))
    
    # Lower tier score (relevant when avg_pct < 50)
    lower_score = 0.35 * stability + 0.30 * direction + 0.35 * ir_score
    
    # Smooth blending with sigmoid-like transitions
    if avg_pct >= 90:
        # Pure elite
        return float(np.clip(elite_score, 0, 100))
    elif avg_pct >= 75:
        # Blend elite ↔ strong over 75-90 range (15 pct transition zone)
        blend = (avg_pct - 75) / 15.0  # 0 at 75, 1 at 90
        return float(np.clip(blend * elite_score + (1 - blend) * strong_score, 0, 100))
    elif avg_pct >= 50:
        # Pure strong
        return float(np.clip(strong_score, 0, 100))
    elif avg_pct >= 35:
        # Blend strong ↔ lower over 35-50 range (15 pct transition zone)
        blend = (avg_pct - 35) / 15.0  # 0 at 35, 1 at 50
        return float(np.clip(blend * strong_score + (1 - blend) * lower_score, 0, 100))
    else:
        # Pure lower
        return float(np.clip(lower_score, 0, 100))


def _calc_resilience(pcts: List[float], n: int) -> float:
    """
    Multi-factor Recovery Resilience (v5.0).
    
    v5.0 UPGRADE: Full drawdown analysis (replaces simple recovery ratio).
    
    OLD approach: Just checked "are you at your peak?" (recovery_ratio) with
    a drawdown magnitude penalty. Gave the SAME score to:
      - A stock that crashed 30 pct and recovered in 2 weeks (excellent!)
      - A stock that crashed 30 pct and recovered in 8 weeks (mediocre)
    
    NEW approach: 4-factor resilience scoring:
    
    1. RECOVERY RATIO (30%): How much of the max drawdown has been recovered?
       Current drawdown vs max drawdown. Fully recovered = 100.
    
    2. RECOVERY SPEED (25%): How FAST was the recovery from the deepest point?
       V-shaped recovery (2 weeks) >> L-shaped recovery (8+ weeks).
       Measured as: what % of max_dd was recovered in the first 3 weeks after trough.
    
    3. DRAWDOWN SEVERITY (25%): How deep were the drawdowns?
       Inverse penalty: bigger drops = less resilient. Uses both max and average
       drawdowns to distinguish "one big crash" vs "many moderate dips."
    
    4. DRAWDOWN FREQUENCY (20%): How often does it fall into drawdowns?
       A stock that dips 5 times is more fragile than one that dipped once.
       Counted as episodes where dd > 3 pct.
    """
    if n < 4:
        return 50.0

    arr = np.array(pcts)
    peak = np.maximum.accumulate(arr)
    drawdowns = peak - arr
    max_dd = float(np.max(drawdowns))
    current_dd = float(drawdowns[-1])

    if max_dd < 1.0:
        return 100.0  # No meaningful drawdown — perfect resilience

    # ── Factor 1: Recovery Ratio (are you recovered?) ──
    recovery_ratio = 1.0 - safe_div(current_dd, max_dd, 1.0)
    recovery_score = recovery_ratio * 100  # 0-100

    # ── Factor 2: Recovery Speed (how fast did you bounce back?) ──
    max_dd_idx = int(np.argmax(drawdowns))
    speed_score = 50.0  # Default: neutral
    if max_dd_idx < n - 1:
        # Measure how much recovered in first 3 weeks after trough
        recovery_window = min(3, n - max_dd_idx - 1)
        if recovery_window > 0:
            # Percentile at trough and 3 weeks later
            trough_pct = arr[max_dd_idx]
            post_pct = arr[min(max_dd_idx + recovery_window, n - 1)]
            recovered_amount = post_pct - trough_pct
            # What % of the drop was recovered in that window?
            recovery_pct = safe_div(recovered_amount, max_dd, 0.0)
            recovery_pct = max(0, min(1.0, recovery_pct))
            # V-shape: >80% recovered in 3 weeks = 100 speed score
            if recovery_pct >= 0.8:
                speed_score = 90 + (recovery_pct - 0.8) * 50  # 90-100
            elif recovery_pct >= 0.5:
                speed_score = 65 + (recovery_pct - 0.5) / 0.3 * 25  # 65-90
            elif recovery_pct >= 0.2:
                speed_score = 35 + (recovery_pct - 0.2) / 0.3 * 30  # 35-65
            else:
                speed_score = max(10, recovery_pct / 0.2 * 35)  # 10-35
    else:
        # Still at the deepest point — slow recovery
        speed_score = 15.0

    # ── Factor 3: Drawdown Severity (how deep were the dips?) ──
    avg_dd = float(np.mean(drawdowns[drawdowns > 0.5])) if np.any(drawdowns > 0.5) else 0
    # Max dd penalty: 50 pct drop → 0 score. 10 pct drop → ~80 score
    max_dd_score = float(np.clip(100 - max_dd * 2, 0, 100))
    avg_dd_score = float(np.clip(100 - avg_dd * 3, 0, 100))
    severity_score = 0.6 * max_dd_score + 0.4 * avg_dd_score

    # ── Factor 4: Drawdown Frequency (how often do you dip?) ──
    # Count episodes: transitions from non-drawdown to drawdown state
    episodes = 0
    in_dd = False
    for dd in drawdowns:
        if dd > 3.0 and not in_dd:
            episodes += 1
            in_dd = True
        elif dd < 1.0:
            in_dd = False
    # 0 episodes = 100, 1 = 85, 2 = 70, 3 = 55, 4+ = max 40
    frequency_score = float(np.clip(100 - episodes * 15, 40, 100))

    # ── Combined Score ──
    resilience = (
        0.30 * recovery_score +
        0.25 * speed_score +
        0.25 * severity_score +
        0.20 * frequency_score
    )

    return float(np.clip(resilience, 0, 100))


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
        categories = sorted(traj_df['category'].dropna().unique().tolist())
        selected_cats = st.multiselect("Category", categories, default=[], placeholder="All", key='sb_cat')

        # Sector filter (top sectors by count)
        sector_counts = traj_df['sector'].value_counts()
        top_sectors = sector_counts[sector_counts >= 3].index.tolist()
        sectors = sorted(top_sectors)
        selected_sectors = st.multiselect("Sector", sectors, default=[], placeholder="All", key='sb_sector')

        # Industry filter (dynamic — cascades from category/sector selection)
        industry_pool = traj_df
        if selected_cats:
            industry_pool = industry_pool[industry_pool['category'].isin(selected_cats)]
        if selected_sectors:
            industry_pool = industry_pool[industry_pool['sector'].isin(selected_sectors)]
        industries = sorted(industry_pool['industry'].dropna().loc[lambda s: s.str.strip() != ''].unique().tolist())
        selected_industries = st.multiselect("Industry", industries, default=[], placeholder="All", key='sb_industry')

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
        st.caption("v8.0 | Wave Signal Fusion + 18 WAVE Signals")

    return {
        'categories': selected_cats,
        'sectors': selected_sectors,
        'industries': selected_industries,
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
    if filters['categories']:
        df = df[df['category'].isin(filters['categories'])]

    # Sector
    if filters['sectors']:
        df = df[df['sector'].isin(filters['sectors'])]

    # Industry
    if filters.get('industries'):
        df = df[df['industry'].isin(filters['industries'])]

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
                         ('sector', ''), ('return_quality', 50),
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
            'Consistency', 'Return Quality', 'Price Alignment', 'Decay Score', 'Sector Alpha'
        ], key='rank_sort', label_visibility='collapsed')
    with ctl2:
        view_mode = st.selectbox("View", [
            'Standard', 'Compact', 'Signals', 'Trading', 'Complete', 'Custom'
        ], key='rank_view', label_visibility='collapsed')
    with ctl3:
        export_btn = st.button("📥 Export CSV", key='rank_export', use_container_width=True)

    # ── Apply Show Top limit ──
    filtered_df = filtered_df.head(display_n).copy()
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df['t_rank'] = range(1, len(filtered_df) + 1)
    shown = len(filtered_df)

    # ── T-Rank = rank within full universe (stable, never changes with filters) ──
    t_rank_sorted = all_df.sort_values(
        ['trajectory_score', 'confidence', 'consistency'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    t_rank_map = {t: i + 1 for i, t in enumerate(t_rank_sorted['ticker'])}
    filtered_df['t_rank_universe'] = filtered_df['ticker'].map(t_rank_map).fillna(0).astype(int)

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
        'Return Quality': ('return_quality', False),
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
        'T-Rank': ('t_rank_universe', 'T-Rank', 'Rank in full universe (all stocks, unfiltered)',
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
        'RetQuality': ('return_quality', 'RetQuality', 'Return quality component (v6.0)',
                     st.column_config.ProgressColumn('RetQuality', min_value=0, max_value=100, format="%.1f")),
        'Price Signal': ('price_label', 'Price Signal', 'Price alignment: CONFIRMED/DIVERGENT/NEUTRAL', None),
        'Decay':    ('decay_label', 'Decay', 'Momentum decay level: HIGH/MODERATE/MILD/CLEAN', None),
        'Alpha':    ('sector_alpha_tag', 'Alpha', 'Sector alpha classification', None),
        'Trajectory': ('sparkline', 'Trajectory', 'Score trajectory over time',
                       st.column_config.LineChartColumn('Trajectory', y_min=0, y_max=100, width="medium")),
        # v6.3: Advanced Trading Signals
        'Conviction': ('conviction', 'Conviction', 'Buy conviction score 0-100 (higher = stronger BUY signal)',
                       st.column_config.ProgressColumn('Conviction', min_value=0, max_value=100, format="%.0f")),
        'Conv Tag': ('conviction_tag', 'Conv', 'Conviction level: VERY_HIGH/HIGH/MODERATE/LOW/VERY_LOW', None),
        'Risk-Adj': ('risk_adj_score', 'Risk-Adj', 'Risk-adjusted T-Score (volatility penalized)',
                     st.column_config.ProgressColumn('Risk-Adj', min_value=0, max_value=100, format="%.1f")),
        'Exit Risk': ('exit_risk', 'Exit Risk', 'Exit/sell risk score 0-100 (higher = consider selling)',
                      st.column_config.ProgressColumn('Exit Risk', min_value=0, max_value=100, format="%.0f")),
        'Exit Tag': ('exit_tag', 'Exit', 'Exit warning: EXIT_NOW/CAUTION/WATCH/HOLD', None),
        'Hot Streak': ('hot_streak', 'Hot', 'Hot streak detected (4+ weeks improving at high position)', None),
        'Vol Conf': ('vol_confirmed', 'Vol', 'Volume confirmation: STRONG/MODERATE/WEAK/DISTRIBUTION/NEUTRAL', None),
        # v8.0: Wave Signal Fusion
        'Wave':     ('wave_fusion_score', 'Wave', 'Wave Signal Fusion score 0-100 (cross-system validation)',
                     st.column_config.ProgressColumn('Wave', min_value=0, max_value=100, format="%.0f")),
        'WF Label': ('wave_fusion_label', 'WF', 'Wave fusion: STRONG/CONFIRMED/NEUTRAL/WEAK/CONFLICT', None),
        'Confluence': ('wave_confluence', 'Conf', 'Wave-Trajectory confluence agreement 0-100',
                       st.column_config.ProgressColumn('Conf', min_value=0, max_value=100, format="%.0f")),
        'Inst Flow': ('wave_inst_flow', 'Flow', 'Institutional money flow signal 0-100',
                      st.column_config.ProgressColumn('Flow', min_value=0, max_value=100, format="%.0f")),
        'Harmony':  ('wave_harmony', 'Harm', 'Momentum harmony score 0-100',
                     st.column_config.ProgressColumn('Harm', min_value=0, max_value=100, format="%.0f")),
    }

    VIEW_PRESETS = {
        'Compact':  ['T-Rank', 'Ticker', '₹ Price', 'T-Score', 'Grade', 'Pattern',
                     'Δ Total', 'Streak', 'Trajectory'],
        'Standard': ['T-Rank', 'Ticker', 'Company', 'Sector', '₹ Price', 'T-Score', 'Grade',
                     'Pattern', 'Signals', 'TMI', 'Best', 'Δ Total', 'Δ Week', 'Streak', 'Wks', 'Trajectory'],
        'Signals':  ['T-Rank', 'Ticker', 'Company', 'Sector', '₹ Price', 'T-Score', 'Grade',
                     'Pattern', 'Signals', 'Price Signal', 'Decay', 'Alpha', 'Trajectory'],
        'Trading':  ['T-Rank', 'Ticker', 'Company', '₹ Price', 'T-Score', 'Grade', 'Conviction',
                     'Conv Tag', 'Risk-Adj', 'Exit Risk', 'Exit Tag', 'Hot Streak', 'Vol Conf',
                     'Wave', 'WF Label', 'Confluence', 'Inst Flow', 'Streak', 'Trajectory'],
        'Complete': ['T-Rank', 'Ticker', 'Company', 'Sector', 'Category', '₹ Price', 'T-Score',
                     'Grade', 'Pattern', 'Signals', 'TMI', 'Best', 'Δ Total', 'Δ Week', 'Streak', 'Wks',
                     'Trend', 'Velocity', 'Consistency', 'Positional', 'RetQuality', 'Price Signal', 'Decay', 'Alpha', 
                     'Conviction', 'Risk-Adj', 'Exit Risk', 'Hot Streak', 'Vol Conf',
                     'Wave', 'WF Label', 'Confluence', 'Inst Flow', 'Harmony', 'Trajectory'],
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
            selected_cols = ['T-Rank', 'Ticker', 'T-Score', 'Grade']
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
            file_name=f"trajectory_rankings_{metadata.get('last_date','export')}.csv",
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
    """Search & Analyse — v3.0 (Price Trajectory, T-Rank, Latest Price, Clean UI)

    Args:
        filtered_df: Category/sector-filtered stocks (for dropdown).
        traj_df:     Full unfiltered data (for T-Rank against universe).
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
    # T-Rank = rank within full universe (all stocks, not filtered)
    sorted_df = traj_df.sort_values(
        ['trajectory_score', 'confidence', 'consistency'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    t_rank_idx = sorted_df[sorted_df['ticker'] == ticker].index
    t_rank = int(t_rank_idx[0]) + 1 if len(t_rank_idx) > 0 else 0

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
                    <div style="font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">T-Rank</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#58a6ff;">#{t_rank}</div>
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
            'Component': ['Positional', 'Trend', 'Velocity', 'Acceleration', 'Consistency', 'Resilience', 'RetQuality'],
            'Wt': [f"{adp_w[k]*100:.0f}%" for k in ['positional','trend','velocity','acceleration','consistency','resilience','return_quality']],
            'Score': [row['positional'], row['trend'], row['velocity'], row['acceleration'], row['consistency'], row['resilience'], row.get('return_quality', 50)],
            'Contrib': [round(row[k] * adp_w[k], 1) for k in ['positional','trend','velocity','acceleration','consistency','resilience','return_quality']]
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
            <div style="margin-top:6px; color:#484f58; font-size:0.7rem;">Base Score: {row.get('pre_price_score', row['trajectory_score']):.1f}</div>
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
            <div style="margin-top:6px; color:#484f58; font-size:0.7rem;">Unified ×{row.get('combined_mult', 1.0):.3f} → Final: {row['trajectory_score']:.1f}</div>
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

    # ── Wave Signal Fusion Detail ──
    wf_score = row.get('wave_fusion_score', 50)
    wf_label = row.get('wave_fusion_label', 'NEUTRAL')
    wf_mult  = row.get('wave_fusion_multiplier', 1.0)
    wf_conf  = row.get('wave_confluence', 50)
    wf_flow  = row.get('wave_inst_flow', 50)
    wf_harm  = row.get('wave_harmony', 50)
    wf_fund  = row.get('wave_fundamental', 50)
    wf_tension = row.get('wave_position_tension') or 0
    wf_from_low = row.get('wave_from_low') or 0
    wf_colors = {'WAVE_STRONG': '#00E676', 'WAVE_CONFIRMED': '#3fb950', 'WAVE_NEUTRAL': '#484f58',
                 'WAVE_WEAK': '#FF9800', 'WAVE_CONFLICT': '#FF1744'}
    wf_c = wf_colors.get(wf_label, '#484f58')
    wf1, wf2, wf3 = st.columns(3)
    with wf1:
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">🌊 Wave Signal Fusion</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Fusion Score</span>
                <span style="color:{wf_c}; font-weight:700;">{wf_score:.0f}/100</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Multiplier</span>
                <span style="color:{wf_c}; font-weight:600;">×{wf_mult:.3f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Classification</span>
                <span style="color:{wf_c}; font-weight:700;">{wf_label.replace('WAVE_', '')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with wf2:
        _conf_c = '#3fb950' if wf_conf >= 65 else '#FF9800' if wf_conf < 40 else '#8b949e'
        _flow_c = '#3fb950' if wf_flow >= 65 else '#FF9800' if wf_flow < 40 else '#8b949e'
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">📡 Fusion Signals</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Confluence</span>
                <span style="color:{_conf_c}; font-weight:600;">{wf_conf:.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Inst. Flow</span>
                <span style="color:{_flow_c}; font-weight:600;">{wf_flow:.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Harmony</span>
                <span style="color:#8b949e; font-weight:600;">{wf_harm:.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Fundamental</span>
                <span style="color:#8b949e; font-weight:600;">{wf_fund:.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with wf3:
        _tens_c = '#FF1744' if wf_tension > 0.7 else '#FF9800' if wf_tension > 0.4 else '#3fb950'
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">⚡ Supplementary</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Position Tension</span>
                <span style="color:{_tens_c}; font-weight:600;">{wf_tension:.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">From 52w Low</span>
                <span style="color:#8b949e; font-weight:600;">{wf_from_low:.1f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">1D Return</span>
                <span style="color:#8b949e; font-weight:600;">{(row.get('wave_ret_1d') or 0):.2f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">1Y Return</span>
                <span style="color:#8b949e; font-weight:600;">{(row.get('wave_ret_1y') or 0):.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
    """Render radar/spider chart for component scores (7 components, v6.0)"""
    categories = ['Positional', 'Trend', 'Velocity', 'Acceleration', 'Consistency', 'Resilience', 'RetQuality']
    values = [row['positional'], row['trend'], row['velocity'], row['acceleration'],
              row['consistency'], row['resilience'], row.get('return_quality', 50)]
    values_closed = values + [values[0]]  # Close the polygon
    cats_closed = categories + [categories[0]]

    fig = go.Figure()

    # Reference circle at 50
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(cats_closed),
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

        # Precompute T-Rank map (Confidence-Aware Ranking)
        t_rank_sorted = traj_df.sort_values(
            ['trajectory_score', 'confidence', 'consistency'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        t_rank_map = {t: i + 1 for i, t in enumerate(t_rank_sorted['ticker'])}
        total_stocks = len(t_rank_sorted)

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
                t_rank = t_rank_map.get(r['ticker'], 0)

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
                            <div><span style="color:#6e7681; font-size:0.65rem;">T-Rank</span><br><span style="color:#58a6ff; font-weight:700;">#{t_rank}</span></div>
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
# STRATEGY BACKTEST ENGINE (v8.0)
# ============================================
# Walk-forward backtest that proves which selection strategy
# generates the best ACTUAL returns using your CSV data.
# Tests 8 strategies from "buy everything" to "max filter".
# Uses price-based forward returns — no lookahead bias.

def _run_strategy_backtest(uploaded_files, progress_callback=None):
    """
    Walk-forward strategy backtest.

    For each week N (where N >= min_history):
      1. Build histories using ONLY weeks 1..N (no future data)
      2. Compute trajectory scores at that point
      3. Apply 8 different selection strategies
      4. Measure ACTUAL forward returns from week N+1 prices

    Returns dict with results per strategy, or None if insufficient data.
    """
    # ── Step 1: Parse all CSVs ──
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
            logger.warning(f"Backtest: Failed to load {ufile.name}: {e}")

    dates = sorted(weekly_data.keys())
    n_weeks = len(dates)
    if n_weeks < 7:  # Need 5 history + 1 test + 1 forward
        return None

    min_history = 5  # Minimum weeks of data before first test

    strategy_names = [
        'S1: Universe Avg',
        'S2: Top 10 Rank',
        'S3: Top 20 Rank',
        'S4: Persistent Top 50',
        'S5: T-Score ≥ 70',
        'S6: T-Score ≥ 70 + No Decay',
        'S7: Conviction ≥ 65',
        'S8: Full Signal',
    ]
    all_results = {name: [] for name in strategy_names}

    # ── Step 2: Incremental walk-forward ──
    histories = {}  # Built incrementally — grows each week
    total_windows = len(dates) - min_history - 1

    for week_idx, date in enumerate(dates):
        df = weekly_data[date]
        total = len(df)

        # ── Add this week's data to histories ──
        for _, row in df.iterrows():
            ticker = str(row.get('ticker', '')).strip()
            if not ticker or ticker == 'nan':
                continue

            if ticker not in histories:
                histories[ticker] = {
                    'dates': [], 'ranks': [], 'scores': [], 'prices': [],
                    'total_per_week': [],
                    'trend_qualities': [], 'market_states': [], 'pattern_history': [],
                    'ret_7d': [], 'ret_30d': [], 'ret_3m': [], 'ret_6m': [],
                    'from_high_pct': [], 'momentum_score': [], 'volume_score': [],
                    'position_score': [], 'acceleration_score': [], 'breakout_score': [],
                    'rvol_score': [], 'pe': [], 'eps_current': [], 'eps_change_pct': [],
                    'from_low_pct': [], 'ret_1d': [], 'ret_1y': [],
                    'rvol': [], 'vmi': [], 'money_flow_mm': [],
                    'position_tension': [], 'momentum_harmony': [],
                    'eps_tier': [], 'pe_tier': [], 'overall_market_strength': [],
                    'company_name': '', 'category': '', 'sector': '',
                    'industry': '', 'market_state': '', 'patterns': ''
                }

            h = histories[ticker]
            h['dates'].append(date.strftime('%Y-%m-%d'))
            h['ranks'].append(float(row['rank']) if pd.notna(row['rank']) else total)
            h['scores'].append(float(row['master_score']))
            h['prices'].append(float(row['price']))
            h['total_per_week'].append(total)

            tq_val = row.get('trend_quality', 0)
            h['trend_qualities'].append(float(tq_val) if pd.notna(tq_val) else 0)
            ms_val = row.get('market_state', '')
            h['market_states'].append(str(ms_val).strip() if pd.notna(ms_val) else '')
            pat_val = row.get('patterns', '')
            h['pattern_history'].append(str(pat_val).strip() if pd.notna(pat_val) else '')

            for col_name, hist_key in [
                ('ret_7d', 'ret_7d'), ('ret_30d', 'ret_30d'), ('ret_3m', 'ret_3m'),
                ('ret_6m', 'ret_6m'), ('from_high_pct', 'from_high_pct'),
                ('momentum_score', 'momentum_score'), ('volume_score', 'volume_score'),
                ('position_score', 'position_score'), ('acceleration_score', 'acceleration_score'),
                ('breakout_score', 'breakout_score'), ('rvol_score', 'rvol_score'),
                ('pe', 'pe'), ('eps_current', 'eps_current'),
                ('eps_change_pct', 'eps_change_pct'), ('from_low_pct', 'from_low_pct'),
                ('ret_1d', 'ret_1d'), ('ret_1y', 'ret_1y'),
                ('rvol', 'rvol'), ('vmi', 'vmi'),
                ('money_flow_mm', 'money_flow_mm'), ('position_tension', 'position_tension'),
                ('momentum_harmony', 'momentum_harmony'),
                ('overall_market_strength', 'overall_market_strength'),
            ]:
                col_val = row.get(col_name, None)
                if col_val is not None and pd.notna(col_val):
                    try:
                        h[hist_key].append(float(col_val))
                    except (ValueError, TypeError):
                        h[hist_key].append(float('nan'))
                else:
                    h[hist_key].append(float('nan'))

            for fld in ['company_name', 'category', 'sector', 'industry', 'market_state', 'patterns']:
                val = row.get(fld, '')
                if pd.notna(val) and str(val).strip():
                    h[fld] = str(val).strip()

            for tier_col in ['eps_tier', 'pe_tier']:
                tier_val = row.get(tier_col, '')
                h[tier_col].append(str(tier_val).strip() if pd.notna(tier_val) else '')

        # ── Only test when we have enough history AND a next week exists ──
        if week_idx < min_history or week_idx >= len(dates) - 1:
            continue

        window_num = week_idx - min_history
        if progress_callback:
            progress_callback(window_num / max(total_windows, 1),
                              f"Testing week {date.strftime('%Y-%m-%d')} ({window_num + 1}/{total_windows})")

        forward_date = dates[week_idx + 1]
        forward_df = weekly_data[forward_date]

        # ── Compute PRICE-BASED forward returns (ground truth) ──
        cur_prices = {}
        for _, row in df.iterrows():
            t = str(row['ticker']).strip()
            p = float(row['price']) if pd.notna(row['price']) and float(row['price']) > 0 else 0
            if p > 0:
                cur_prices[t] = p

        fwd_prices = {}
        for _, row in forward_df.iterrows():
            t = str(row['ticker']).strip()
            p = float(row['price']) if pd.notna(row['price']) and float(row['price']) > 0 else 0
            if p > 0:
                fwd_prices[t] = p

        forward_rets = {}
        for t in cur_prices:
            if t in fwd_prices:
                forward_rets[t] = (fwd_prices[t] / cur_prices[t] - 1) * 100

        if not forward_rets:
            continue

        # ── RAW strategies (no trajectory computation needed) ──
        # S1: Universe average — all stocks with valid forward returns
        s1_tickers = list(forward_rets.keys())

        # S2: Top 10 by WAVE rank
        s2_tickers = df.nsmallest(10, 'rank')['ticker'].astype(str).str.strip().tolist()

        # S3: Top 20 by rank (broader basket — tests concentration vs diversification)
        s3_tickers = df.nsmallest(20, 'rank')['ticker'].astype(str).str.strip().tolist()

        # S4: Persistent top 50 — rank in top 50 for last 4 consecutive weeks
        s4_tickers = []
        for t, h_dict in histories.items():
            ranks_list = h_dict['ranks']
            totals_list = h_dict['total_per_week']
            if len(ranks_list) >= 4:
                last4_pcts = [(1 - ranks_list[-(i+1)] / max(totals_list[-(i+1)], 1)) * 100
                              for i in range(4)]
                if all(p >= 97.5 for p in last4_pcts):  # Top ~50 out of ~2100
                    s4_tickers.append(t)
        # Relax to top 100 if too few
        if len(s4_tickers) < 5:
            for t, h_dict in histories.items():
                ranks_list = h_dict['ranks']
                totals_list = h_dict['total_per_week']
                if len(ranks_list) >= 4 and t not in s4_tickers:
                    last4_pcts = [(1 - ranks_list[-(i+1)] / max(totals_list[-(i+1)], 1)) * 100
                                  for i in range(4)]
                    if all(p >= 95 for p in last4_pcts):
                        s4_tickers.append(t)

        # ── TRAJECTORY strategies (need full computation) ──
        traj = {}
        for ticker, h_dict in histories.items():
            if len(h_dict['ranks']) >= 2:
                try:
                    traj[ticker] = _compute_single_trajectory(h_dict)
                except Exception:
                    pass

        # S5: T-Score >= 70
        s5_tickers = [t for t, r in traj.items() if r['trajectory_score'] >= 70]

        # S6: T-Score >= 70 + No Decay (trap filter)
        s6_tickers = [t for t, r in traj.items()
                      if r['trajectory_score'] >= 70
                      and r.get('decay_label', '') not in ('DECAY_HIGH', 'DECAY_MODERATE')]

        # S7: Conviction >= 65 (HIGH or above)
        s7_tickers = [t for t, r in traj.items() if r.get('conviction', 0) >= 65]

        # S8: Full Signal — max filtering
        s8_tickers = [t for t, r in traj.items()
                      if r['trajectory_score'] >= 70
                      and r.get('conviction', 0) >= 65
                      and r.get('decay_label', '') not in ('DECAY_HIGH', 'DECAY_MODERATE')
                      and r.get('wave_fusion_label', 'WAVE_NEUTRAL') in ('WAVE_CONFIRMED', 'WAVE_STRONG')]

        # ── Measure forward returns for each strategy ──
        strategy_picks = {
            'S1: Universe Avg': s1_tickers,
            'S2: Top 10 Rank': s2_tickers,
            'S3: Top 20 Rank': s3_tickers,
            'S4: Persistent Top 50': s4_tickers,
            'S5: T-Score ≥ 70': s5_tickers,
            'S6: T-Score ≥ 70 + No Decay': s6_tickers,
            'S7: Conviction ≥ 65': s7_tickers,
            'S8: Full Signal': s8_tickers,
        }

        week_label = date.strftime('%Y-%m-%d')

        for sname, tickers in strategy_picks.items():
            valid_rets = [forward_rets[t] for t in tickers if t in forward_rets]
            avg_ret = float(np.mean(valid_rets)) if valid_rets else 0.0
            med_ret = float(np.median(valid_rets)) if valid_rets else 0.0
            all_results[sname].append({
                'week': week_label,
                'forward_week': forward_date.strftime('%Y-%m-%d'),
                'avg_return': avg_ret,
                'median_return': med_ret,
                'n_stocks': len(valid_rets),
                'n_positive': sum(1 for r in valid_rets if r > 0),
                'best': max(valid_rets) if valid_rets else 0,
                'worst': min(valid_rets) if valid_rets else 0,
            })

    if progress_callback:
        progress_callback(1.0, "Backtest complete!")

    return all_results, dates


# ============================================
# UI: BACKTEST TAB
# ============================================

def render_backtest_tab(uploaded_files):
    """Render strategy backtest validation tab"""
    st.markdown("### 📊 Strategy Backtest — Walk-Forward Validation")
    st.markdown("""
    <div style="background:#161b22; border-radius:10px; padding:16px; border:1px solid #30363d; margin-bottom:16px;">
        <div style="font-size:0.85rem; color:#8b949e;">
            <b>What this does:</b> For each week, it computes scores using ONLY past data,
            then checks what ACTUALLY happened to the selected stocks next week.
            No lookahead bias — pure walk-forward validation using real price returns.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check cache
    bt_cache_key = tuple(sorted((f.name, f.size) for f in uploaded_files))
    cached = st.session_state.get('_bt_result')
    cached_key = st.session_state.get('_bt_key')

    if cached is not None and cached_key == bt_cache_key:
        bt_results, bt_dates = cached
    else:
        bt_results = None

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    with col_info:
        if bt_results is None:
            st.caption("Click Run to test 8 strategies against actual forward returns")
        else:
            st.caption("✅ Backtest results loaded. Click Run to refresh.")

    if run_btn:
        progress_bar = st.progress(0, text="Initializing backtest...")

        def _progress(pct, text):
            progress_bar.progress(min(pct, 1.0), text=text)

        with st.spinner("Running walk-forward backtest..."):
            result = _run_strategy_backtest(uploaded_files, progress_callback=_progress)

        progress_bar.empty()

        if result is None:
            st.error("❌ Need at least 7 weeks of CSV data for backtest.")
            return

        bt_results, bt_dates = result
        st.session_state['_bt_result'] = (bt_results, bt_dates)
        st.session_state['_bt_key'] = bt_cache_key
        st.rerun()

    if bt_results is None:
        return

    # ── Build Summary Statistics ──
    summary_rows = []
    for sname, weeks in bt_results.items():
        if not weeks:
            continue
        returns = [w['avg_return'] for w in weeks]
        n_weeks_tested = len(returns)
        avg_weekly = np.mean(returns)
        cumulative = np.prod([1 + r / 100 for r in returns]) * 100 - 100
        win_rate = sum(1 for r in returns if r > 0) / max(n_weeks_tested, 1) * 100
        avg_stocks = np.mean([w['n_stocks'] for w in weeks])
        std_ret = np.std(returns) if len(returns) > 1 else 0.001
        sharpe = avg_weekly / max(std_ret, 0.001) * np.sqrt(52)  # Annualized
        worst_week = min(returns)
        best_week = max(returns)

        summary_rows.append({
            'Strategy': sname,
            'Weeks Tested': n_weeks_tested,
            'Avg Stocks/Wk': round(avg_stocks, 0),
            'Avg Wk Return %': round(avg_weekly, 2),
            'Cumulative %': round(cumulative, 2),
            'Win Rate %': round(win_rate, 1),
            'Sharpe (Ann.)': round(sharpe, 2),
            'Best Wk %': round(best_week, 2),
            'Worst Wk %': round(worst_week, 2),
        })

    if not summary_rows:
        st.warning("No test windows available. Need more week coverage.")
        return

    summary_df = pd.DataFrame(summary_rows)

    # ── Key Insight ──
    universe_cum = summary_df.loc[summary_df['Strategy'] == 'S1: Universe Avg', 'Cumulative %']
    universe_val = float(universe_cum.iloc[0]) if len(universe_cum) > 0 else 0

    best_strat = summary_df.loc[summary_df['Cumulative %'].idxmax()]
    worst_strat = summary_df.loc[summary_df['Cumulative %'].idxmin()]
    best_name = best_strat['Strategy']
    best_cum = best_strat['Cumulative %']
    best_vs_uni = best_cum - universe_val

    if best_vs_uni > 0:
        insight_color = '#3fb950'
        insight_icon = '✅'
        insight_text = f"**{best_name}** generated **+{best_cum:.1f}%** cumulative return, beating the universe by **+{best_vs_uni:.1f}%**"
    else:
        insight_color = '#FF9800'
        insight_icon = '⚠️'
        insight_text = f"No strategy beat the universe average ({universe_val:.1f}%). Best: **{best_name}** at {best_cum:.1f}%"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #0d1117, #161b22); border-radius:12px;
                padding:20px; border:2px solid {insight_color}; margin-bottom:20px;">
        <div style="font-size:1.2rem; font-weight:800; color:{insight_color}; margin-bottom:8px;">
            {insight_icon} KEY FINDING
        </div>
        <div style="font-size:0.95rem; color:#e6edf3;">
            {insight_text}
        </div>
        <div style="margin-top:8px; font-size:0.75rem; color:#484f58;">
            Based on {summary_rows[0]['Weeks Tested']} walk-forward test windows with no lookahead bias
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Strategy Performance Table ──
    st.markdown("#### 📋 Strategy Comparison")

    # Color the key columns
    def _highlight_returns(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #3fb950; font-weight: 700'
            elif val < 0:
                return 'color: #ff7b72; font-weight: 700'
        return ''

    styled_df = summary_df.style.applymap(
        _highlight_returns,
        subset=['Avg Wk Return %', 'Cumulative %', 'Best Wk %', 'Worst Wk %']
    ).format({
        'Avg Wk Return %': '{:+.2f}',
        'Cumulative %': '{:+.2f}',
        'Win Rate %': '{:.0f}',
        'Sharpe (Ann.)': '{:.2f}',
        'Best Wk %': '{:+.2f}',
        'Worst Wk %': '{:+.2f}',
        'Avg Stocks/Wk': '{:.0f}',
    })
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── Cumulative Return Chart ──
    st.markdown("#### 📈 Cumulative Return Over Time")

    fig = go.Figure()
    strat_colors = {
        'S1: Universe Avg': '#484f58',
        'S2: Top 10 Rank': '#8b949e',
        'S3: Top 20 Rank': '#79c0ff',
        'S4: Persistent Top 50': '#d2a8ff',
        'S5: T-Score ≥ 70': '#ffa657',
        'S6: T-Score ≥ 70 + No Decay': '#f0883e',
        'S7: Conviction ≥ 65': '#3fb950',
        'S8: Full Signal': '#FFD700',
    }

    for sname, weeks in bt_results.items():
        if not weeks:
            continue
        cum_vals = []
        cum = 100  # Start at 100
        x_dates = []
        for w in weeks:
            cum = cum * (1 + w['avg_return'] / 100)
            cum_vals.append(cum)
            x_dates.append(w['forward_week'])

        line_width = 3 if sname in ('S1: Universe Avg', best_name) else 1.5
        dash = 'dash' if sname == 'S1: Universe Avg' else None

        fig.add_trace(go.Scatter(
            x=x_dates, y=cum_vals,
            name=sname,
            mode='lines+markers',
            marker=dict(size=5),
            line=dict(color=strat_colors.get(sname, '#8b949e'), width=line_width, dash=dash),
            hovertemplate=f'{sname}<br>Week: %{{x}}<br>Value: %{{y:.1f}}<br>Return: %{{customdata:+.2f}}%<extra></extra>',
            customdata=[w['avg_return'] for w in weeks],
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=450,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.35, xanchor='center', x=0.5,
                    font=dict(size=10)),
        yaxis=dict(title='Portfolio Value (₹100 start)', gridcolor='#21262d'),
        xaxis=dict(title='Week', gridcolor='#21262d'),
        hovermode='x unified',
    )
    fig.add_hline(y=100, line_dash='dot', line_color='#484f58', opacity=0.5)
    st.plotly_chart(fig, use_container_width=True, key='bt_cumulative_chart')

    # ── Weekly Breakdown ──
    with st.expander("📅 Weekly Breakdown", expanded=False):
        # Create a combined table: rows = weeks, columns = strategies
        if bt_results:
            first_strat = list(bt_results.keys())[0]
            week_labels = [w['week'] for w in bt_results[first_strat]]

            breakdown_data = {'Decision Week': week_labels}
            for sname in bt_results:
                breakdown_data[sname] = [f"{w['avg_return']:+.2f}% ({w['n_stocks']})"
                                         for w in bt_results[sname]]

            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # ── Excess Return vs Universe ──
    st.markdown("#### 🎯 Excess Return vs Universe (Alpha)")

    fig2 = go.Figure()
    universe_returns = [w['avg_return'] for w in bt_results.get('S1: Universe Avg', [])]

    for sname, weeks in bt_results.items():
        if sname == 'S1: Universe Avg' or not weeks:
            continue
        excess = [w['avg_return'] - ur for w, ur in zip(weeks, universe_returns)]
        x_dates = [w['forward_week'] for w in weeks]
        avg_excess = np.mean(excess) if excess else 0

        colors = ['#3fb950' if e >= 0 else '#ff7b72' for e in excess]

        fig2.add_trace(go.Bar(
            x=x_dates, y=excess,
            name=f"{sname} (avg: {avg_excess:+.2f}%)",
            marker_color=strat_colors.get(sname, '#8b949e'),
            opacity=0.7,
            visible=True if sname == best_name else 'legendonly',
            hovertemplate=f'{sname}<br>Week: %{{x}}<br>Alpha: %{{y:+.2f}}%<extra></extra>',
        ))

    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=350,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.4, xanchor='center', x=0.5,
                    font=dict(size=10)),
        yaxis=dict(title='Excess Return vs Universe %', gridcolor='#21262d', zeroline=True,
                   zerolinecolor='#484f58'),
        xaxis=dict(title='Week', gridcolor='#21262d'),
        barmode='group',
    )
    st.plotly_chart(fig2, use_container_width=True, key='bt_alpha_chart')

    # ── Strategy Descriptions ──
    with st.expander("📖 Strategy Definitions", expanded=False):
        st.markdown("""
        | Strategy | Selection Rule | Tests |
        |----------|---------------|-------|
        | **S1: Universe Avg** | Buy all stocks equally | Baseline — market average |
        | **S2: Top 10 Rank** | 10 lowest WAVE rank numbers | Does raw ranking predict? |
        | **S3: Top 20 Rank** | 20 lowest WAVE rank numbers | Does broader selection reduce risk? |
        | **S4: Persistent Top 50** | Top ~2.5% for 4+ consecutive weeks | Does persistence add value? |
        | **S5: T-Score ≥ 70** | Full Trajectory Engine score ≥ 70 | Does the 7-component system work? |
        | **S6: T-Score ≥ 70 + No Decay** | S5 + no momentum decay trap | Does trap detection help? |
        | **S7: Conviction ≥ 65** | Multi-signal conviction score ≥ 65 | Does conviction predict returns? |
        | **S8: Full Signal** | T-Score ≥ 70 + Conviction ≥ 65 + No Decay + WAVE Confirmed/Strong | Does max filtering produce best results? |

        **Forward Return:** Actual price change from current week to next week. Computed from price data, not ret_7d.

        **Walk-Forward:** At each test point, only data available UP TO that week is used. No future information leaks into the scoring.
        """)


# ============================================
# UI: ABOUT TAB
# ============================================

def render_about_tab():
    """Render about/documentation tab"""

    st.markdown("""
    ## 📊 Rank Trajectory Engine v6.3 — Advanced Trading Signals

    The **ALL TIME BEST** stock rank trajectory analysis system with **7-component adaptive scoring**,
    **signal-isolated return quality**, **directional price-rank alignment**, **momentum decay warning**,
    and **sector alpha detection**.

    ---

    ### 🧠 The Architecture: Signal-Isolated Pipeline

    ```
    7-Component Adaptive Scoring → Elite Dominance Bonus → Bayesian Shrinkage
        → Hurst Persistence × Directional Price-Rank Alignment
        → Momentum Decay Penalty → Sector Alpha Tag
    ```

    **SIGNAL ISOLATION PRINCIPLE:** Each data source enters through exactly ONE scoring path.
    No signal leakage — return magnitude is scored only in ReturnQuality component.

    | Layer | What It Does | Impact |
    |---|---|---|
    | **7 Components** | Weighted scoring by position tier (7 dimensions) | 100% of base score |
    | **Elite Bonus** | Sustained top-tier → guaranteed score floor | Top 3% for 60% weeks → floor 88 |
    | **Bayesian Shrinkage** | Short-history stocks pulled toward neutral | 4 weeks → 75% shrunk |
    | **Hurst Multiplier** | Persistent trends boosted, mean-reverting penalized | ×0.94 to ×1.06 |
    | **Price-Rank Alignment** | DIRECTIONAL agreement only — sign(return) vs sign(rank Δ) | ×0.88 to ×1.08 |
    | **Momentum Decay** | Catches stocks with good rank but deteriorating returns | ×0.93 to ×1.00 |
    | **Sector Alpha** | Separates leaders from sector-beta riders | Tag: LEADER / BETA / LAGGARD |

    ---

    ### 💰 Directional Price-Rank Alignment (v6.1)

    Measures whether return **direction** agrees with rank movement.
    Does NOT score return **magnitude** — that is ReturnQuality's job.

    #### Two Directional Signals

    | Signal | Weight | What It Measures |
    |--------|--------|------------------|
    | **Weekly Direction** | 55% | Does sign(EMA ret_7d) match sign(percentile change)? |
    | **Monthly Direction** | 45% | Does sign(ret_30d) match sign(percentile change)? |

    #### Multiplier Range

    | Alignment Score | Multiplier | Label | Meaning |
    |----------------|-----------|-------|---------|
    | **72-100** | ×1.04 — ×1.08 | 💰 CONFIRMED | Return direction validates rank trajectory |
    | **50-72** | ×1.00 — ×1.04 | NEUTRAL | Inconclusive |
    | **35-50** | ×0.96 — ×1.00 | NEUTRAL | Mild concern |
    | **0-35** | ×0.88 — ×0.96 | ⚠️ DIVERGENT | Return direction contradicts rank |

    ---

    ### 🔻 Momentum Decay Warning (v6.1)

    **The Problem:** 11.4% of top-10% stocks have negative 30-day returns.
    These are TRAP stocks — ranked well based on PAST momentum that has now faded.

    #### 4 Decay Signals

    | Signal | What It Checks | Max Points |
    |--------|---------------|------------|
    | **Weekly Return** | ret_7d < -5% → 30 pts, < -2% → 15 pts | 30 |
    | **30-Day Return** | Top stock + ret_30d < -15% → 40 pts (THE TRAP!) | 40 |
    | **From High** | from_high_pct < -20% on ranked stock → 20 pts | 20 |
    | **Consecutive Negative** | 3+ weeks of ret_7d < -1% → 15 pts | 15 |

    #### v6.1 Improvements
    - **Proven Winner Exemption:** Uses actual ret_6m (not rank proxy):
      ret_6m > 60% → 55% score reduction, > 30% → 40%, > 15% → 25%
    - **Smooth multiplier:** Continuous curve instead of step-function cliffs.
      Score 0 → ×1.00, Score 35 → ×0.976, Score 60 → ×0.958, Score 100 → ×0.93

    ---

    ### 🏛️ Sector Alpha Check

    | Z-Score | Classification | Icon | Meaning |
    |---------|---------------|------|---------|
    | **> 1.5** | SECTOR_LEADER | 👑 | Genuine alpha — outperforms sector significantly |
    | **0.5 - 1.5** | SECTOR_OUTPERFORM | ⬆️ | Above sector average |
    | **-0.5 - 0.5** | SECTOR_ALIGNED | ➖ | Moving with sector |
    | **-1.0 - -0.5** (hot sector) | SECTOR_BETA | 🏷️ | Riding sector wave, not genuine alpha |
    | **< -1.0** | SECTOR_LAGGARD | 📉 | Below sector average |

    ---

    ### � Advanced Trading Signals (v6.3)

    Five new signals designed for actionable trading decisions:

    #### 1. Conviction Score (0-100)
    Aggregates 5 bullish signals into a single BUY confidence metric.

    | Signal | Max Points | Measures |
    |--------|-----------|----------|
    | Price-Rank Alignment | 25 | CONFIRMED=25, NEUTRAL=10 |
    | Return Quality | 20 | ≥75=20, ≥60=12, ≥50=5 |
    | Data Confidence | 20 | ≥0.85=20, ≥0.6=12, ≥0.4=6 |
    | Momentum (TMI) | 20 | ≥70=20, ≥60=12, ≥50=5 |
    | Positional Strength | 15 | ≥90th=15, ≥80th=10, ≥70th=5 |

    | Tag | Score | Emoji |
    |-----|-------|-------|
    | VERY_HIGH | ≥ 80 | 🎯 |
    | HIGH | ≥ 65 | ✅ |
    | MODERATE | ≥ 45 | ⚡ |
    | LOW | ≥ 25 | ⚠️ |
    | VERY_LOW | < 25 | ❌ |

    #### 2. Risk-Adjusted T-Score
    `Risk-Adj = T-Score / (1 + rank_volatility / 50)`
    Penalizes high-volatility stocks. Rank vol of 25 → 1.5× divisor.

    #### 3. Exit Warning System (0-100)
    Detects when to SELL existing holdings. Aggregates 5 exit signals:

    | Signal | Max Points | Trigger |
    |--------|-----------|---------|
    | TMI Collapse | 25 | TMI < 40 while score > 50 |
    | Price Divergence | 25 | PRICE_DIVERGENT label |
    | Momentum Decay | 30 | DECAY_HIGH or DECAY_MODERATE |
    | Negative Streak | 20 | 3+ consecutive rank drops |
    | Pattern Warning | 20 | fading / crash / topping_out |

    | Tag | Score | Action |
    |-----|-------|--------|
    | 🚨 EXIT_NOW | ≥ 60 | Strong sell signal |
    | ⚠️ CAUTION | ≥ 40 | Review position |
    | 👀 WATCH | ≥ 20 | Monitor closely |
    | ✅ HOLD | < 20 | Position is safe |

    #### 4. Hot Streak Detection
    Flags stocks with sustained momentum + high position:
    - 4+ consecutive improving weeks AND ≥ 70th percentile
    - 3+ weeks AND ≥ 80th percentile
    - 5+ weeks AND ≥ 60th percentile

    #### 5. Volume Confirmation
    Validates rank moves with Wave Engine volume score:

    | Condition | Label | Meaning |
    |-----------|-------|---------|
    | Improving + Vol ≥ 70 | STRONG | High-conviction rank improvement |
    | Improving + Vol ≥ 50 | MODERATE | Decent volume support |
    | Improving + Vol < 30 | WEAK | Low volume → may reverse |
    | Falling + Vol ≥ 70 | DISTRIBUTION | Selling pressure detected |

    ---

    ### �🏗️ Adaptive Weight System (7 Components)

    Weights **dynamically shift** based on the stock's average percentile.
    Smooth interpolation between tiers — no hard cutoffs.

    | Tier | Avg Pctl | Positional | Trend | Velocity | Accel | Consistency | Resilience | RetQuality |
    |------|----------|------------|-------|----------|-------|-------------|------------|------------|
    | **Elite** | > 90% | **40%** | 10% | 7% | 4% | 16% | 10% | 13% |
    | **Strong** | 70-90% | 28% | 16% | 10% | 7% | 14% | 12% | 13% |
    | **Mid** | 40-70% | 15% | 19% | **17%** | 10% | 12% | 12% | 15% |
    | **Bottom** | < 40% | 8% | 17% | **21%** | **15%** | 10% | 12% | **17%** |

    ---

    ### 📈 Return Quality Component (v6.0+)

    Dedicated 7th component — ALL return data enters through this single door.

    | Sub-Signal | Weight | Data Source | What It Scores |
    |-----------|--------|-------------|----------------|
    | **3-Month Return** | 30% | ret_3m | Medium-term momentum quality |
    | **6-Month Return** | 30% | ret_6m | Institutional horizon confirmation |
    | **Short-Term Momentum** | 20% | ret_7d + ret_30d | Recency signal (7d=35%, 30d=65%) |
    | **Return Health** | 20% | Cross-timeframe + from_high | Agreement + correction distance |

    Score range: 0-100 where 50=neutral, ≥75=🔥 strong, ≤30=💧 weak

    ---

    ### 🛡️ Elite Dominance Bonus

    | Tier | Percentile | Required Duration | Score Floor |
    |------|-----------|------------------|-------------|
    | Top 3% | > 97th | ≥ 60% of weeks | **88** |
    | Top 5% | > 95th | ≥ 60% of weeks | **82** |
    | Top 10% | > 90th | ≥ 60% of weeks | **73** |
    | Top 20% | > 85th | ≥ 55% of weeks | **65** |

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

    ### 🌊 Wave Signal Fusion (v8.0)

    The **Wave Signal Fusion Engine** cross-validates 18 WAVE Detection columns with the
    Trajectory Engine's own scoring, producing a fusion multiplier (×0.92 — ×1.10).

    | Signal | Weight | What It Measures |
    |--------|--------|-----------------|
    | **Confluence** | 35% | Agreement between WAVE Detection scores (position, acceleration, breakout) and Trajectory components |
    | **Institutional Flow** | 30% | Money flow, VMI, relative volume, overall market strength |
    | **Harmony** | 20% | Momentum harmony score from WAVE Detection |
    | **Fundamental Quality** | 15% | EPS growth, PE reasonableness, EPS tier (quality gate, not driver) |

    **Classifications:**

    | Label | Score | Multiplier | Meaning |
    |-------|-------|-----------|---------|
    | 🌊 **STRONG** | ≥ 72 | ×1.06 — ×1.10 | Both systems strongly agree — high conviction |
    | ✅ **CONFIRMED** | 58 — 71 | ×1.01 — ×1.05 | Cross-system confirmation |
    | ➖ **NEUTRAL** | 42 — 57 | ×0.99 — ×1.01 | No strong signal either way |
    | ⚠️ **WEAK** | 30 — 41 | ×0.95 — ×0.98 | Mild disagreement — caution |
    | 🔇 **CONFLICT** | < 30 | ×0.92 — ×0.94 | Systems fundamentally disagree — danger |

    ---

    ### 📡 Signal Tags (v6.1)

    The **Signals** column in rankings combines multiple indicators:

    | Icon | Signal | Meaning |
    |------|--------|---------|
    | 💰 | Price Confirmed | Return direction validates rank trajectory |
    | ⚠️ | Price Divergent | Return direction contradicts rank |
    | 🔥 | Strong Returns | ReturnQuality ≥ 75 — strong return profile |
    | 💧 | Weak Returns | ReturnQuality ≤ 30 — weak return profile |
    | 🔻 | Decay High | Good rank but severely negative returns — TRAP! |
    | ⚡ | Decay Moderate | Moderate momentum decay warning |
    | 👑 | Sector Leader | Genuine alpha — significantly above sector average |
    | 🏷️ | Sector Beta | Riding hot sector, not genuine alpha |
    | 📉 | Sector Laggard | Below sector average |
    | 🌊 | Wave Strong | Both WAVE Detection and Trajectory agree — high fusion score |
    | 🔇 | Wave Conflict | WAVE Detection and Trajectory disagree — systems in conflict |

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

    *Built for the Wave Detection ecosystem • v8.0 • Wave Signal Fusion • March 2026*
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
            help="Upload your Wave Detection weekly CSV exports (Stocks_Weekly_YYYY-MM-DD_*_data.csv)"
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
        4. Files should be named: `Stocks_Weekly_YYYY-MM-DD_Month_Year_data.csv`
        5. Minimum **3 weeks** recommended for meaningful trajectory analysis
        """)
        return

    st.caption(f"📁 {len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''} uploaded")

    # ── Session-state caching (recompute only when files change) ──
    cache_key = tuple(sorted((f.name, f.size) for f in uploaded_files))
    if st.session_state.get('_traj_key') != cache_key:
        with st.spinner("📊 Computing trajectories across all weeks..."):
            try:
                result = load_and_compute(uploaded_files)
            except Exception as e:
                st.error(f"❌ Computation error: {e}")
                logger.exception("load_and_compute failed")
                return
        st.session_state['_traj_key'] = cache_key
        st.session_state['_traj_result'] = result

    result = st.session_state.get('_traj_result')
    if result is None or not isinstance(result, (tuple, list)) or len(result) != 4:
        st.error("❌ Invalid computation result. Please re-upload your files.")
        # Clear stale cache so next rerun recomputes
        st.session_state.pop('_traj_key', None)
        st.session_state.pop('_traj_result', None)
        return

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
    tab_ranking, tab_search, tab_funnel, tab_backtest, tab_alerts, tab_export, tab_about = st.tabs([
        "🏆 Rankings", "🔍 Search & Analyze", "🎯 Funnel", "📊 Backtest",
        "🚨 Alerts", "📤 Export", "ℹ️ About"
    ])

    with tab_ranking:
        render_rankings_tab(filtered_df, traj_df, histories, metadata)

    with tab_search:
        render_search_tab(filtered_df, traj_df, histories, dates_iso)

    with tab_funnel:
        render_funnel_tab(filtered_df, traj_df, histories, metadata)

    with tab_backtest:
        render_backtest_tab(uploaded_files)

    with tab_alerts:
        render_alerts_tab(filtered_df, histories)

    with tab_export:
        render_export_tab(filtered_df, traj_df, histories)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
