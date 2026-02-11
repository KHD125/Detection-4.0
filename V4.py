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

# Return-Based Price-Rank Alignment Configuration (v2.3 — uses ret_7d/ret_30d)
PRICE_ALIGNMENT = {
    'noise_band_stable': 2.0,        # Ignore rank moves < this for stable stocks
    'noise_band_normal': 1.0,        # Ignore rank moves < this for normal stocks
    'min_weeks': 4,                  # Minimum weeks needed for alignment calculation
    'multiplier_max_boost': 1.08,    # Maximum upward multiplier
    'multiplier_max_penalty': 0.92,  # Maximum downward multiplier
    'confirmed_threshold': 72,       # Alignment score above this = PRICE_CONFIRMED
    'divergent_threshold': 35,       # Alignment score below this = PRICE_DIVERGENT
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
    'stage1_patterns': ['rocket', 'breakout'],
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
PATTERN_DEFS = {
    'rocket':         ('🚀', 'Rocket',         'Rapid strong improvement across all dimensions'),
    'breakout':       ('⚡', 'Breakout',        'Sudden significant rank jump beyond normal variance'),
    'stable_elite':   ('🎯', 'Stable Elite',    'Consistently top-ranked with low volatility'),
    'at_peak':        ('🏔️', 'At Peak',         'Currently at or near all-time best rank'),
    'steady_climber': ('📈', 'Steady Climber',  'Gradual but consistent rank improvement'),
    'recovery':       ('🔄', 'Recovery',        'Bouncing back from rank deterioration'),
    'fading':         ('📉', 'Fading',          'Rank deteriorating from recent levels'),
    'volatile':       ('🌊', 'Volatile',        'Large and unpredictable rank swings'),
    'new_entry':      ('💎', 'New Entry',        'Recently appeared or insufficient history'),
    'stagnant':       ('⏸️', 'Stagnant',        'No significant rank movement'),
    'price_confirmed': ('💰', 'Price Confirmed', 'Price movement validates rank trajectory'),
    'price_divergent': ('⚠️', 'Price Divergent', 'Price movement contradicts rank trajectory'),
    'decay_warning':   ('🔻', 'Decay Warning',   'Good rank but negative recent returns — momentum fading')
}

PATTERN_COLORS = {
    'rocket': '#FF4500', 'breakout': '#FFD700', 'stable_elite': '#8A2BE2',
    'at_peak': '#FF69B4', 'steady_climber': '#32CD32', 'recovery': '#00BFFF',
    'fading': '#808080', 'volatile': '#FF8C00', 'new_entry': '#00CED1',
    'stagnant': '#A9A9A9', 'price_confirmed': '#00E676', 'price_divergent': '#FF1744',
    'decay_warning': '#FF6347'
}

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #FF6B35, #004E98);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0; line-height: 1.2;
    }
    .sub-header {
        font-size: 1.05rem; color: #888; margin-top: -8px; margin-bottom: 20px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px; padding: 20px; text-align: center;
        border: 1px solid #333; box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #FF6B35; }
    .kpi-label { font-size: 0.8rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .stock-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px; padding: 25px; margin-bottom: 20px;
        border: 1px solid #333;
    }
    .grade-S { color: #FFD700; font-weight: 900; font-size: 1.5rem; }
    .grade-A { color: #00C853; font-weight: 900; font-size: 1.5rem; }
    .grade-B { color: #2196F3; font-weight: 900; font-size: 1.5rem; }
    .grade-C { color: #FF9800; font-weight: 900; font-size: 1.5rem; }
    .grade-D { color: #FF5722; font-weight: 900; font-size: 1.5rem; }
    .grade-F { color: #F44336; font-weight: 900; font-size: 1.5rem; }
    .pattern-tag {
        display: inline-block; padding: 4px 12px; border-radius: 15px;
        font-size: 0.85rem; background: rgba(255,107,53,0.15); color: #FF6B35;
        border: 1px solid rgba(255,107,53,0.3); margin: 2px;
    }
    .mover-up { color: #00C853; font-weight: 700; }
    .mover-down { color: #F44336; font-weight: 700; }
    .divider { border-top: 1px solid #333; margin: 15px 0; }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px; font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    .funnel-stage {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px; padding: 18px; margin-bottom: 12px;
        border-left: 5px solid;
    }
    .stage-discovery { border-left-color: #2196F3; }
    .stage-validation { border-left-color: #FF9800; }
    .stage-final { border-left-color: #00C853; }
    .rule-pass { color: #00C853; font-weight: 600; }
    .rule-fail { color: #F44336; font-weight: 600; }
    .final-buy-card {
        background: linear-gradient(135deg, #0d2b0d, #1a3a1a);
        border: 2px solid #00C853; border-radius: 14px;
        padding: 20px; margin-bottom: 12px;
        box-shadow: 0 0 15px rgba(0,200,83,0.15);
    }
    .funnel-stat {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 10px; padding: 15px; text-align: center;
        border: 1px solid #444;
    }
    .funnel-stat-value { font-size: 1.8rem; font-weight: 700; }
    .funnel-stat-label { font-size: 0.75rem; color: #aaa; text-transform: uppercase; }
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

    # ── Price-Rank Alignment Multiplier (v2.3 — return-based) ──
    # Uses CSV ret_7d/ret_30d instead of raw prices — no split detection needed
    ret_7d = h.get('ret_7d', [])
    ret_30d = h.get('ret_30d', [])
    price_multiplier, price_label, price_alignment = _calc_price_alignment(ret_7d, ret_30d, pcts, avg_pct)
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


# ── Return-Based Price-Rank Alignment Engine (v2.3) ──

def _calc_price_alignment(ret_7d: List[float], ret_30d: List[float],
                          pcts: List[float], avg_pct: float) -> Tuple[float, str, float]:
    """
    Return-Based Price-Rank Alignment Multiplier (v2.3).

    Uses CSV-provided ret_7d and ret_30d which are ALREADY split-adjusted by the
    data provider. No manual split detection needed — cleaner and more accurate.

    TWO CLEAN SIGNALS:

    Signal 1 — Return-Rank Directional Agreement (55%):
      For each week, does the sign of ret_7d match the direction of rank change?
      Positive ret_7d + improving rank = agreement.
      Uses wider noise band for stable elite stocks.

    Signal 2 — Return Quality Confirmation (45%):
      For highly ranked stocks, are recent returns positive (confirming value)?
      Uses avg of recent ret_30d values. A stock at rank 5 with -22% monthly
      return is a TRAP — this signal catches it.

    MULTIPLIER RANGE: ×0.92 (strong divergence) to ×1.08 (strong confirmation)

    Returns: (multiplier, label, alignment_score)
    """
    cfg = PRICE_ALIGNMENT
    n = len(pcts)

    # ── Guard: Need valid return data ──
    valid_ret7 = [r for r in ret_7d if r is not None and not np.isnan(r)]
    if len(valid_ret7) < cfg['min_weeks'] or n < cfg['min_weeks']:
        return 1.0, 'NEUTRAL', 50.0

    # Build aligned triplets: (ret_7d, ret_30d, percentile)
    pairs = []
    for i in range(n):
        r7 = ret_7d[i] if i < len(ret_7d) else float('nan')
        r30 = ret_30d[i] if i < len(ret_30d) else float('nan')
        if r7 is not None and not np.isnan(r7):
            pairs.append((r7, r30, pcts[i]))

    if len(pairs) < cfg['min_weeks']:
        return 1.0, 'NEUTRAL', 50.0

    # ── Signal 1: Return-Rank Directional Agreement (55%) ──
    agree = 0
    counted = 0
    noise_band = cfg['noise_band_stable'] if avg_pct > 80 else cfg['noise_band_normal']

    for i in range(1, len(pairs)):
        r7 = pairs[i][0]
        r_chg = pairs[i][2] - pairs[i - 1][2]  # Percentile change

        # Skip noise — tiny rank moves for elite stocks
        if abs(r_chg) < noise_band and abs(r7) < 1.0:
            continue

        counted += 1
        # Positive ret_7d should align with improving percentile
        if r7 > 0 and r_chg > 0:
            agree += 1.0   # Both positive — strong agreement
        elif r7 < -1.0 and r_chg < -1.0:
            agree += 0.8   # Both negative — at least consistent
        elif abs(r7) < 1.0:
            agree += 0.3   # Return near zero — not really disagreeing
        else:
            agree -= 0.3   # Divergent direction

    if counted > 0:
        dir_score = float(np.clip((agree / counted) * 50 + 50, 0, 100))
    else:
        # No significant rank moves — check cumulative using latest ret_30d
        latest_r30 = float('nan')
        for _, r30, _ in reversed(pairs):
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

    # ── Signal 2: Return Quality Confirmation (45%) ──
    # Recent ret_30d tells us if the stock is actually making money
    recent_window = min(6, len(pairs))
    recent_r30 = [p[1] for p in pairs[-recent_window:]
                  if p[1] is not None and not np.isnan(p[1])]
    recent_r7 = [p[0] for p in pairs[-recent_window:]]

    if recent_r30:
        avg_r30 = float(np.mean(recent_r30))
        if avg_pct >= 70:
            # High-ranked stocks SHOULD have positive returns
            if avg_r30 > 10:
                trend_score = 85.0
            elif avg_r30 > 0:
                trend_score = 65.0
            elif avg_r30 > -10:
                trend_score = 45.0
            else:
                trend_score = 20.0   # TRAP: rank high but returns very negative
        else:
            # Lower-ranked stocks — any positive returns are a bonus
            if avg_r30 > 20:
                trend_score = 80.0
            elif avg_r30 > 5:
                trend_score = 60.0
            elif avg_r30 > -5:
                trend_score = 50.0
            else:
                trend_score = 35.0
    else:
        trend_score = 50.0

    # ── Composite Alignment Score ──
    alignment = 0.55 * dir_score + 0.45 * trend_score
    alignment = float(np.clip(alignment, 0, 100))

    # ── Convert to Multiplier ──
    conf_thresh = cfg['confirmed_threshold']
    div_thresh = cfg['divergent_threshold']
    max_boost = cfg['multiplier_max_boost']
    max_pen = cfg['multiplier_max_penalty']

    if alignment >= conf_thresh:
        t = (alignment - conf_thresh) / (100 - conf_thresh)
        multiplier = 1.03 + t * (max_boost - 1.03)
        label = 'PRICE_CONFIRMED'
    elif alignment >= 50:
        t = (alignment - 50) / (conf_thresh - 50)
        multiplier = 1.00 + t * 0.03
        label = 'NEUTRAL'
    elif alignment >= div_thresh:
        t = (alignment - div_thresh) / (50 - div_thresh)
        multiplier = 0.97 + t * 0.03
        label = 'NEUTRAL'
    else:
        t = alignment / div_thresh
        multiplier = max_pen + t * (0.97 - max_pen)
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
    Position-Aware Consistency.
    
    THE FIX: A stock oscillating between rank 1 and rank 8 (pct 99.6-99.95)
    has changes of [-0.35, +0.2, -0.15, +0.3...]. Old system sees these as
    "bidirectional = inconsistent". But oscillating within the TOP 1% is
    INCREDIBLE consistency!
    
    Solution: Measure consistency RELATIVE to position band.
    - Elite stocks: Consistency = are they staying in their percentile band?
    - Other stocks: Consistency = are they moving in a consistent DIRECTION?
    """
    if n < 3:
        return 50.0

    changes = np.diff(pcts)
    std = np.std(changes)
    positive_ratio = np.sum(changes > 0) / len(changes)
    avg_pct = np.mean(pcts)
    current_pct = pcts[-1]
    
    # === POSITION-RELATIVE CONSISTENCY ===
    if avg_pct >= 85:
        # ELITE CONSISTENCY: Staying within a tight band at the top
        # Even if direction flips, small oscillations at 95-100% = very consistent
        pct_range = max(pcts) - min(pcts)  # How wide is the band?
        
        # Band score: range of 0 = 100, range of 30 = 50, range of 60+ = 0
        band_score = float(np.clip(100 - pct_range * 1.67, 0, 100))
        
        # Time-at-top: what % of weeks were they above 80th percentile?
        time_at_top = sum(1 for p in pcts if p >= 80) / len(pcts) * 100
        
        # Low absolute volatility bonus (std < 3 at elite = very stable)
        vol_bonus = float(np.clip(100 - std * 5, 0, 100))
        
        # Elite: 40% band tightness, 35% time at top, 25% low volatility
        return 0.40 * band_score + 0.35 * time_at_top + 0.25 * vol_bonus
    
    elif avg_pct >= 60:
        # STRONG STOCKS: Mix of band + direction
        stability = float(np.clip(100 - std * 2, 0, 100))
        direction = positive_ratio * 100
        # Bonus: if currently higher than start, consistency gets a lift
        trajectory_lift = min((pcts[-1] - pcts[0]) / 20.0 * 10, 15)  # max +15
        base = 0.50 * stability + 0.50 * direction
        return float(np.clip(base + trajectory_lift, 0, 100))
    
    else:
        # LOWER STOCKS: Pure direction + stability (original formula)
        stability = float(np.clip(100 - std * 2, 0, 100))
        direction = positive_ratio * 100
        return 0.55 * stability + 0.45 * direction


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


# ── Pattern Detection ──

def _detect_pattern(ranks, totals, pcts, positional, trend, velocity, acceleration, consistency) -> str:
    """Classify trajectory into a pattern based on metrics and shape (v2.0 position-aware)"""
    n = len(ranks)
    if n < MIN_WEEKS_DEFAULT:
        return 'new_entry'

    current_pct = pcts[-1]
    avg_pct = np.mean(pcts)
    current_rank = ranks[-1]
    best_rank = min(ranks)

    # 🎯 Stable Elite - Consistently top-ranked (PRIORITIZED in v2.0)
    # This MUST come before Rocket to prevent elite stocks from being misclassified
    if positional > 88 and consistency > 60 and current_pct > 85:
        return 'stable_elite'

    # 🚀 Rocket - Strong improvement everywhere
    if trend > 78 and velocity > 72 and acceleration > 55:
        return 'rocket'

    # ⚡ Breakout - Sudden jump beyond normal variance
    if n >= 4:
        recent_change = pcts[-1] - pcts[-3]
        avg_abs_change = np.mean(np.abs(np.diff(pcts)))
        if avg_abs_change > 0 and recent_change > 0 and recent_change > 2.8 * avg_abs_change:
            return 'breakout'

    # 🏔️ At Peak - Near best rank ever (position-aware)
    if best_rank > 0 and current_rank <= best_rank * 1.12 and current_pct > 68:
        return 'at_peak'

    # 📈 Steady Climber - Gradual consistent improvement
    if trend > 58 and consistency > 58 and velocity > 48:
        return 'steady_climber'

    # 🔄 Recovery - Bouncing back
    if velocity > 62 and current_pct > avg_pct and trend < 55:
        return 'recovery'

    # 📉 Fading - Deteriorating
    if velocity < 35 and trend < 40:
        return 'fading'

    # 🌊 Volatile - Wild swings
    if consistency < 32:
        return 'volatile'

    return 'stagnant'


# ============================================
# TOP MOVERS CALCULATION
# ============================================

def get_top_movers(histories: dict, n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get biggest rank gainers and decliners in the latest week"""
    movers = []
    for ticker, h in histories.items():
        if len(h['ranks']) < 2:
            continue
        change = int(h['ranks'][-2] - h['ranks'][-1])  # Positive = improved
        movers.append({
            'ticker': ticker,
            'company_name': h['company_name'],
            'category': h['category'],
            'prev_rank': int(h['ranks'][-2]),
            'current_rank': int(h['ranks'][-1]),
            'rank_change': change
        })

    mover_df = pd.DataFrame(movers)
    if mover_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    gainers = mover_df.nlargest(n, 'rank_change')
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

        # Min weeks
        min_weeks = st.slider("Min Weeks of Data", 2, metadata['total_weeks'], MIN_WEEKS_DEFAULT, key='sb_weeks')

        # Min T-Score
        min_score = st.slider("Min Trajectory Score", 0, 100, 0, key='sb_score')

        # Display count
        display_n = st.select_slider("Show Top N", options=[25, 50, 100, 200, 500, 1000, 5000],
                                      value=MAX_DISPLAY_DEFAULT, key='sb_topn')

        st.markdown("---")
        st.markdown("#### 📋 Quick Filters")
        quick_filter = st.radio("Preset", ['None', '🚀 Rockets Only', '🎯 Elite Only',
                                           '📈 Climbers', '⚡ Breakouts', '🏔️ At Peak',
                                           'TMI > 70', 'Positional > 80'],
                                index=0, key='sb_quick')

        st.markdown("---")
        st.caption("v2.3.0 | Return-Based + Decay + Sector Alpha")

    return {
        'categories': selected_cats,
        'sectors': selected_sectors,
        'min_weeks': min_weeks,
        'min_score': min_score,
        'display_n': display_n,
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
    elif qf == 'TMI > 70':
        df = df[df['tmi'] > 70]
    elif qf == 'Positional > 80':
        df = df[df['positional'] > 80]

    # Limit
    df = df.head(filters['display_n'])

    # Re-rank after filtering
    df = df.reset_index(drop=True)
    df['t_rank'] = range(1, len(df) + 1)

    return df


# ============================================
# UI: RANKINGS TAB — ALL TIME BEST (v2.3)
# ============================================

def render_rankings_tab(filtered_df: pd.DataFrame, all_df: pd.DataFrame,
                        histories: dict, metadata: dict):
    """Render the main rankings tab — ALL TIME BEST smart engineering"""

    # ── Ensure all v2.3 columns exist (defensive) ──
    for col, default in [('price_tag', ''), ('signal_tags', ''), ('decay_tag', ''),
                         ('decay_label', ''), ('decay_score', 0), ('decay_multiplier', 1.0),
                         ('sector_alpha_tag', 'NEUTRAL'), ('sector_alpha_value', 0),
                         ('price_label', 'NEUTRAL'), ('price_alignment', 50),
                         ('price_multiplier', 1.0), ('pre_price_score', 0),
                         ('pre_decay_score', 0)]:
        if col not in all_df.columns:
            all_df[col] = default
        if col not in filtered_df.columns:
            filtered_df[col] = default

    # ── Row 1: Smart KPI Cards (8 metrics) ──
    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)

    rockets = len(all_df[all_df['pattern_key'] == 'rocket'])
    elites = len(all_df[all_df['pattern_key'] == 'stable_elite'])
    avg_score = all_df['trajectory_score'].mean()
    confirmed = len(all_df[all_df['price_label'] == 'PRICE_CONFIRMED'])
    divergent = len(all_df[all_df['price_label'] == 'PRICE_DIVERGENT'])
    decay_high = len(all_df[all_df['decay_label'] == 'DECAY_HIGH'])
    decay_any = len(all_df[all_df['decay_label'].isin(['DECAY_HIGH', 'DECAY_MODERATE', 'DECAY_MILD'])])
    sect_leaders = len(all_df[all_df['sector_alpha_tag'] == 'SECTOR_LEADER'])

    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{len(filtered_df):,}</div>
            <div class="kpi-label">Shown</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{avg_score:.1f}</div>
            <div class="kpi-label">Avg Score</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{rockets}</div>
            <div class="kpi-label">🚀 Rockets</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{elites}</div>
            <div class="kpi-label">🎯 Elites</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{confirmed}</div>
            <div class="kpi-label">💰 Confirmed</div></div>""", unsafe_allow_html=True)
    with k6:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#FF1744;">{decay_high}</div>
            <div class="kpi-label">🔻 Traps</div></div>""", unsafe_allow_html=True)
    with k7:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#FFD700;">{sect_leaders}</div>
            <div class="kpi-label">👑 Sect Alpha</div></div>""", unsafe_allow_html=True)
    with k8:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{metadata['total_weeks']}</div>
            <div class="kpi-label">📅 Weeks</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Row 2: Signal Alert Cards (only if alerts exist) ──
    # Show DECAY_HIGH stocks as red danger cards + PRICE_DIVERGENT as orange
    traps = all_df[(all_df['decay_label'] == 'DECAY_HIGH') & (all_df['trajectory_score'] >= 50)].head(8)
    divergent_stocks = all_df[(all_df['price_label'] == 'PRICE_DIVERGENT') & (all_df['trajectory_score'] >= 55)].head(5)

    if not traps.empty or not divergent_stocks.empty:
        with st.expander("🚨 SIGNAL ALERTS — Stocks Requiring Attention", expanded=True):
            if not traps.empty:
                st.markdown("##### 🔻 Momentum Decay Traps — Good Rank, Bad Returns")
                st.caption("These stocks rank well but have significantly negative recent returns. The rank hasn't caught up yet — proceed with caution.")
                trap_cols = st.columns(min(len(traps), 4))
                for idx, (_, t_row) in enumerate(traps.iterrows()):
                    with trap_cols[idx % len(trap_cols)]:
                        # Get latest ret_7d and ret_30d from history
                        t_h = histories.get(t_row['ticker'], {})
                        _r7 = [v for v in t_h.get('ret_7d', []) if v is not None and not np.isnan(v)]
                        _r30 = [v for v in t_h.get('ret_30d', []) if v is not None and not np.isnan(v)]
                        lr7 = f"{_r7[-1]:+.1f}%" if _r7 else 'N/A'
                        lr30 = f"{_r30[-1]:+.1f}%" if _r30 else 'N/A'
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#2a1010,#1a0505); border-radius:10px;
                            padding:12px; border:1px solid #FF1744; margin-bottom:8px;">
                            <div style="font-weight:700; color:#FF1744; font-size:1rem;">{t_row['ticker']}</div>
                            <div style="color:#aaa; font-size:0.75rem;">{str(t_row.get('company_name',''))[:25]}</div>
                            <div style="display:flex; justify-content:space-between; margin-top:6px;">
                                <span style="color:#888;">Rank #{int(t_row['current_rank'])}</span>
                                <span style="color:#FF6B35; font-weight:600;">T-Score {t_row['trajectory_score']:.0f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-top:4px;">
                                <span style="color:#FF1744;">7d: {lr7}</span>
                                <span style="color:#FF1744;">30d: {lr30}</span>
                            </div>
                            <div style="color:#FF6347; font-size:0.7rem; margin-top:4px;">
                                Decay: {int(t_row.get('decay_score',0))}/100 • ×{t_row.get('decay_multiplier',1):.3f}
                            </div>
                        </div>""", unsafe_allow_html=True)

            if not divergent_stocks.empty:
                st.markdown("##### ⚠️ Price Divergent — Returns Contradict Rank")
                div_cards = st.columns(min(len(divergent_stocks), 5))
                for idx, (_, d_row) in enumerate(divergent_stocks.iterrows()):
                    with div_cards[idx % len(div_cards)]:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#2a1f10,#1a1505); border-radius:10px;
                            padding:10px; border:1px solid #FF9800; margin-bottom:8px;">
                            <div style="font-weight:600; color:#FF9800;">{d_row['ticker']}</div>
                            <div style="color:#aaa; font-size:0.7rem;">{str(d_row.get('company_name',''))[:20]}</div>
                            <div style="color:#888; font-size:0.8rem; margin-top:4px;">
                                Rank #{int(d_row['current_rank'])} • Score {d_row['trajectory_score']:.0f}
                            </div>
                            <div style="color:#FF9800; font-size:0.7rem;">
                                Align: {d_row.get('price_alignment',50):.0f} • ×{d_row.get('price_multiplier',1):.3f}
                            </div>
                        </div>""", unsafe_allow_html=True)

    # ── Row 3: Top Movers This Week (upgraded) ──
    with st.expander("🔥 Top Movers This Week", expanded=False):
        gainers, decliners = get_top_movers(histories, n=10)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### ⬆️ Biggest Climbers")
            if not gainers.empty:
                # Enrich with trajectory scores and signals
                g_enriched = gainers.copy()
                g_enriched = g_enriched.merge(
                    all_df[['ticker', 'trajectory_score', 'grade', 'signal_tags', 'pattern']],
                    on='ticker', how='left'
                )
                display_g = g_enriched[['ticker', 'company_name', 'prev_rank', 'current_rank',
                                         'rank_change', 'trajectory_score', 'grade', 'signal_tags']].copy()
                display_g.columns = ['Ticker', 'Company', 'Prev', 'Now', 'Δ', 'T-Score', 'Grade', 'Signals']
                display_g['Company'] = display_g['Company'].str[:25]
                st.dataframe(display_g, column_config={
                    'Δ': st.column_config.NumberColumn(format="%+d"),
                    'T-Score': st.column_config.ProgressColumn('T-Score', min_value=0, max_value=100, format="%.0f"),
                }, hide_index=True, use_container_width=True)
            else:
                st.info("No data")
        with c2:
            st.markdown("##### ⬇️ Biggest Decliners")
            if not decliners.empty:
                d_enriched = decliners.copy()
                d_enriched = d_enriched.merge(
                    all_df[['ticker', 'trajectory_score', 'grade', 'signal_tags', 'pattern']],
                    on='ticker', how='left'
                )
                display_d = d_enriched[['ticker', 'company_name', 'prev_rank', 'current_rank',
                                         'rank_change', 'trajectory_score', 'grade', 'signal_tags']].copy()
                display_d.columns = ['Ticker', 'Company', 'Prev', 'Now', 'Δ', 'T-Score', 'Grade', 'Signals']
                display_d['Company'] = display_d['Company'].str[:25]
                st.dataframe(display_d, column_config={
                    'Δ': st.column_config.NumberColumn(format="%+d"),
                    'T-Score': st.column_config.ProgressColumn('T-Score', min_value=0, max_value=100, format="%.0f"),
                }, hide_index=True, use_container_width=True)
            else:
                st.info("No data")

    # ── Sort Options ──
    sort_col1, sort_col2 = st.columns([1, 3])
    with sort_col1:
        sort_by = st.selectbox("Sort By", [
            'Trajectory Score', 'Positional Quality', 'TMI', 'Current Rank', 'Rank Change',
            'Best Rank', 'Streak', 'Trend', 'Velocity', 'Consistency',
            'Price Alignment', 'Decay Score', 'Sector Alpha'
        ], key='rank_sort')

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
    col_name, ascending = sort_map[sort_by]
    display_df = filtered_df.sort_values(col_name, ascending=ascending).reset_index(drop=True)
    display_df['t_rank'] = range(1, len(display_df) + 1)

    # ── Rankings Table ──
    st.markdown("##### 📋 Trajectory Rankings")

    # Prepare display columns
    table_df = display_df[[
        't_rank', 'ticker', 'company_name', 'sector',
        'trajectory_score', 'grade', 'pattern', 'signal_tags', 'tmi',
        'current_rank', 'best_rank', 'rank_change', 'last_week_change',
        'streak', 'weeks', 'sparkline'
    ]].copy()

    table_df.columns = [
        '#', 'Ticker', 'Company', 'Sector',
        'T-Score', 'Grade', 'Pattern', 'Signals', 'TMI',
        'Rank Now', 'Best', 'Δ Total', 'Δ Week',
        'Streak', 'Wks', 'Trajectory'
    ]

    # Truncate display
    table_df['Company'] = table_df['Company'].str[:25]
    table_df['Sector'] = table_df['Sector'].str[:18]

    st.dataframe(
        table_df,
        column_config={
            '#': st.column_config.NumberColumn(width="small"),
            'T-Score': st.column_config.ProgressColumn(
                'T-Score', min_value=0, max_value=100, format="%.1f"
            ),
            'TMI': st.column_config.ProgressColumn(
                'TMI', min_value=0, max_value=100, format="%.0f"
            ),
            'Trajectory': st.column_config.LineChartColumn(
                'Trajectory', y_min=0, y_max=100, width="medium"
            ),
            'Δ Total': st.column_config.NumberColumn(format="%+d"),
            'Δ Week': st.column_config.NumberColumn(format="%+d"),
            'Streak': st.column_config.NumberColumn(format="%d 🔥"),
        },
        hide_index=True,
        use_container_width=True,
        height=650
    )

    # ── Insights Section (v2.3 — ALL TIME BEST) ──
    with st.expander("📊 Trajectory Intelligence Dashboard", expanded=False):

        # ─── Row 1: Score Pipeline Waterfall + Decay Breakdown ───
        wf_c1, wf_c2 = st.columns(2)

        with wf_c1:
            # Score Pipeline Waterfall — how many stocks survive each scoring layer
            total = len(all_df)
            above70 = len(all_df[all_df['pre_price_score'] >= 70])
            price_boosted = len(all_df[all_df['price_multiplier'] > 1.01])
            price_penalized = len(all_df[all_df['price_multiplier'] < 0.99])
            decay_penalized = len(all_df[all_df['decay_multiplier'] < 0.99])
            final_s_grade = len(all_df[all_df['grade'] == 'S'])
            final_a_grade = len(all_df[all_df['grade'] == 'A'])

            fig_wf = go.Figure(go.Waterfall(
                name="Pipeline", orientation="v",
                measure=["absolute", "absolute", "relative", "relative", "relative", "absolute"],
                x=["Universe", "T≥70 (Pre)", "💰 Boosted", "⚠️ Price Pen", "🔻 Decay Pen", "Grade S+A"],
                y=[total, above70, price_boosted, -price_penalized, -decay_penalized, final_s_grade + final_a_grade],
                connector={"line": {"color": "#444"}},
                increasing={"marker": {"color": "#00C853"}},
                decreasing={"marker": {"color": "#FF1744"}},
                totals={"marker": {"color": "#FF6B35"}},
                text=[total, above70, f"+{price_boosted}", f"-{price_penalized}", f"-{decay_penalized}", final_s_grade + final_a_grade],
                textposition="outside"
            ))
            fig_wf.update_layout(
                title="Score Pipeline — How Multipliers Shape Rankings",
                height=380, template='plotly_dark',
                margin=dict(t=40, b=30, l=30, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        with wf_c2:
            # Decay Health — donut showing universe decay status
            clean = len(all_df[~all_df['decay_label'].isin(['DECAY_HIGH', 'DECAY_MODERATE', 'DECAY_MILD'])])
            mild = len(all_df[all_df['decay_label'] == 'DECAY_MILD'])
            moderate = len(all_df[all_df['decay_label'] == 'DECAY_MODERATE'])
            severe = len(all_df[all_df['decay_label'] == 'DECAY_HIGH'])

            fig_decay = go.Figure(data=[go.Pie(
                labels=['✅ Clean', '~ Mild', '⚡ Moderate', '🔻 Severe'],
                values=[clean, mild, moderate, severe],
                marker_colors=['#00C853', '#FFD700', '#FF9800', '#FF1744'],
                hole=0.55,
                textinfo='label+value',
                textfont_size=12,
                sort=False
            )])
            fig_decay.update_layout(
                title="Momentum Decay Health — Universe Overview",
                height=380, template='plotly_dark',
                showlegend=False,
                margin=dict(t=40, b=20, l=20, r=20),
                annotations=[dict(text=f'{clean}/{total}<br>Clean',
                                   x=0.5, y=0.5, font_size=14, showarrow=False,
                                   font_color='#00C853')]
            )
            st.plotly_chart(fig_decay, use_container_width=True)

        # ─── Row 2: Sector Heatmap + Pattern Distribution ───
        sec_c1, sec_c2 = st.columns([3, 2])

        with sec_c1:
            # Sector Scoreboard Heatmap
            qualified = all_df[all_df['weeks'] >= 3].copy()
            if not qualified.empty:
                sect_agg = qualified.groupby('sector').agg(
                    avg_score=('trajectory_score', 'mean'),
                    count=('trajectory_score', 'count'),
                    leaders=('sector_alpha_tag', lambda x: (x == 'SECTOR_LEADER').sum()),
                    betas=('sector_alpha_tag', lambda x: (x == 'SECTOR_BETA').sum()),
                    rockets_n=('pattern_key', lambda x: (x == 'rocket').sum()),
                    decay_n=('decay_label', lambda x: x.isin(['DECAY_HIGH', 'DECAY_MODERATE']).sum()),
                    avg_decay=('decay_score', 'mean'),
                ).reset_index()
                sect_agg = sect_agg[sect_agg['count'] >= 3].sort_values('avg_score', ascending=False).head(18)

                if not sect_agg.empty:
                    # Color bars by avg_score — green=high, red=low
                    max_s = sect_agg['avg_score'].max()
                    min_s = sect_agg['avg_score'].min()
                    colors = []
                    for s in sect_agg['avg_score']:
                        ratio = (s - min_s) / max(max_s - min_s, 1)
                        if ratio > 0.7:
                            colors.append('#00C853')
                        elif ratio > 0.4:
                            colors.append('#FF9800')
                        else:
                            colors.append('#FF5722')

                    labels = [f"{row['sector'][:20]}  ({int(row['count'])} stk, 👑{int(row['leaders'])}, 🔻{int(row['decay_n'])})"
                              for _, row in sect_agg.iterrows()]

                    fig_sh = go.Figure(data=[go.Bar(
                        x=sect_agg['avg_score'].values,
                        y=labels,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:.1f}" for v in sect_agg['avg_score']],
                        textposition='auto'
                    )])
                    fig_sh.update_layout(
                        title="Sector Scoreboard — Avg T-Score (stocks, 👑leaders, 🔻decay)",
                        height=480, template='plotly_dark',
                        xaxis_title='Avg T-Score',
                        yaxis=dict(autorange='reversed'),
                        margin=dict(t=40, b=40, l=280, r=20)
                    )
                    st.plotly_chart(fig_sh, use_container_width=True)

        with sec_c2:
            # Pattern distribution donut
            pattern_counts = all_df['pattern_key'].value_counts()
            fig_pat = go.Figure(data=[go.Pie(
                labels=[PATTERN_DEFS.get(k, ('', k, ''))[1] for k in pattern_counts.index],
                values=pattern_counts.values,
                marker_colors=[PATTERN_COLORS.get(k, '#888') for k in pattern_counts.index],
                hole=0.45,
                textinfo='label+percent',
                textfont_size=10
            )])
            fig_pat.update_layout(
                title="Pattern Distribution",
                height=480, template='plotly_dark',
                showlegend=False,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pat, use_container_width=True)

        # ─── Row 3: Grade Distribution + Sector Alpha Breakdown ───
        gr_c1, gr_c2 = st.columns(2)

        with gr_c1:
            # Grade distribution with colored bars
            grade_counts = all_df['grade'].value_counts().reindex(['S', 'A', 'B', 'C', 'D', 'F']).fillna(0)
            fig_grade = go.Figure(data=[go.Bar(
                x=grade_counts.index,
                y=grade_counts.values,
                marker_color=[GRADE_COLORS.get(g, '#888') for g in grade_counts.index],
                text=[f"{int(v)}" for v in grade_counts.values],
                textposition='outside',
                width=0.7
            )])
            fig_grade.update_layout(
                title="Grade Distribution",
                height=320, template='plotly_dark',
                xaxis_title='Grade', yaxis_title='Count',
                margin=dict(t=40, b=40, l=40, r=20)
            )
            st.plotly_chart(fig_grade, use_container_width=True)

        with gr_c2:
            # Sector Alpha breakdown
            alpha_counts = all_df['sector_alpha_tag'].value_counts()
            alpha_order = ['SECTOR_LEADER', 'SECTOR_OUTPERFORM', 'SECTOR_ALIGNED', 'SECTOR_BETA', 'SECTOR_LAGGARD', 'NEUTRAL']
            alpha_colors_map = {'SECTOR_LEADER': '#FFD700', 'SECTOR_OUTPERFORM': '#00C853',
                                'SECTOR_ALIGNED': '#888', 'SECTOR_BETA': '#FF9800',
                                'SECTOR_LAGGARD': '#FF1744', 'NEUTRAL': '#555'}
            alpha_labels = []
            alpha_vals = []
            alpha_cols = []
            for a in alpha_order:
                if a in alpha_counts.index:
                    alpha_labels.append(a.replace('SECTOR_', ''))
                    alpha_vals.append(alpha_counts[a])
                    alpha_cols.append(alpha_colors_map.get(a, '#888'))

            if alpha_labels:
                fig_alpha = go.Figure(data=[go.Bar(
                    x=alpha_labels, y=alpha_vals,
                    marker_color=alpha_cols,
                    text=[int(v) for v in alpha_vals],
                    textposition='outside', width=0.6
                )])
                fig_alpha.update_layout(
                    title="Sector Alpha Classification",
                    height=320, template='plotly_dark',
                    xaxis_title='Classification', yaxis_title='Stocks',
                    margin=dict(t=40, b=40, l=40, r=20)
                )
                st.plotly_chart(fig_alpha, use_container_width=True)

        # ─── Row 4:  Top-10 Elite Conviction list ───
        st.markdown("---")
        st.markdown("##### 🏆 Top-10 Highest Conviction Stocks")
        st.caption("Stocks with Grade S/A + 💰 Price Confirmed + ✅ Clean Decay + 👑 Sector Leader/Outperform")
        conviction_mask = (
            (all_df['grade'].isin(['S', 'A'])) &
            (all_df['price_label'] == 'PRICE_CONFIRMED') &
            (~all_df['decay_label'].isin(['DECAY_HIGH', 'DECAY_MODERATE'])) &
            (all_df['sector_alpha_tag'].isin(['SECTOR_LEADER', 'SECTOR_OUTPERFORM', 'NEUTRAL']))
        )
        conviction = all_df[conviction_mask].sort_values('trajectory_score', ascending=False).head(10)

        if not conviction.empty:
            conv_cols = st.columns(min(len(conviction), 5))
            for idx, (_, c_row) in enumerate(conviction.iterrows()):
                with conv_cols[idx % len(conv_cols)]:
                    g_color = GRADE_COLORS.get(c_row['grade'], '#888')
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#0d1a0d,#0a1a0a); border-radius:12px;
                        padding:14px; border:1px solid #00C853; margin-bottom:8px;
                        box-shadow:0 0 10px rgba(0,200,83,0.1);">
                        <div style="font-weight:800; color:#00C853; font-size:1.05rem;">{c_row['ticker']}</div>
                        <div style="color:#aaa; font-size:0.72rem;">{str(c_row.get('company_name',''))[:22]}</div>
                        <div style="color:{g_color}; font-weight:700; font-size:1.4rem; margin:4px 0;">
                            {c_row['grade_emoji']} {c_row['trajectory_score']:.0f}
                        </div>
                        <div style="color:#888; font-size:0.72rem;">
                            Rank #{int(c_row['current_rank'])} • {c_row['pattern']}
                        </div>
                        <div style="color:#555; font-size:0.65rem; margin-top:4px;">
                            💰 ×{c_row['price_multiplier']:.3f} • {c_row['signal_tags']}
                        </div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("No stocks currently pass all conviction filters. This is normal — high conviction requires ALL signals aligned.")


# ============================================
# UI: SEARCH TAB
# ============================================

def render_search_tab(traj_df: pd.DataFrame, histories: dict, dates_iso: list):
    """Render stock search and detailed analysis tab"""

    # ── Search Input ──
    ticker_options = sorted(traj_df['ticker'].unique().tolist())
    # Build display labels: "TICKER - Company Name"
    label_map = {}
    for _, row in traj_df.iterrows():
        label = f"{row['ticker']} — {row['company_name'][:35]}"
        label_map[label] = row['ticker']

    labels = sorted(label_map.keys())

    selected_label = st.selectbox("🔍 Search Stock (type ticker or company name)",
                                   labels, index=None,
                                   placeholder="Start typing a ticker or company name...",
                                   key='search_select')

    if selected_label is None:
        st.info("👆 Select a stock to view detailed trajectory analysis")
        return

    ticker = label_map[selected_label]
    row = traj_df[traj_df['ticker'] == ticker].iloc[0]
    h = histories.get(ticker, {})

    if not h:
        st.warning("No history data available for this ticker")
        return

    # ── Stock Header Card ──
    st.markdown(f"""
    <div class="stock-card">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <h2 style="margin:0; color:#fff;">{row['ticker']}</h2>
                <p style="margin:2px 0; color:#aaa; font-size:1.1rem;">{row['company_name']}</p>
                <p style="margin:0; color:#666;">{row['category']} • {row['sector']} • {row['industry']}</p>
            </div>
            <div style="text-align:center;">
                <div class="grade-{row['grade']}">{row['grade_emoji']} Grade {row['grade']}</div>
                <div style="font-size:2rem; font-weight:800; color:#FF6B35;">{row['trajectory_score']}</div>
                <div style="color:#888; font-size:0.8rem;">TRAJECTORY SCORE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Current Rank", f"#{row['current_rank']}", f"{row['last_week_change']:+d} this week")
    k2.metric("Best Rank", f"#{row['best_rank']}")
    k3.metric("Total Δ Rank", f"{row['rank_change']:+d}", "improved" if row['rank_change'] > 0 else "declined")
    k4.metric("TMI", f"{row['tmi']:.0f}")
    k5.metric("Streak", f"{row['streak']} weeks")
    price_label_display = row.get('price_label', 'NEUTRAL')
    price_icon = '💰' if price_label_display == 'PRICE_CONFIRMED' else '⚠️' if price_label_display == 'PRICE_DIVERGENT' else '➖'
    k6.metric("Price Align", f"{price_icon} {row.get('price_alignment', 50):.0f}")
    decay_lbl = row.get('decay_label', '')
    decay_icon = '🔻' if decay_lbl == 'DECAY_HIGH' else '⚡' if decay_lbl == 'DECAY_MODERATE' else '~' if decay_lbl == 'DECAY_MILD' else '✅'
    k7.metric("Decay Check", f"{decay_icon} {row.get('decay_score', 0)}")

    st.markdown("---")

    # ── Charts ──
    chart_c1, chart_c2 = st.columns([2, 1])

    with chart_c1:
        st.markdown("##### 📈 Rank Trajectory Over Time")
        _render_trajectory_chart(h, ticker)

    with chart_c2:
        st.markdown("##### 🎯 Component Breakdown")
        _render_radar_chart(row)

    st.markdown("---")

    # ── Detailed Stats ──
    stat_c1, stat_c2 = st.columns(2)

    with stat_c1:
        st.markdown("##### 📊 Trajectory Statistics")
        stats_data = {
            'Metric': [
                'Trajectory Score', 'Grade', 'Pattern', 'TMI',
                'Current Rank', 'Best Rank', 'Worst Rank', 'Avg Rank',
                'Total Rank Change', 'Last Week Change',
                'Improvement Streak', 'Weeks of Data', 'Rank Volatility'
            ],
            'Value': [
                f"{row['trajectory_score']:.1f} / 100",
                f"{row['grade_emoji']} {row['grade']}",
                row['pattern'],
                f"{row['tmi']:.0f} / 100",
                f"#{row['current_rank']}",
                f"#{row['best_rank']}",
                f"#{row['worst_rank']}",
                f"#{row['avg_rank']}",
                f"{row['rank_change']:+d} positions",
                f"{row['last_week_change']:+d} positions",
                f"{row['streak']} consecutive weeks",
                f"{row['weeks']} weeks",
                f"{row['rank_volatility']:.1f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)

    with stat_c2:
        st.markdown("##### 🧩 Component Scores")
        # Get adaptive weights for this stock's percentile tier
        avg_pct_val = float(np.mean(histories.get(ticker, {}).get('ranks', [500])))
        total_wk = histories.get(ticker, {}).get('total_per_week', [2000])
        avg_total = float(np.mean(total_wk)) if total_wk else 2000
        stock_avg_pct = (1 - avg_pct_val / max(avg_total, 1)) * 100
        adp_w = _get_adaptive_weights(stock_avg_pct)
        comp_data = {
            'Component': ['Positional', 'Trend', 'Velocity', 'Acceleration', 'Consistency', 'Resilience'],
            'Weight': [f"{adp_w['positional']*100:.0f}%", f"{adp_w['trend']*100:.0f}%",
                       f"{adp_w['velocity']*100:.0f}%", f"{adp_w['acceleration']*100:.0f}%",
                       f"{adp_w['consistency']*100:.0f}%", f"{adp_w['resilience']*100:.0f}%"],
            'Score': [row['positional'], row['trend'], row['velocity'], row['acceleration'],
                      row['consistency'], row['resilience']],
            'Contribution': [
                round(row['positional'] * adp_w['positional'], 1),
                round(row['trend'] * adp_w['trend'], 1),
                round(row['velocity'] * adp_w['velocity'], 1),
                round(row['acceleration'] * adp_w['acceleration'], 1),
                round(row['consistency'] * adp_w['consistency'], 1),
                round(row['resilience'] * adp_w['resilience'], 1)
            ]
        }
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, column_config={
            'Score': st.column_config.ProgressColumn('Score', min_value=0, max_value=100, format="%.1f")
        }, hide_index=True, use_container_width=True)

        # Price-Rank Alignment detail (v2.3 — return-based)
        pa_score = row.get('price_alignment', 50)
        pa_mult = row.get('price_multiplier', 1.0)
        pa_label = row.get('price_label', 'NEUTRAL')
        pre_price = row.get('pre_price_score', row['trajectory_score'])
        pa_color = '#00E676' if pa_label == 'PRICE_CONFIRMED' else '#FF1744' if pa_label == 'PRICE_DIVERGENT' else '#888'
        st.markdown(f"""
        ##### 💰 Return-Based Price Alignment
        <div style="background:#1e1e2e; border-radius:10px; padding:12px; border:1px solid {pa_color};">
            <div style="display:flex; justify-content:space-between;">
                <div><span style="color:#aaa;">Alignment Score:</span> <b style="color:{pa_color};">{pa_score:.0f}/100</b></div>
                <div><span style="color:#aaa;">Multiplier:</span> <b style="color:{pa_color};">×{pa_mult:.3f}</b></div>
            </div>
            <div style="margin-top:6px;"><span style="color:#aaa;">Status:</span> <b style="color:{pa_color};">{pa_label}</b></div>
            <div style="margin-top:4px; color:#666; font-size:0.8rem;">Pre-price: {pre_price:.1f} → Post-price: {row.get('pre_decay_score', row['trajectory_score']):.1f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Momentum Decay Warning detail (v2.3)
        d_score = row.get('decay_score', 0)
        d_mult = row.get('decay_multiplier', 1.0)
        d_label = row.get('decay_label', '')
        pre_decay = row.get('pre_decay_score', row['trajectory_score'])
        d_color = '#FF1744' if d_label == 'DECAY_HIGH' else '#FF9800' if d_label == 'DECAY_MODERATE' else '#FFD700' if d_label == 'DECAY_MILD' else '#00E676'
        d_status = d_label if d_label else 'CLEAN ✅'
        st.markdown(f"""
        ##### 🔻 Momentum Decay Check
        <div style="background:#1e1e2e; border-radius:10px; padding:12px; border:1px solid {d_color};">
            <div style="display:flex; justify-content:space-between;">
                <div><span style="color:#aaa;">Decay Score:</span> <b style="color:{d_color};">{d_score}/100</b></div>
                <div><span style="color:#aaa;">Multiplier:</span> <b style="color:{d_color};">×{d_mult:.3f}</b></div>
            </div>
            <div style="margin-top:6px;"><span style="color:#aaa;">Status:</span> <b style="color:{d_color};">{d_status}</b></div>
            <div style="margin-top:4px; color:#666; font-size:0.8rem;">Pre-decay: {pre_decay:.1f} → Final: {row['trajectory_score']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Sector Alpha detail (v2.3)
        sa_tag = row.get('sector_alpha_tag', 'NEUTRAL')
        sa_val = row.get('sector_alpha_value', 0)
        sa_colors = {'SECTOR_LEADER': '#FFD700', 'SECTOR_OUTPERFORM': '#00E676',
                     'SECTOR_ALIGNED': '#888', 'SECTOR_BETA': '#FF9800', 'SECTOR_LAGGARD': '#FF1744'}
        sa_icons = {'SECTOR_LEADER': '👑', 'SECTOR_OUTPERFORM': '⬆️',
                    'SECTOR_ALIGNED': '➖', 'SECTOR_BETA': '🏷️', 'SECTOR_LAGGARD': '📉'}
        sa_color = sa_colors.get(sa_tag, '#888')
        sa_icon = sa_icons.get(sa_tag, '➖')
        st.markdown(f"""
        ##### 🏛️ Sector Alpha
        <div style="background:#1e1e2e; border-radius:10px; padding:12px; border:1px solid {sa_color};">
            <div style="display:flex; justify-content:space-between;">
                <div><span style="color:#aaa;">Sector:</span> <b style="color:#fff;">{row.get('sector', 'N/A')}</b></div>
                <div><span style="color:#aaa;">Alpha:</span> <b style="color:{sa_color};">{sa_val:+.1f}</b></div>
            </div>
            <div style="margin-top:6px;"><span style="color:#aaa;">Classification:</span> <b style="color:{sa_color};">{sa_icon} {sa_tag}</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Latest patterns from Wave Detection
        if row.get('latest_patterns', ''):
            st.markdown("##### 🏷️ Latest Wave Detection Patterns")
            st.markdown(f"<div class='pattern-tag'>{row['latest_patterns']}</div>", unsafe_allow_html=True)

    # ── Week-by-Week Table ──
    with st.expander("📅 Week-by-Week Rank History", expanded=False):
        week_data = {
            'Date': h['dates'],
            'Rank': [int(r) for r in h['ranks']],
            'Master Score': [round(s, 1) for s in h['scores']],
            'Price': [round(p, 1) for p in h['prices']],
            'Total Stocks': h['total_per_week']
        }
        # Add percentile column
        week_data['Percentile'] = [
            round((1 - r / max(t, 1)) * 100, 1)
            for r, t in zip(h['ranks'], h['total_per_week'])
        ]
        # Add week-over-week change
        wk_changes = [0] + [int(h['ranks'][i - 1] - h['ranks'][i]) for i in range(1, len(h['ranks']))]
        week_data['Δ Rank'] = wk_changes

        # v2.3: Add return data
        def _safe_round(lst, decimals=1):
            return [round(v, decimals) if v is not None and not np.isnan(v) else None for v in lst]
        if 'ret_7d' in h and h['ret_7d']:
            week_data['Ret 7d%'] = _safe_round(h['ret_7d'])
        if 'ret_30d' in h and h['ret_30d']:
            week_data['Ret 30d%'] = _safe_round(h['ret_30d'])

        wk_df = pd.DataFrame(week_data)
        wk_df = wk_df.iloc[::-1]  # Latest first
        wk_col_config = {
            'Δ Rank': st.column_config.NumberColumn(format="%+d"),
            'Percentile': st.column_config.ProgressColumn('Percentile', min_value=0, max_value=100, format="%.1f")
        }
        if 'Ret 7d%' in wk_df.columns:
            wk_col_config['Ret 7d%'] = st.column_config.NumberColumn(format="%.1f%%")
        if 'Ret 30d%' in wk_df.columns:
            wk_col_config['Ret 30d%'] = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(wk_df, column_config=wk_col_config, hide_index=True, use_container_width=True)

    # ── Comparison Mode ──
    with st.expander("⚖️ Compare with Other Stocks", expanded=False):
        compare_labels = [l for l in labels if l != selected_label]
        compare_selections = st.multiselect("Select stocks to compare (max 4)",
                                             compare_labels, max_selections=4,
                                             key='compare_select')
        if compare_selections:
            compare_tickers = [label_map[l] for l in compare_selections]
            _render_comparison_chart(ticker, compare_tickers, histories, traj_df)


def _render_trajectory_chart(h: dict, ticker: str):
    """Render main trajectory chart with rank (inverted) + master score"""
    dates = h['dates']
    ranks = h['ranks']
    scores = h['scores']
    prices = h['prices']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Rank line (inverted y-axis: lower rank = higher on chart)
    fig.add_trace(go.Scatter(
        x=dates, y=ranks,
        mode='lines+markers',
        name='Rank',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=7),
        fill='tozeroy',
        fillcolor='rgba(255,107,53,0.1)',
        hovertemplate='Date: %{x}<br>Rank: #%{y}<extra></extra>'
    ), secondary_y=False)

    # Master score overlay
    fig.add_trace(go.Scatter(
        x=dates, y=scores,
        mode='lines+markers',
        name='Master Score',
        line=dict(color='#00C853', width=2, dash='dot'),
        marker=dict(size=5),
        opacity=0.7,
        hovertemplate='Date: %{x}<br>Score: %{y:.1f}<extra></extra>'
    ), secondary_y=True)

    # Best rank annotation
    best_idx = int(np.argmin(ranks))
    fig.add_annotation(
        x=dates[best_idx], y=ranks[best_idx],
        text=f"Best: #{int(ranks[best_idx])}",
        showarrow=True, arrowhead=2,
        font=dict(color='#FFD700', size=11),
        bgcolor='rgba(0,0,0,0.7)', bordercolor='#FFD700'
    )

    fig.update_layout(
        height=400,
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.15),
        margin=dict(t=20, b=60, l=60, r=60),
        xaxis=dict(title='Week', tickangle=-45)
    )
    fig.update_yaxes(title_text="Rank (lower = better)", autorange="reversed",
                     secondary_y=False, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(title_text="Master Score", secondary_y=True,
                     gridcolor='rgba(255,255,255,0.03)')

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

def render_funnel_tab(traj_df: pd.DataFrame, histories: dict, metadata: dict):
    """Render the 3-Stage Selection Funnel with visual pipeline"""
    
    st.markdown("### 🎯 3-Stage Selection Funnel")
    st.markdown("*Systematic filtering: Discovery → Validation → Final Buys*")
    
    # ── Funnel Configuration ──
    with st.expander("⚙️ Funnel Configuration", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown("**Stage 1: Discovery**")
            s1_score = st.number_input("Min Trajectory Score", 30, 100, FUNNEL_DEFAULTS['stage1_score'], key='f_s1')
            s1_patterns = st.multiselect(
                "Include Patterns (OR condition)",
                ['rocket', 'breakout', 'stable_elite', 'at_peak', 'steady_climber'],
                default=FUNNEL_DEFAULTS['stage1_patterns'], key='f_s1_pat'
            )
        with fc2:
            st.markdown("**Stage 2: Validation**")
            s2_tq = st.number_input("Min Trend Quality", 30, 100, FUNNEL_DEFAULTS['stage2_tq'], key='f_s2_tq')
            s2_ms = st.number_input("Min Master Score", 20, 100, FUNNEL_DEFAULTS['stage2_master_score'], key='f_s2_ms')
            s2_rules = st.number_input("Min Rules (of 5)", 2, 5, FUNNEL_DEFAULTS['stage2_min_rules'], key='f_s2_r')
        with fc3:
            st.markdown("**Stage 3: Final Filter**")
            s3_tq = st.number_input("Min TQ (strict)", 50, 100, FUNNEL_DEFAULTS['stage3_tq'], key='f_s3_tq')
            s3_leader = st.checkbox("Require Leader Pattern", FUNNEL_DEFAULTS['stage3_require_leader'], key='f_s3_l')
            s3_dt = st.number_input("No DOWNTREND (weeks)", 1, 10, FUNNEL_DEFAULTS['stage3_no_downtrend_weeks'], key='f_s3_dt')
    
    funnel_config = {
        'stage1_score': s1_score, 'stage1_patterns': s1_patterns,
        'stage2_tq': s2_tq, 'stage2_master_score': s2_ms, 'stage2_min_rules': s2_rules,
        'stage3_tq': s3_tq, 'stage3_require_leader': s3_leader, 'stage3_no_downtrend_weeks': s3_dt
    }
    
    # Execute funnel
    stage1, stage2, stage3 = run_funnel(traj_df, histories, funnel_config)
    
    s1_count = len(stage1)
    s2_total = len(stage2)
    s2_pass = len(stage2[stage2['s2_pass']]) if not stage2.empty and 's2_pass' in stage2.columns else 0
    s3_total = len(stage3)
    s3_pass = len(stage3[stage3['final_pass']]) if not stage3.empty and 'final_pass' in stage3.columns else 0
    
    # ── Visual Funnel Chart ──
    st.markdown("---")
    
    fig_funnel = go.Figure(go.Funnel(
        y=['📊 All Stocks', '🔍 Stage 1: Discovery', '✅ Stage 2: Validated', '🏆 Stage 3: Final Buys'],
        x=[len(traj_df), s1_count, s2_pass, s3_pass],
        textinfo='value+percent initial',
        textposition='inside',
        marker=dict(
            color=['#555555', '#2196F3', '#FF9800', '#00C853'],
            line=dict(width=2, color='#333')
        ),
        connector=dict(line=dict(color='#444', width=2))
    ))
    fig_funnel.update_layout(
        title="Selection Funnel Pipeline",
        height=350,
        template='plotly_dark',
        margin=dict(t=50, b=20, l=20, r=20),
        font=dict(size=14)
    )
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    # ── Funnel Stats ──
    ks1, ks2, ks3, ks4 = st.columns(4)
    with ks1:
        st.markdown(f"""<div class="funnel-stat">
            <div class="funnel-stat-value" style="color:#2196F3;">{s1_count}</div>
            <div class="funnel-stat-label">Stage 1 Discovery</div></div>""", unsafe_allow_html=True)
    with ks2:
        rate = round(s2_pass / max(s1_count, 1) * 100, 1)
        st.markdown(f"""<div class="funnel-stat">
            <div class="funnel-stat-value" style="color:#FF9800;">{s2_pass}</div>
            <div class="funnel-stat-label">Stage 2 Validated ({rate}%)</div></div>""", unsafe_allow_html=True)
    with ks3:
        rate3 = round(s3_pass / max(s2_pass, 1) * 100, 1)
        st.markdown(f"""<div class="funnel-stat">
            <div class="funnel-stat-value" style="color:#00C853;">{s3_pass}</div>
            <div class="funnel-stat-label">Stage 3 Final ({rate3}%)</div></div>""", unsafe_allow_html=True)
    with ks4:
        overall = round(s3_pass / max(len(traj_df), 1) * 100, 2)
        st.markdown(f"""<div class="funnel-stat">
            <div class="funnel-stat-value" style="color:#FFD700;">{overall}%</div>
            <div class="funnel-stat-label">Selection Rate</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── STAGE 3: FINAL BUYS (Show First — Most Important) ──
    st.markdown("### 🏆 FINAL BUYS — Stage 3 Passed")
    if s3_pass > 0:
        final_buys = stage3[stage3['final_pass']].copy()
        for idx, row in final_buys.iterrows():
            h = histories.get(row['ticker'], {})
            pats = row.get('latest_pats', '')
            # Truncate patterns for display
            display_pats = pats[:120] + '...' if len(pats) > 120 else pats
            
            st.markdown(f"""
            <div class="final-buy-card">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <div>
                        <h3 style="margin:0; color:#00C853;">🏆 {row['ticker']}</h3>
                        <p style="margin:2px 0; color:#ccc;">{row.get('company_name', '')}</p>
                        <p style="margin:0; color:#888;">{row.get('category', '')} • {row.get('sector', '')}</p>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:2rem; font-weight:800; color:#00C853;">{row['trajectory_score']:.1f}</div>
                        <div style="color:#aaa; font-size:0.8rem;">T-SCORE</div>
                        <div style="color:#FFD700; font-size:1.1rem;">Rank #{row['current_rank']}</div>
                    </div>
                </div>
                <div style="margin-top:10px; padding-top:10px; border-top:1px solid #2a5a2a;">
                    <span style="color:#aaa; font-size:0.85rem;">
                        TQ: {row.get('latest_tq', 0):.0f} | 
                        TMI: {row['tmi']:.0f} | 
                        Grade: {row['grade_emoji']} {row['grade']} | 
                        Pattern: {row['pattern']} |
                        Rules: {row.get('rules_passed', 0)}/5
                    </span>
                </div>
                <div style="margin-top:5px;">
                    <span style="color:#777; font-size:0.8rem;">{display_pats}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Final buys table
        if len(final_buys) > 0:
            fb_cols = ['ticker', 'company_name', 'category', 'trajectory_score', 'grade',
                       'pattern', 'tmi', 'current_rank', 'best_rank', 'rank_change',
                       'positional', 'trend', 'latest_tq', 'rules_passed']
            fb_display = final_buys[[c for c in fb_cols if c in final_buys.columns]].copy()
            fb_display.columns = [c.replace('_', ' ').title() for c in fb_display.columns]
            st.dataframe(fb_display, hide_index=True, use_container_width=True)
    else:
        st.warning("⚠️ No stocks passed all 3 stages. Try relaxing Stage 3 criteria in the config above.")
    
    st.markdown("---")
    
    # ── STAGE 2: Validation Details ──
    with st.expander(f"✅ Stage 2: Validation Results ({s2_pass} passed / {s2_total} tested)", expanded=False):
        if not stage2.empty:
            st.markdown("**5 Rules:** TQ≥{} | Not DOWNTREND | MS≥{} | Rank Δ≥-20 | Volume Pattern".format(s2_tq, s2_ms))
            
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
            st.info("No stocks entered Stage 2")
    
    # ── STAGE 1: Discovery ──
    with st.expander(f"🔍 Stage 1: Discovery ({s1_count} candidates)", expanded=False):
        if not stage1.empty:
            s1_cols = ['ticker', 'company_name', 'category', 'trajectory_score', 
                       'grade', 'pattern', 'tmi', 'current_rank', 'positional']
            s1_display = stage1[[c for c in s1_cols if c in stage1.columns]].head(100).copy()
            s1_display.columns = ['Ticker', 'Company', 'Category', 'T-Score', 
                                  'Grade', 'Pattern', 'TMI', 'Rank', 'Positional']
            s1_display['Company'] = s1_display['Company'].str[:30]
            st.dataframe(s1_display, column_config={
                'T-Score': st.column_config.ProgressColumn('T-Score', min_value=0, max_value=100, format="%.1f"),
                'TMI': st.column_config.ProgressColumn('TMI', min_value=0, max_value=100, format="%.0f"),
                'Positional': st.column_config.ProgressColumn('Positional', min_value=0, max_value=100, format="%.0f"),
            }, hide_index=True, use_container_width=True, height=400)
        else:
            st.info("No stocks passed Stage 1 discovery threshold")
    
    # ── Stage 3 Failures (near misses) ──
    if not stage3.empty:
        near_misses = stage3[~stage3['final_pass']].copy()
        if len(near_misses) > 0:
            with st.expander(f"📋 Stage 3 Near Misses ({len(near_misses)} stocks — passed Stage 2 but failed Stage 3)", expanded=False):
                nm_cols = ['ticker', 'company_name', 'trajectory_score', 's3_detail', 'latest_tq', 'current_rank']
                nm_display = near_misses[[c for c in nm_cols if c in near_misses.columns]].copy()
                nm_display.columns = ['Ticker', 'Company', 'T-Score', 'Stage 3 Detail', 'TQ', 'Rank']
                nm_display['Company'] = nm_display['Company'].str[:30]
                st.dataframe(nm_display, hide_index=True, use_container_width=True)


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
    tab_ranking, tab_search, tab_funnel, tab_export, tab_about = st.tabs([
        "🏆 Rankings", "🔍 Search & Analyze", "🎯 Funnel", "📤 Export", "ℹ️ About"
    ])

    with tab_ranking:
        render_rankings_tab(filtered_df, traj_df, histories, metadata)

    with tab_search:
        render_search_tab(traj_df, histories, dates_iso)

    with tab_funnel:
        render_funnel_tab(traj_df, histories, metadata)

    with tab_export:
        render_export_tab(filtered_df, traj_df, histories)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
