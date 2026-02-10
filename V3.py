# ═══════════════════════════════════════════════════════════════════════════
#  WAVE GAINER SCANNER
#  Finds the next big winners by studying what past gainers looked like
#  BEFORE they gained. Data-driven. No guessing.
#  Companion to WAVE Detection 3.0
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="WAVE Gainer Scanner", page_icon="🎯",
    layout="wide", initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
#  STYLING
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0a0e1a; color: #e2e8f0; }
    [data-testid="stSidebar"] { background: #111827; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b; border-radius: 8px; color: #94a3b8;
        padding: 8px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #059669, #0d9488) !important;
        color: white !important;
    }
    .hero {
        background: linear-gradient(135deg, #059669 0%, #0d9488 50%, #0ea5e9 100%);
        padding: 24px 32px; border-radius: 16px; margin-bottom: 24px; text-align: center;
    }
    .hero h1 { margin: 0; font-size: 2.2rem; color: white; letter-spacing: -0.5px; }
    .hero p { margin: 6px 0 0; color: rgba(255,255,255,0.85); font-size: 0.95rem; }
    .scard {
        background: #1e293b; border-radius: 12px; padding: 18px;
        margin: 10px 0; border-left: 5px solid #059669;
    }
    .scard.high { border-left-color: #22c55e; }
    .scard.med { border-left-color: #f59e0b; }
    .scard.low { border-left-color: #64748b; }
    .scard.conf { border-left-color: #38bdf8; background: #172033; }
    .big { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
    .big.green { color: #22c55e; }
    .big.amber { color: #f59e0b; }
    .big.gray { color: #94a3b8; }
    .big.blue { color: #38bdf8; }
    .muted { color: #64748b; font-size: 0.82rem; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 6px;
        font-size: 0.75rem; font-weight: 700; margin-right: 4px;
    }
    .badge-new { background: #059669; color: white; }
    .badge-confirmed { background: #0ea5e9; color: white; }
    .badge-loser { background: #dc2626; color: white; }
    div[data-testid="stMetric"] { background: #1e293b; border-radius: 10px; padding: 12px; }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
SCORE_COLS = [
    'position_score', 'volume_score', 'momentum_score', 'breakout_score',
    'acceleration_score', 'rvol_score', 'trend_quality'
]

ANALYSIS_FEATURES = [
    'rank', 'master_score', 'position_score', 'volume_score', 'momentum_score',
    'breakout_score', 'acceleration_score', 'rvol_score', 'trend_quality',
    'rvol', 'momentum_harmony', 'from_low_pct', 'from_high_pct',
    'ret_7d', 'ret_30d', 'overall_market_strength',
]

PATTERN_CHECKS = {
    'p_stealth': 'stealth',
    'p_institutional': 'institutional',
    'p_cat_leader': 'cat leader',
    'p_market_leader': 'market leader',
    'p_vol_explosion': 'vol explosion',
    'p_velocity': 'velocity',
    'p_premium_mom': 'premium momentum',
    'p_golden_cross': 'golden cross',
    'p_mom_wave': 'momentum wave',
    'p_acceleration': 'acceleration',
    'p_garp': 'garp',
    'p_value_mom': 'value momentum',
    'p_liquid_leader': 'liquid leader',
    'p_inst_tsunami': 'tsunami',
    'p_capitulation': 'capitulation',
    'p_distribution': 'distribution',
    'p_rotation': 'rotation',
    'p_52w_high': '52w high',
    'p_52w_low': '52w low',
    'p_high_pe': 'high pe',
    'p_low_pe': 'low pe',
    'p_mom_diverge': 'diverge',
}

FEAT_DISPLAY = {
    'rank': 'Rank', 'master_score': 'Master Score', 'position_score': 'Position',
    'volume_score': 'Volume', 'momentum_score': 'Momentum', 'breakout_score': 'Breakout',
    'acceleration_score': 'Acceleration', 'rvol_score': 'RVOL Score',
    'trend_quality': 'Trend Quality', 'rvol': 'Relative Volume',
    'momentum_harmony': 'Harmony', 'from_low_pct': 'From 52W Low %',
    'from_high_pct': 'From 52W High %', 'ret_7d': '7D Return',
    'ret_30d': '30D Return', 'overall_market_strength': 'Market Strength',
    'rank_velocity': 'Rank Velocity', 'score_velocity': 'Score Velocity',
}


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_weekly_csvs(file_contents):
    """Load and combine weekly CSV files with date parsing."""
    frames = []
    for name, data in file_contents:
        try:
            from io import BytesIO
            df = pd.read_csv(BytesIO(data))
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', name)
            if not date_match:
                continue
            df['week'] = pd.to_datetime(date_match.group(1))
            df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
            df['ticker'] = df['ticker'].str.replace(r'\.0$', '', regex=True)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(['ticker', 'week']).reset_index(drop=True)
    return panel


def parse_outcome_file(uploaded):
    """Parse gainer/loser CSV → DataFrame with symbol and return_pct."""
    df = pd.read_csv(uploaded)
    cols = df.columns.tolist()
    sym_col = cols[0]
    company_col = cols[1] if len(cols) > 1 else cols[0]
    ret_col = cols[-1]

    df['symbol'] = df[sym_col].astype(str).str.strip().str.upper()
    df['symbol'] = df['symbol'].str.replace(r'\.0$', '', regex=True)
    df['company'] = df[company_col].astype(str).str.strip()
    df['return_pct'] = (
        df[ret_col].astype(str)
        .str.replace('%', '', regex=False)
        .str.replace('+', '', regex=False)
        .str.strip()
    )
    df['return_pct'] = pd.to_numeric(df['return_pct'], errors='coerce')
    return df[['symbol', 'company', 'return_pct']].dropna(subset=['return_pct'])


def parse_outcome_filename(filename):
    """Extract date and period from gainer/loser filename."""
    date_match = re.search(
        r'(\d+)\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{4})',
        filename, re.IGNORECASE
    )
    period_match = re.search(r'\((\d+)\s+months?\)', filename, re.IGNORECASE)
    months_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
    }
    if date_match and period_match:
        day = int(date_match.group(1))
        month = months_map[date_match.group(2).upper()]
        year = int(date_match.group(3))
        return datetime(year, month, day), int(period_match.group(1))
    return None, None


def match_outcomes_to_weekly(outcome_df, panel):
    """Match gainer/loser symbols to weekly CSV tickers.
    Returns dict {ticker: return_pct} and list of unmatched."""
    weekly_tickers = set(panel['ticker'].unique())

    # Build company name lookup
    company_lookup = {}
    for _, row in panel.drop_duplicates('ticker').iterrows():
        name = str(row.get('company_name', '')).lower().strip()
        name_clean = re.sub(r'\b(ltd|limited|pvt|private|inc|corp|co)\b', '', name).strip()
        if name_clean:
            company_lookup[name_clean] = row['ticker']

    matched = {}
    unmatched = []

    for _, row in outcome_df.iterrows():
        sym = str(row['symbol']).strip().upper()
        company = str(row['company']).strip()
        ret = row['return_pct']

        # Direct ticker match
        if sym in weekly_tickers:
            matched[sym] = ret
            continue

        # Company name fuzzy match
        comp_clean = re.sub(
            r'\b(ltd|limited|pvt|private|inc|corp|co)\b',
            '', company.lower()
        ).strip()

        found = False
        # Substring match on significant words
        comp_words = [w for w in comp_clean.split() if len(w) > 2]
        if len(comp_words) >= 2:
            for name, ticker in company_lookup.items():
                if comp_words[0] in name and comp_words[1] in name:
                    matched[ticker] = ret
                    found = True
                    break
        elif len(comp_words) == 1 and len(comp_words[0]) > 4:
            for name, ticker in company_lookup.items():
                if comp_words[0] in name:
                    matched[ticker] = ret
                    found = True
                    break

        if not found:
            unmatched.append((sym, company))

    return matched, unmatched


def add_pattern_flags(panel):
    """Add binary pattern columns from the patterns string."""
    pat_str = panel['patterns'].fillna('').astype(str).str.lower()
    for col, keyword in PATTERN_CHECKS.items():
        panel[col] = pat_str.str.contains(keyword, regex=False).astype(int)
    return panel


def add_trajectory_features(panel):
    """Add rank/score velocity and freshness features."""
    panel = panel.sort_values(['ticker', 'week'])

    panel['rank_prev1'] = panel.groupby('ticker')['rank'].shift(1)
    panel['rank_prev4'] = panel.groupby('ticker')['rank'].shift(4)
    panel['score_prev1'] = panel.groupby('ticker')['master_score'].shift(1)
    panel['score_prev4'] = panel.groupby('ticker')['master_score'].shift(4)

    panel['rank_delta_1w'] = panel['rank_prev1'] - panel['rank']   # positive = improving
    panel['rank_delta_4w'] = panel['rank_prev4'] - panel['rank']
    panel['score_delta_1w'] = panel['master_score'] - panel['score_prev1']
    panel['score_delta_4w'] = panel['master_score'] - panel['score_prev4']

    panel['rank_velocity'] = (panel['rank_delta_4w'] / 4).fillna(panel['rank_delta_1w'])
    panel['score_velocity'] = (panel['score_delta_4w'] / 4).fillna(panel['score_delta_1w'])

    panel['in_top100'] = (panel['rank'] <= 100).astype(int)
    panel['weeks_in_top100'] = panel.groupby('ticker')['in_top100'].cumsum()
    panel['is_fresh_top100'] = (
        (panel['weeks_in_top100'] == 1) & (panel['in_top100'] == 1)
    ).astype(int)
    panel['n_weeks_seen'] = panel.groupby('ticker').cumcount() + 1

    for col in ['rank_prev1', 'rank_prev4', 'score_prev1', 'score_prev4']:
        panel.drop(col, axis=1, inplace=True, errors='ignore')

    return panel


# ═══════════════════════════════════════════════════════════════════════════
#  GAINER FINGERPRINT ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class GainerFingerprint:
    """Learns what gainers looked like BEFORE they gained,
    then scores current stocks on similarity."""

    def __init__(self, panel, gainer_tickers, loser_tickers, pre_gain_weeks):
        self.features = [
            f for f in ANALYSIS_FEATURES + ['rank_velocity', 'score_velocity']
            if f in panel.columns
        ]
        self.pattern_cols = [c for c in PATTERN_CHECKS if c in panel.columns]
        self.profile = {}
        self.pattern_lifts = {}
        self.state_probs = {}
        self.feature_weights = {}
        self.gainer_tickers = set(gainer_tickers)
        self.loser_tickers = set(loser_tickers) if loser_tickers else set()
        self.n_gainers_found = 0
        self.n_pre_weeks = 0
        self.important_features = []
        self._build(panel, pre_gain_weeks)

    @staticmethod
    def _cohens_d(g1, g2):
        n1, n2 = len(g1), len(g2)
        if n1 < 3 or n2 < 3:
            return 0.0
        v1, v2 = g1.var(), g2.var()
        pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        return 0.0 if pooled < 1e-9 else (g1.mean() - g2.mean()) / pooled

    def _build(self, panel, pre_gain_weeks):
        if len(pre_gain_weeks) == 0:
            return
        early = panel[panel['week'].isin(pre_gain_weeks)]
        self.n_pre_weeks = len(pre_gain_weeks)

        g_data = early[early['ticker'].isin(self.gainer_tickers)]
        o_data = early[~early['ticker'].isin(self.gainer_tickers)]
        self.n_gainers_found = g_data['ticker'].nunique()
        if len(g_data) < 10:
            return

        # ── Numerical features ──
        for feat in self.features:
            if feat not in early.columns:
                continue
            g = g_data[feat].dropna()
            o = o_data[feat].dropna()
            if len(g) < 5 or len(o) < 5:
                continue
            d = self._cohens_d(g, o)
            self.profile[feat] = {
                'g_mean': g.mean(), 'g_std': max(g.std(), 1e-9),
                'g_q25': g.quantile(0.25), 'g_q75': g.quantile(0.75),
                'o_mean': o.mean(), 'o_std': max(o.std(), 1e-9),
                'effect_size': d,
                'direction': 'higher' if g.mean() > o.mean() else 'lower',
                'discriminative': abs(d) > 0.15,
            }

        abs_eff = {f: abs(p['effect_size']) for f, p in self.profile.items()
                   if p['discriminative']}
        if not abs_eff:
            # fallback: use all features
            abs_eff = {f: max(abs(p['effect_size']), 0.01)
                       for f, p in self.profile.items()}
            for f in self.profile:
                self.profile[f]['discriminative'] = True

        total = sum(abs_eff.values()) + 1e-9
        self.feature_weights = {f: v / total for f, v in abs_eff.items()}
        self.important_features = sorted(abs_eff, key=lambda x: -abs_eff[x])

        # ── Pattern lifts ──
        for col in self.pattern_cols:
            if col not in early.columns:
                continue
            g_rate = g_data[col].mean()
            o_rate = o_data[col].mean()
            if o_rate > 0.005:
                lift = g_rate / o_rate
            elif g_rate > 0.005:
                lift = 5.0
            else:
                lift = 1.0
            self.pattern_lifts[col] = {
                'gainer_pct': g_rate * 100,
                'other_pct': o_rate * 100,
                'lift': lift,
            }

        # ── Market state analysis ──
        if 'market_state' in early.columns:
            for state in early['market_state'].dropna().unique():
                g_pct = (g_data['market_state'] == state).mean()
                o_pct = (o_data['market_state'] == state).mean()
                self.state_probs[state] = {
                    'gainer_pct': g_pct * 100,
                    'other_pct': o_pct * 100,
                    'lift': g_pct / (o_pct + 1e-9),
                }

    # ── Vectorised batch scoring ──
    def score_batch(self, df):
        """Score all stocks in a DataFrame at once (0-100)."""
        n = len(df)
        if not self.profile or n == 0:
            return np.zeros(n)

        scores = np.zeros(n)
        total_w = 0.0

        for feat, prof in self.profile.items():
            if not prof['discriminative'] or feat not in df.columns:
                continue
            w = self.feature_weights.get(feat, 0.01)
            vals = df[feat].values.astype(float)
            g_z = np.abs(vals - prof['g_mean']) / prof['g_std']
            o_z = np.abs(vals - prof['o_mean']) / prof['o_std']
            raw = o_z / (g_z + o_z + 0.5)

            # IQR bonus
            if prof['direction'] == 'higher':
                mask = vals >= prof['g_q25']
            else:
                mask = vals <= prof['g_q75']
            raw = np.where(mask, np.minimum(raw * 1.2, 1.0), raw)
            raw = np.clip(raw, 0, 1)
            raw = np.where(np.isnan(raw), 0.5, raw)

            scores += raw * w
            total_w += w

        if total_w > 0:
            scores = (scores / total_w) * 70

        # Pattern bonus (capped 15)
        pat_bonus = np.zeros(n)
        for col, info in self.pattern_lifts.items():
            if col in df.columns and info['lift'] > 1.5:
                pat_bonus += np.minimum(info['lift'] * 1.5, 5) * df[col].values
        scores += np.minimum(pat_bonus, 15)

        # Market state bonus (capped 10)
        if 'market_state' in df.columns:
            for state, info in self.state_probs.items():
                if info.get('lift', 1) > 1.2:
                    mask = df['market_state'].values == state
                    scores[mask] += min((info['lift'] - 1) * 10, 10)

        # Rank velocity bonus (capped 5)
        if 'rank_velocity' in df.columns:
            rv = df['rank_velocity'].fillna(0).values
            scores += np.minimum(np.maximum(rv, 0) / 10, 5)

        return np.clip(np.round(scores, 1), 0, 100)

    # ── Single-stock explain ──
    def explain(self, row):
        """Return list of (class, reason) tuples for a single stock."""
        reasons = []
        for feat in self.important_features[:8]:
            prof = self.profile[feat]
            val = row.get(feat, np.nan) if isinstance(row, dict) else row[feat] if feat in row.index else np.nan
            if pd.isna(val):
                continue
            g_z = abs(val - prof['g_mean']) / prof['g_std']
            o_z = abs(val - prof['o_mean']) / prof['o_std']
            raw = o_z / (g_z + o_z + 0.5)
            name = FEAT_DISPLAY.get(feat, feat.replace('_', ' ').title())
            if raw > 0.55:
                reasons.append(
                    ('ok', f"✅ {name}: {val:.0f} — gainers averaged {prof['g_mean']:.0f} pre-gain")
                )
            elif raw < 0.4:
                reasons.append(
                    ('bad', f"⚠️ {name}: {val:.0f} — "
                     f"{'below' if prof['direction'] == 'higher' else 'above'} "
                     f"gainer avg {prof['g_mean']:.0f}")
                )

        for col, info in sorted(self.pattern_lifts.items(), key=lambda x: -x[1]['lift']):
            v = row.get(col, 0) if isinstance(row, dict) else (row[col] if col in row.index else 0)
            if v == 1 and info['lift'] > 1.5:
                pat_name = col.replace('p_', '').replace('_', ' ').upper()
                reasons.append(
                    ('ok', f"🏷️ {pat_name} — {info['lift']:.1f}x more common in pre-gain stocks")
                )

        state_val = row.get('market_state', '') if isinstance(row, dict) else (row['market_state'] if 'market_state' in row.index else '')
        si = self.state_probs.get(state_val, {})
        if si.get('lift', 1) > 1.3:
            reasons.append(('ok', f"🌐 {state_val} — {si['lift']:.1f}x more common for gainers"))
        elif si.get('lift', 1) < 0.7 and state_val:
            reasons.append(('bad', f"🌐 {state_val} — less common for gainers ({si['lift']:.1f}x)"))

        rv = row.get('rank_velocity', 0) if isinstance(row, dict) else (row['rank_velocity'] if 'rank_velocity' in row.index else 0)
        if pd.notna(rv):
            if rv > 30:
                reasons.append(('ok', f"🚀 Rank improving fast (+{rv:.0f}/week)"))
            elif rv < -30:
                reasons.append(('bad', f"📉 Rank declining ({rv:.0f}/week)"))

        return reasons

    # ── Historical validation ──
    def validate_hit_rate(self, panel, weeks, top_ns=(20, 50, 100)):
        """For each historical week, check what % of gainers fall in top-N by match score."""
        results = []
        for week in weeks:
            wd = panel[panel['week'] == week].copy()
            if len(wd) < 100:
                continue
            wd['_ms'] = self.score_batch(wd)
            for n in top_ns:
                top_set = set(wd.nlargest(n, '_ms')['ticker'])
                g_present = set(wd['ticker']) & self.gainer_tickers
                g_caught = top_set & self.gainer_tickers
                rate = len(g_caught) / max(len(g_present), 1) * 100
                results.append({
                    'week': week, 'top_n': n,
                    'gainers_caught': len(g_caught),
                    'gainers_present': len(g_present),
                    'hit_rate': rate,
                })
        return pd.DataFrame(results) if results else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def render_stock_card(row, fp, gainer_returns, card_class):
    """Render one stock card as HTML."""
    score = row['match_score']
    sc = 'green' if score >= 70 else 'amber' if score >= 50 else 'gray'

    ticker = row['ticker']
    company = str(row.get('company_name', ''))[:45]
    rank_val = row.get('rank', 0)
    price_val = row.get('price', 0)
    sector = str(row.get('sector', ''))[:25]
    ms = row.get('master_score', 0)
    bs = row.get('breakout_score', 0)
    rd = row.get('rank_delta_1w', 0)
    state = row.get('market_state', '')
    ret7 = row.get('ret_7d', 0)
    ret30 = row.get('ret_30d', 0)
    patterns = str(row.get('patterns', ''))

    # Badge
    if row.get('is_known_gainer', 0) == 1:
        rp = gainer_returns.get(ticker, 0)
        badge = f'<span class="badge badge-confirmed">CONFIRMED GAINER +{rp:.0f}%</span>'
    elif score >= 70:
        badge = '<span class="badge badge-new">🆕 NEW CANDIDATE</span>'
    else:
        badge = ''

    # Rank delta arrow
    if pd.notna(rd) and rd != 0:
        rd_html = (f'<span style="color:#4ade80">▲{rd:.0f}</span>' if rd > 0
                   else f'<span style="color:#f87171">▼{abs(rd):.0f}</span>')
    else:
        rd_html = ''

    # Reasons
    reasons = fp.explain(row)
    reasons_html = ''
    for cls, reason in reasons[:5]:
        color = '#4ade80' if cls == 'ok' else '#f87171'
        reasons_html += f'<div style="color:{color};font-size:0.85rem;margin:2px 0">{reason}</div>'

    # Patterns (shortened)
    pat_html = ''
    if patterns and patterns != 'nan' and len(patterns) > 3:
        pat_html = f'<div class="muted" style="margin-top:6px">{patterns[:120]}</div>'

    return f"""
    <div class="scard {card_class}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div style="flex:1">
                <div>{badge}</div>
                <div style="font-size:1.3rem;font-weight:800;color:#f1f5f9;margin:4px 0">{ticker}</div>
                <div class="muted">{company}</div>
                <div style="margin-top:8px">
                    Rank <b>#{rank_val:.0f}</b> {rd_html} · Score <b>{ms:.0f}</b> · Breakout <b>{bs:.0f}</b> · ₹{price_val:,.0f}
                </div>
                <div class="muted">{state} · {sector} · 7d:{ret7:+.1f}% · 30d:{ret30:+.1f}%</div>
                {pat_html}
            </div>
            <div style="text-align:right;min-width:85px">
                <div class="big {sc}">{score:.0f}%</div>
                <div class="muted">match</div>
            </div>
        </div>
        <div style="margin-top:10px;border-top:1px solid #334155;padding-top:8px">
            {reasons_html}
        </div>
    </div>
    """


def display_stock_cards(df, fp, gainer_returns, card_class):
    """Display a batch of stock cards."""
    for _, row in df.iterrows():
        st.markdown(render_stock_card(row, fp, gainer_returns, card_class),
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 📁 Upload Data")
weekly_files = st.sidebar.file_uploader(
    "Weekly CSV Files (from WAVE Detection)",
    type=['csv'], accept_multiple_files=True,
    help="Upload all your weekly stock screening CSV files",
)

gainer_file = st.sidebar.file_uploader(
    "Gainer File (3m or 6m)",
    type=['csv'],
    help="e.g., LATEST GAINER 6 FEB 2026 (3 months).csv",
)

loser_file = st.sidebar.file_uploader(
    "Loser File (optional)",
    type=['csv'],
    help="Optional: e.g., LATEST LOSERS 6 FEB 2026 (3 months).csv",
)

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Settings")
min_match = st.sidebar.slider("Minimum Match %", 0, 80, 30, 5)
max_stocks = st.sidebar.slider("Max Stocks to Show", 10, 200, 50, 10)

# ═══════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="hero">
    <h1>🎯 WAVE GAINER SCANNER</h1>
    <p>Finds the next big winners by studying what past gainers looked like BEFORE they gained</p>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  GATE: require uploads
# ═══════════════════════════════════════════════════════════════════════════
if not weekly_files or not gainer_file:
    st.info("👈 Upload your **Weekly CSV files** and **Gainer file** in the sidebar to begin.")
    st.markdown("""
### How it works
1. **Upload Weekly CSVs** — your WAVE Detection weekly snapshots (more weeks = better)
2. **Upload Gainer File** — the LATEST GAINER CSV (tells the system who actually gained)
3. **Get Predictions** — the system studies what gainers looked like *before* they gained, then finds current stocks with the same DNA

### What makes this different
- **Not guessing** — purely data-driven, learns from YOUR data
- **Shows NEW candidates** — stocks not yet in your gainer list that match the gainer DNA
- **Explains WHY** — each pick comes with clear reasons backed by data
- **Self-validating** — shows how many known gainers it would have caught
    """)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════
# Read file contents once (for caching stability)
weekly_contents = [(f.name, f.read()) for f in weekly_files]
for f in weekly_files:
    f.seek(0)

with st.spinner("Loading weekly data…"):
    panel = load_weekly_csvs(weekly_contents)

if len(panel) == 0:
    st.error("❌ No valid weekly CSV files found. Ensure filenames contain dates like Stocks_Weekly_2025-08-30.")
    st.stop()

weeks = sorted(panel['week'].unique())
n_weeks = len(weeks)
latest_week = weeks[-1]
n_stocks = panel['ticker'].nunique()

# ── Parse gainer file ──
gainer_df = parse_outcome_file(gainer_file)
gainer_filename = gainer_file.name if hasattr(gainer_file, 'name') else 'unknown'
outcome_date, gain_months = parse_outcome_filename(gainer_filename)

# ── Parse loser file ──
loser_tickers = set()
loser_matched_returns = {}
if loser_file:
    loser_df = parse_outcome_file(loser_file)
    loser_matched_returns, _ = match_outcomes_to_weekly(loser_df, panel)
    loser_tickers = set(loser_matched_returns.keys())

# ── Match gainers ──
with st.spinner("Matching gainers to weekly data…"):
    gainer_matched, gainer_unmatched = match_outcomes_to_weekly(gainer_df, panel)
    gainer_tickers = set(gainer_matched.keys())

# ── Pre-gain window ──
if outcome_date and gain_months:
    cutoff = outcome_date - timedelta(days=gain_months * 30)
    pre_gain_weeks = sorted([w for w in weeks if w < pd.Timestamp(cutoff)])
else:
    mid = max(n_weeks // 2, 3)
    pre_gain_weeks = sorted(weeks[:mid])
    cutoff = pre_gain_weeks[-1]

if len(pre_gain_weeks) < 3:
    pre_gain_weeks = sorted(weeks[:max(3, n_weeks // 3)])
    cutoff = pre_gain_weeks[-1]

# ── Enrich ──
with st.spinner("Building features…"):
    panel = add_pattern_flags(panel)
    panel = add_trajectory_features(panel)

# ── Build fingerprint ──
with st.spinner("🧬 Learning gainer DNA…"):
    fp = GainerFingerprint(panel, gainer_tickers, loser_tickers, pre_gain_weeks)

# ── Score latest week ──
with st.spinner("🎯 Scoring all stocks…"):
    latest = panel[panel['week'] == latest_week].copy()
    latest['match_score'] = fp.score_batch(latest)
    latest = latest.sort_values('match_score', ascending=False)
    latest['is_known_gainer'] = latest['ticker'].isin(gainer_tickers).astype(int)
    latest['is_known_loser'] = latest['ticker'].isin(loser_tickers).astype(int)

# ── Sidebar summary ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Loaded")
st.sidebar.markdown(
    f"- **{n_weeks}** weeks ({weeks[0].strftime('%b %d, %Y')} → {latest_week.strftime('%b %d, %Y')})"
)
st.sidebar.markdown(f"- **{n_stocks:,}** unique stocks")
st.sidebar.markdown(f"- **{len(gainer_tickers)}/{len(gainer_df)}** gainers matched")
if gainer_unmatched:
    st.sidebar.caption(f"⚠️ {len(gainer_unmatched)} gainers not found in weekly data")
st.sidebar.markdown(f"- **{len(pre_gain_weeks)}** pre-gain weeks used")
if loser_tickers:
    st.sidebar.markdown(f"- **{len(loser_tickers)}** losers matched")
st.sidebar.markdown(f"- **{len(fp.important_features)}** discriminative features")

# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎯 Future Gainers", "🧬 Gainer DNA", "🔬 Stock X-Ray"])

# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#  TAB 1 — FUTURE GAINERS
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
with tab1:
    new_cands = latest[
        (latest['is_known_gainer'] == 0)
        & (latest['is_known_loser'] == 0)
        & (latest['match_score'] >= min_match)
    ]
    confirmed = latest[
        (latest['is_known_gainer'] == 1) & (latest['match_score'] >= min_match)
    ]

    n_high = int((new_cands['match_score'] >= 70).sum())
    n_med = int(((new_cands['match_score'] >= 50) & (new_cands['match_score'] < 70)).sum())
    n_watch = int(((new_cands['match_score'] >= min_match) & (new_cands['match_score'] < 50)).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🟢 High Match (>70%)", n_high)
    with c2:
        st.metric("🟡 Medium (50-70%)", n_med)
    with c3:
        st.metric("👀 Watch (<50%)", n_watch)
    with c4:
        st.metric("✅ Gainers Detected", f"{len(confirmed)}/{len(gainer_tickers)}")
    with c5:
        if fp.important_features:
            top_f = FEAT_DISPLAY.get(fp.important_features[0], fp.important_features[0])
            st.metric("Top Signal", top_f)

    st.markdown("---")

    # ═══ HIGH CONFIDENCE ═══
    if n_high > 0:
        st.markdown(f"### 🟢 HIGH CONFIDENCE — {n_high} New Candidates Match Gainer DNA")
        st.caption("NOT in your gainer list but closely match what gainers looked like before they gained")
        high_df = new_cands[new_cands['match_score'] >= 70].head(max_stocks)
        display_stock_cards(high_df, fp, gainer_matched, 'high')
    else:
        st.info("No stocks above 70% match currently. Try lowering the threshold or adding more weekly data.")

    # ═══ MEDIUM CONFIDENCE ═══
    if n_med > 0:
        st.markdown(f"### 🟡 MEDIUM CONFIDENCE — {n_med} Possible Future Gainers")
        with st.expander(f"Show {n_med} medium-confidence candidates"):
            med_df = new_cands[
                (new_cands['match_score'] >= 50) & (new_cands['match_score'] < 70)
            ].head(max_stocks)
            display_stock_cards(med_df, fp, gainer_matched, 'med')

    # ═══ WATCH LIST ═══
    if n_watch > 0:
        with st.expander(f"👀 {n_watch} stocks on watchlist ({min_match}-50% match)"):
            watch_df = new_cands[
                (new_cands['match_score'] >= min_match) & (new_cands['match_score'] < 50)
            ].head(max_stocks)
            display_stock_cards(watch_df, fp, gainer_matched, 'low')

    st.markdown("---")

    # ═══ CONFIRMED GAINERS (validation) ═══
    st.markdown(f"### ✅ System Validation — {len(confirmed)} Known Gainers Detected")
    st.caption("Stocks from your gainer file that score high → proves the DNA fingerprint works")
    if len(confirmed) > 0:
        with st.expander(f"Show {len(confirmed)} confirmed gainers"):
            display_stock_cards(confirmed.head(50), fp, gainer_matched, 'conf')
    else:
        st.warning("No known gainers scored above the threshold. The pre-gain data window may be too short.")

    # ═══ SECTOR CONCENTRATION ═══
    st.markdown("---")
    st.markdown("### 🗺️ Sector Concentration of Top Picks")
    top_for_sector = new_cands[new_cands['match_score'] >= 50]
    if len(top_for_sector) > 0 and 'sector' in top_for_sector.columns:
        sec = top_for_sector['sector'].value_counts().reset_index()
        sec.columns = ['Sector', 'Count']
        fig = px.bar(sec.head(15), x='Sector', y='Count', color='Count',
                     color_continuous_scale='Greens',
                     title='Top Gainer Candidates by Sector')
        fig.update_layout(height=350, template='plotly_dark', xaxis_tickangle=-35,
                          margin=dict(t=40, b=80))
        st.plotly_chart(fig, use_container_width=True)

    # ═══ DOWNLOAD ═══
    st.markdown("---")
    dl_cols = [
        'ticker', 'company_name', 'match_score', 'is_known_gainer', 'rank',
        'rank_delta_1w', 'master_score', 'breakout_score', 'momentum_score',
        'price', 'ret_7d', 'ret_30d', 'market_state', 'patterns',
        'sector', 'category',
    ]
    avail = [c for c in dl_cols if c in latest.columns]
    csv_out = latest[latest['match_score'] >= min_match][avail].to_csv(index=False)
    st.download_button(
        "📥 Download All Candidates CSV", csv_out,
        "wave_gainer_candidates.csv", "text/csv",
    )

# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#  TAB 2 — GAINER DNA
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
with tab2:
    st.markdown("### 🧬 What Makes a Gainer? — Pre-Gain DNA Analysis")
    st.caption(
        f"Based on **{fp.n_gainers_found}** gainers tracked across "
        f"**{len(pre_gain_weeks)}** pre-gain weeks "
        f"({pre_gain_weeks[0].strftime('%b %d')} → {pre_gain_weeks[-1].strftime('%b %d, %Y')})"
    )

    if not fp.profile:
        st.warning("Not enough data to build fingerprint. Upload more weekly CSVs or check matching.")
        st.stop()

    # ── Feature Importance ──
    st.markdown("#### 📊 Feature Importance — What Distinguishes Gainers?")
    st.caption("Cohen's d: how different gainers are from non-gainers. Higher = more discriminative.")

    feat_rows = []
    for feat, prof in fp.profile.items():
        feat_rows.append({
            'Feature': FEAT_DISPLAY.get(feat, feat),
            'Effect Size': round(abs(prof['effect_size']), 3),
            'Direction': '↑ Higher in gainers' if prof['direction'] == 'higher' else '↓ Lower in gainers',
            'Gainer Avg': round(prof['g_mean'], 1),
            'Others Avg': round(prof['o_mean'], 1),
            'Weight': f"{fp.feature_weights.get(feat, 0):.0%}",
        })

    feat_df = pd.DataFrame(feat_rows).sort_values('Effect Size', ascending=False)

    fig = px.bar(
        feat_df, x='Feature', y='Effect Size', color='Effect Size',
        color_continuous_scale='YlGn',
        title="Feature Importance (Cohen's d — higher = more predictive)",
        hover_data=['Direction', 'Gainer Avg', 'Others Avg', 'Weight'],
    )
    fig.add_hline(y=0.2, line_dash='dash', line_color='#94a3b8',
                  annotation_text='Significance threshold')
    fig.update_layout(height=400, template='plotly_dark', xaxis_tickangle=-35,
                      margin=dict(t=40, b=80))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(feat_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Distribution Comparison ──
    st.markdown("#### 📈 Gainer vs Others Distribution — Top Features")
    st.caption("The more separated the distributions, the more useful the feature.")

    top_feats = [f for f in fp.important_features[:6] if f in panel.columns]
    if top_feats:
        early = panel[panel['week'].isin(pre_gain_weeks)]
        g_early = early[early['ticker'].isin(gainer_tickers)]
        o_early = early[~early['ticker'].isin(gainer_tickers)]

        cols = st.columns(3)
        for i, feat in enumerate(top_feats):
            with cols[i % 3]:
                name = FEAT_DISPLAY.get(feat, feat)
                g_vals = g_early[feat].dropna()
                o_vals_raw = o_early[feat].dropna()
                o_vals = (o_vals_raw.sample(min(len(o_vals_raw), 500), random_state=42)
                          if len(o_vals_raw) > 500 else o_vals_raw)

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=o_vals, name='Others', marker_color='#475569', opacity=0.6, nbinsx=30,
                ))
                fig.add_trace(go.Histogram(
                    x=g_vals, name='Gainers', marker_color='#22c55e', opacity=0.8, nbinsx=30,
                ))
                fig.add_vline(
                    x=g_vals.mean(), line_color='#22c55e', line_dash='dash',
                    annotation_text=f'Gainer: {g_vals.mean():.0f}',
                )
                fig.update_layout(
                    title=name, height=260, template='plotly_dark',
                    barmode='overlay', showlegend=(i == 0),
                    margin=dict(t=35, b=20, l=30, r=10),
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Pattern Analysis ──
    st.markdown("#### 🏷️ Pattern Power — Which Patterns Signal Future Gains?")
    st.caption("Lift > 1.5 = pattern is 1.5× more common in gainers. Higher = stronger signal.")

    pat_rows = []
    for col, info in fp.pattern_lifts.items():
        if info['gainer_pct'] > 0.5 or info['other_pct'] > 0.5:
            pat_rows.append({
                'Pattern': col.replace('p_', '').replace('_', ' ').upper(),
                'Gainer %': round(info['gainer_pct'], 1),
                'Others %': round(info['other_pct'], 1),
                'Lift': round(info['lift'], 2),
                'Signal': ('🟢 STRONG' if info['lift'] > 2
                           else '🟡 MODERATE' if info['lift'] > 1.5
                           else '🔴 ANTI' if info['lift'] < 0.7
                           else '⚪ NEUTRAL'),
            })

    if pat_rows:
        pat_df = pd.DataFrame(pat_rows).sort_values('Lift', ascending=False)
        fig = px.bar(
            pat_df, x='Pattern', y='Lift', color='Lift',
            color_continuous_scale='RdYlGn', color_continuous_midpoint=1,
            title='Pattern Lift (>1 = gainer-favoring, <1 = anti-gainer)',
            hover_data=['Gainer %', 'Others %'],
        )
        fig.add_hline(y=1, line_dash='dash', line_color='#94a3b8', annotation_text='Baseline')
        fig.update_layout(height=400, template='plotly_dark', xaxis_tickangle=-40,
                          margin=dict(t=40, b=80))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pat_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Market State Analysis ──
    st.markdown("#### 🌐 Market State — Where Were Gainers Before They Gained?")
    state_rows = []
    for state, info in fp.state_probs.items():
        state_rows.append({
            'State': state,
            'Gainer %': round(info['gainer_pct'], 1),
            'Others %': round(info['other_pct'], 1),
            'Lift': round(info['lift'], 2),
        })
    if state_rows:
        state_df = pd.DataFrame(state_rows).sort_values('Lift', ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=state_df['State'], y=state_df['Gainer %'],
                             name='Gainers', marker_color='#22c55e', opacity=0.8))
        fig.add_trace(go.Bar(x=state_df['State'], y=state_df['Others %'],
                             name='Others', marker_color='#475569', opacity=0.6))
        fig.update_layout(
            title='Market State Distribution: Gainers vs Others (Pre-Gain Period)',
            height=350, template='plotly_dark', barmode='group',
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Ideal Gainer Profile ──
    st.markdown("#### 🏆 The Ideal Gainer Profile")
    st.caption("A stock matching ALL these criteria is the ideal pre-gainer:")

    for feat in fp.important_features[:8]:
        p = fp.profile[feat]
        name = FEAT_DISPLAY.get(feat, feat)
        if p['direction'] == 'higher':
            st.markdown(
                f"• **{name}** ≥ {p['g_q25']:.0f}  "
                f"(gainer avg **{p['g_mean']:.0f}**, range {p['g_q25']:.0f}–{p['g_q75']:.0f})"
            )
        else:
            st.markdown(
                f"• **{name}** ≤ {p['g_q75']:.0f}  "
                f"(gainer avg **{p['g_mean']:.0f}**, range {p['g_q25']:.0f}–{p['g_q75']:.0f})"
            )

    strong_pats = [
        col.replace('p_', '').replace('_', ' ').upper()
        for col, info in sorted(fp.pattern_lifts.items(), key=lambda x: -x[1]['lift'])
        if info['lift'] > 1.5
    ][:5]
    if strong_pats:
        st.markdown(f"• **Key Patterns**: {', '.join(strong_pats)}")

    gainer_states = [
        s for s, info in sorted(fp.state_probs.items(), key=lambda x: -x[1]['lift'])
        if info['lift'] > 1.2
    ][:3]
    if gainer_states:
        st.markdown(f"• **Favored States**: {', '.join(gainer_states)}")

    st.markdown("---")

    # ── Gainer Journey: Average Rank Over Time ──
    st.markdown("#### 📉 Average Gainer Rank Trajectory Over Time")
    st.caption("How did gainers' rank evolve week-by-week? Falling line = improving rank.")

    g_panel = panel[panel['ticker'].isin(gainer_tickers)]
    if len(g_panel) > 0:
        g_weekly = g_panel.groupby('week').agg(
            avg_rank=('rank', 'mean'),
            avg_score=('master_score', 'mean'),
            count=('ticker', 'nunique'),
        ).reset_index().sort_values('week')

        o_panel = panel[~panel['ticker'].isin(gainer_tickers)]
        o_weekly = o_panel.groupby('week').agg(
            avg_rank=('rank', 'mean'),
        ).reset_index().sort_values('week')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=g_weekly['week'], y=g_weekly['avg_rank'],
            name='Gainers Avg Rank', mode='lines+markers',
            line=dict(color='#22c55e', width=3),
        ))
        fig.add_trace(go.Scatter(
            x=o_weekly['week'], y=o_weekly['avg_rank'],
            name='Others Avg Rank', mode='lines',
            line=dict(color='#475569', width=2, dash='dot'),
        ))
        fig.update_yaxes(autorange='reversed')
        fig.update_layout(
            title='Average Rank Over Time (↓ = better)',
            height=350, template='plotly_dark', margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Historical Validation ──
    st.markdown("---")
    st.markdown("#### 🔍 Historical Validation")
    st.caption("If you'd used this system at each past week, what % of actual gainers would you have caught?")

    if st.button("🚀 Run Validation", type="primary"):
        with st.spinner("Running validation across all pre-gain weeks…"):
            val = fp.validate_hit_rate(panel, pre_gain_weeks, top_ns=(20, 50, 100))

        if len(val) > 0:
            v_cols = st.columns(3)
            for i, n in enumerate((20, 50, 100)):
                sub = val[val['top_n'] == n]
                if len(sub) > 0:
                    with v_cols[i]:
                        st.metric(
                            f"Top {n} Avg Hit Rate",
                            f"{sub['hit_rate'].mean():.1f}%",
                            help=f"Avg {sub['gainers_caught'].mean():.0f} gainers caught per week",
                        )

            v50 = val[val['top_n'] == 50]
            if len(v50) > 0:
                fig = px.bar(
                    v50, x='week', y='hit_rate', color='hit_rate',
                    color_continuous_scale='YlGn',
                    title='Gainer Hit Rate per Week (Top 50 picks)',
                )
                fig.update_layout(height=300, template='plotly_dark', margin=dict(t=40, b=30))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for validation.")


# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#  TAB 3 — STOCK X-RAY
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
with tab3:
    st.markdown("### 🔬 Stock X-Ray — Full Journey + Gainer Match")

    all_tickers = sorted(panel['ticker'].unique())

    col_sel, col_search = st.columns([3, 1])
    with col_sel:
        default_ticker = all_tickers[0] if all_tickers else ''
        ticker_sel = st.selectbox("Select Stock", all_tickers,
                                  index=0 if all_tickers else 0)
    with col_search:
        search_q = st.text_input("Quick search", "")
        if search_q:
            matches = [t for t in all_tickers if search_q.upper() in t]
            if matches:
                ticker_sel = st.selectbox("Matches", matches, key='xr_search')

    stock = panel[panel['ticker'] == ticker_sel].sort_values('week')

    if len(stock) == 0:
        st.warning(f"No data for {ticker_sel}")
    else:
        last_row = stock.iloc[-1]
        first_row = stock.iloc[0]

        # Compute match score for this stock
        match_val = latest.loc[latest['ticker'] == ticker_sel, 'match_score']
        match_val = float(match_val.iloc[0]) if len(match_val) > 0 else fp.score_batch(
            stock.iloc[[-1]]
        )[0]

        is_g = ticker_sel in gainer_tickers
        is_l = ticker_sel in loser_tickers
        sc_class = 'green' if match_val >= 70 else 'amber' if match_val >= 50 else 'gray'

        badge_html = ''
        if is_g:
            badge_html = f'<span class="badge badge-confirmed">CONFIRMED GAINER +{gainer_matched.get(ticker_sel, 0):.0f}%</span>'
        elif is_l:
            badge_html = '<span class="badge badge-loser">KNOWN LOSER</span>'
        elif match_val >= 70:
            badge_html = '<span class="badge badge-new">🆕 GAINER CANDIDATE</span>'

        st.markdown(f"""
        <div class="scard">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    {badge_html}
                    <h2 style="margin:4px 0;color:#f1f5f9">{ticker_sel}</h2>
                    <div class="muted">{str(last_row.get('company_name',''))[:50]} · {last_row.get('sector','')} · {last_row.get('category','')}</div>
                </div>
                <div style="text-align:right">
                    <div class="big {sc_class}">{match_val:.0f}%</div>
                    <div class="muted">Gainer Match</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        total_chg = (
            (last_row['price'] - first_row['price']) / first_row['price'] * 100
            if first_row['price'] > 0 else 0
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Price", f"₹{last_row['price']:,.0f}", f"{total_chg:+.1f}%")
        with c2:
            rd = last_row.get('rank_delta_1w', 0)
            st.metric("Rank", f"#{last_row['rank']:.0f}",
                      f"{rd:+.0f}" if pd.notna(rd) else None)
        with c3:
            sd = last_row.get('score_delta_1w', 0)
            st.metric("Score", f"{last_row['master_score']:.0f}",
                      f"{sd:+.1f}" if pd.notna(sd) else None)
        with c4:
            st.metric("Breakout", f"{last_row.get('breakout_score', 0):.0f}")
        with c5:
            st.metric("State", last_row.get('market_state', ''))
        with c6:
            st.metric("Harmony", f"{last_row.get('momentum_harmony', 0):.0f}")

        # Reasons
        reasons = fp.explain(last_row)
        if reasons:
            st.markdown("#### Why this match score?")
            for cls, reason in reasons:
                color = '#4ade80' if cls == 'ok' else '#f87171'
                st.markdown(f'<span style="color:{color}">{reason}</span>',
                            unsafe_allow_html=True)

        st.markdown("---")

        # ── Journey Charts ──
        st.markdown("#### 📈 Weekly Journey")
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '💰 Price', '📊 Rank (↓=better)', '🎯 Master Score',
                '⚡ Breakout Score', '📈 Momentum Score', '🔊 Volume Score',
            ],
            vertical_spacing=0.08,
        )
        w = stock['week']
        chart_conf = [
            ('price', '#38bdf8', 1, 1),
            ('rank', '#f87171', 1, 2),
            ('master_score', '#4ade80', 2, 1),
            ('breakout_score', '#fbbf24', 2, 2),
            ('momentum_score', '#a855f7', 3, 1),
            ('volume_score', '#06b6d4', 3, 2),
        ]
        for col_name, color, r, c in chart_conf:
            if col_name in stock.columns:
                fig.add_trace(
                    go.Scatter(x=w, y=stock[col_name], mode='lines+markers',
                               line=dict(color=color, width=2), showlegend=False),
                    row=r, col=c,
                )
        fig.update_yaxes(autorange='reversed', row=1, col=2)
        fig.update_layout(height=700, template='plotly_dark', margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # ── Match Score Over Time ──
        st.markdown("#### 🎯 Gainer Match Score Trajectory")
        st.caption("Is this stock BECOMING more gainer-like? Rising = stronger match.")

        match_hist = []
        for _, r in stock.iterrows():
            ms_val = fp.score_batch(pd.DataFrame([r]))[0]
            match_hist.append({'week': r['week'], 'match': ms_val})
        mh_df = pd.DataFrame(match_hist)

        if len(mh_df) > 0:
            bar_colors = [
                '#22c55e' if s >= 70 else '#f59e0b' if s >= 50 else '#64748b'
                for s in mh_df['match']
            ]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mh_df['week'], y=mh_df['match'],
                                 marker_color=bar_colors))
            fig.add_hline(y=70, line_dash='dash', line_color='#22c55e',
                          annotation_text='High')
            fig.add_hline(y=50, line_dash='dash', line_color='#f59e0b',
                          annotation_text='Medium')
            fig.update_layout(
                height=300, template='plotly_dark',
                yaxis=dict(range=[0, 100]),
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Pattern & State Timeline ──
        st.markdown("#### 🏷️ Pattern & State Timeline")
        state_icons = {
            'STRONG_UPTREND': '🟢', 'UPTREND': '🟢', 'PULLBACK': '🟡',
            'ROTATION': '🟡', 'SIDEWAYS': '⚪', 'BOUNCE': '🔵',
            'DOWNTREND': '🔴', 'STRONG_DOWNTREND': '🔴',
        }
        for _, r in stock.iterrows():
            pat = str(r.get('patterns', ''))
            st_val = r.get('market_state', '')
            icon = state_icons.get(st_val, '⚪')
            line = (
                f"**{r['week'].strftime('%b %d')}** | {icon} {st_val} | "
                f"#{r['rank']:.0f} | Score {r['master_score']:.0f} | ₹{r['price']:.0f}"
            )
            if pat and pat != 'nan' and len(pat) > 3:
                line += f" | {pat[:100]}"
            st.markdown(line)

        # ── Full Data ──
        with st.expander("📋 Full Weekly Data"):
            show_cols = [
                'week', 'rank', 'rank_delta_1w', 'rank_velocity', 'master_score',
                'score_delta_1w', 'price', 'breakout_score', 'momentum_score',
                'volume_score', 'ret_7d', 'ret_30d', 'ret_3m', 'market_state',
                'patterns', 'weeks_in_top100',
            ]
            avail = [c for c in show_cols if c in stock.columns]
            disp = stock[avail].copy()
            disp['week'] = disp['week'].dt.strftime('%Y-%m-%d')
            st.dataframe(disp.reset_index(drop=True), use_container_width=True,
                         hide_index=True)

        # ── Peer Comparison ──
        st.markdown("#### 👥 Industry Peers in Latest Week")
        industry = last_row.get('industry', '')
        if industry:
            peers = panel[
                (panel['week'] == latest_week) & (panel['industry'] == industry)
            ].nsmallest(10, 'rank')
            if len(peers) > 0:
                peer_cols = [
                    'ticker', 'company_name', 'rank', 'master_score',
                    'price', 'ret_7d', 'ret_30d', 'market_state',
                ]
                pavail = [c for c in peer_cols if c in peers.columns]
                st.dataframe(peers[pavail].reset_index(drop=True),
                             use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "WAVE Gainer Scanner — Finds future winners by studying past gainer DNA. "
    "Companion to WAVE Detection 3.0. Not financial advice."
)
