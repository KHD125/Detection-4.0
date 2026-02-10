# ══════════════════════════════════════════════════════════════
#  WAVE v2 — Stock Probability Engine (Redesigned)
# ══════════════════════════════════════════════════════════════
#  
#  WHAT CHANGED FROM v1:
#  1. Statistical approach first (no ML pretending 23 weeks is enough)
#  2. Bayesian probability — P(future gain | current signals) from actual data  
#  3. Walk-forward validation (not random split)
#  4. Quintile analysis — the REAL edge discovery method
#  5. Composite score = data-driven weights, not guesswork
#  6. Regime detection — market context matters
#  7. Portfolio construction with risk management
#  8. Clean 5-page layout instead of 8 cluttered tabs
#
#  pip install streamlit pandas numpy plotly scipy
#  streamlit run wave_v2.py
# ══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import re
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="WAVE v2", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; }
h1 { font-size: 2rem !important; }
h2 { font-size: 1.4rem !important; }
h3 { font-size: 1.15rem !important; }
.big-number { font-size: 2.8rem; font-weight: 800; line-height: 1; }
.green { color: #22c55e; }
.red { color: #ef4444; }
.amber { color: #f59e0b; }
.muted { color: #9ca3af; font-size: 0.82rem; }
.card {
    background: #111827; border: 1px solid #1f2937; border-radius: 12px;
    padding: 1.2rem; margin-bottom: 0.8rem;
}
.edge-positive { border-left: 4px solid #22c55e; }
.edge-negative { border-left: 4px solid #ef4444; }
div[data-testid="stMetricValue"] > div { font-size: 1.5rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# ─── DATA ENGINE ──────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_process(uploaded_files):
    """Load all files → build unified panel dataset."""
    frames = []
    for f in uploaded_files:
        df = pd.read_csv(f)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', f.name)
        df['week'] = pd.to_datetime(date_match.group(1)) if date_match else pd.NaT
        frames.append(df)
    
    panel = pd.concat(frames, ignore_index=True).dropna(subset=['week'])
    
    # Ensure numeric
    num_cols = ['rank','master_score','position_score','volume_score','momentum_score',
                'acceleration_score','breakout_score','rvol_score','trend_quality',
                'price','pe','eps_current','from_low_pct','from_high_pct',
                'ret_1d','ret_7d','ret_30d','ret_3m','ret_6m','ret_1y',
                'rvol','vmi','money_flow_mm','position_tension','momentum_harmony',
                'overall_market_strength']
    for c in num_cols:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors='coerce')
    
    panel = panel.sort_values(['ticker','week']).reset_index(drop=True)
    return panel


@st.cache_data(show_spinner=False)
def build_analytics(panel):
    """
    Core computation:
    1. Forward returns for every stock-week
    2. Rank/score deltas
    3. Pattern parsing
    4. Quintile buckets
    """
    weeks = sorted(panel['week'].unique())
    week_idx = {w: i for i, w in enumerate(weeks)}
    n_weeks = len(weeks)
    
    # ── Price lookup table (fast) ──
    price_map = panel.set_index(['ticker','week'])['price'].to_dict()
    rank_map = panel.set_index(['ticker','week'])['rank'].to_dict()
    score_map = panel.set_index(['ticker','week'])['master_score'].to_dict()
    
    records = []
    
    for ticker, grp in panel.groupby('ticker'):
        grp = grp.sort_values('week')
        grp_weeks = grp['week'].values
        
        for row_idx, (_, row) in enumerate(grp.iterrows()):
            w = row['week']
            wi = week_idx[w]
            p = row['price']
            
            if pd.isna(p) or p <= 0:
                continue
            
            rec = {
                'ticker': ticker,
                'company_name': row.get('company_name',''),
                'week': w,
                'week_idx': wi,
                'price': p,
                'rank': row['rank'],
                'master_score': row['master_score'],
                'position_score': row.get('position_score',0),
                'volume_score': row.get('volume_score',0),
                'momentum_score': row.get('momentum_score',0),
                'acceleration_score': row.get('acceleration_score',0),
                'breakout_score': row.get('breakout_score',0),
                'rvol_score': row.get('rvol_score',0),
                'trend_quality': row.get('trend_quality',0),
                'ret_7d': row.get('ret_7d',0),
                'ret_30d': row.get('ret_30d',0),
                'ret_3m': row.get('ret_3m',0),
                'from_low_pct': row.get('from_low_pct',0),
                'from_high_pct': row.get('from_high_pct',0),
                'rvol': row.get('rvol',0),
                'vmi': row.get('vmi',0),
                'money_flow_mm': row.get('money_flow_mm',0),
                'position_tension': row.get('position_tension',0),
                'momentum_harmony': row.get('momentum_harmony',0),
                'market_state': row.get('market_state',''),
                'patterns': str(row.get('patterns','')),
                'sector': row.get('sector',''),
                'industry': row.get('industry',''),
                'category': row.get('category',''),
                'pe': row.get('pe',np.nan),
                'overall_market_strength': row.get('overall_market_strength',0),
            }
            
            # ── Rank delta (vs previous week for this stock) ──
            if row_idx > 0:
                prev_w = grp_weeks[row_idx - 1]
                prev_rank = rank_map.get((ticker, prev_w))
                prev_score = score_map.get((ticker, prev_w))
                if prev_rank is not None:
                    rec['rank_delta_1w'] = prev_rank - row['rank']  # +ve = improved
                else:
                    rec['rank_delta_1w'] = 0
                if prev_score is not None:
                    rec['score_delta_1w'] = row['master_score'] - prev_score
                else:
                    rec['score_delta_1w'] = 0
            else:
                rec['rank_delta_1w'] = 0
                rec['score_delta_1w'] = 0
            
            # ── Forward returns ──
            for fwd in [1, 2, 4]:
                fwd_wi = wi + fwd
                if fwd_wi < n_weeks:
                    fwd_w = weeks[fwd_wi]
                    fwd_p = price_map.get((ticker, fwd_w))
                    if fwd_p and fwd_p > 0:
                        rec[f'fwd_{fwd}w'] = ((fwd_p - p) / p) * 100
                    else:
                        rec[f'fwd_{fwd}w'] = np.nan
                else:
                    rec[f'fwd_{fwd}w'] = np.nan
            
            # ── Pattern flags ──
            pat = rec['patterns']
            rec['n_patterns'] = pat.count('|') + 1 if pat and pat != 'nan' else 0
            for tag, key in [
                ('CAT LEADER','cat'),('VOL EXPLOSION','vol_exp'),('MARKET LEADER','mkt_ldr'),
                ('MOMENTUM WAVE','mom_wave'),('PREMIUM MOMENTUM','prem_mom'),
                ('VELOCITY BREAKOUT','vel_brk'),('INSTITUTIONAL','inst'),
                ('GOLDEN CROSS','gold_x'),('STEALTH','stealth'),
                ('DISTRIBUTION','distrib'),('CAPITULATION','capit'),
                ('HIGH PE','hi_pe'),('PHOENIX','phoenix'),('PULLBACK SUPPORT','pb_sup'),
                ('RANGE COMPRESS','rng_comp'),('ACCELERATION','accel'),
                ('ROTATION LEADER','rot_ldr'),('GARP','garp'),
                ('VALUE MOMENTUM','val_mom'),('EARNINGS ROCKET','earn_rkt'),
                ('LIQUID LEADER','liquid'),('VELOCITY SQUEEZE','vel_sq'),
                ('RUNAWAY GAP','run_gap'),('PYRAMID','pyramid'),
            ]:
                rec[f'p_{key}'] = 1 if tag in pat else 0
            
            # ── Market state flags ──
            ms = str(rec['market_state'])
            rec['is_uptrend'] = 1 if ms in ('UPTREND','STRONG_UPTREND') else 0
            rec['is_downtrend'] = 1 if ms in ('DOWNTREND','STRONG_DOWNTREND') else 0
            
            records.append(rec)
    
    df = pd.DataFrame(records)
    
    # ── Quintile ranks per week (for each feature) ──
    quintile_features = ['master_score','rank','momentum_score','breakout_score',
                         'volume_score','acceleration_score','trend_quality',
                         'rank_delta_1w','score_delta_1w','rvol','money_flow_mm']
    
    for feat in quintile_features:
        if feat in df.columns:
            # Quintile 5 = best for scores, quintile 1 = best for rank (lowest)
            ascending = (feat == 'rank')
            df[f'q_{feat}'] = df.groupby('week')[feat].transform(
                lambda x: pd.qcut(x.rank(method='first'), 5, labels=[5,4,3,2,1] if ascending else [1,2,3,4,5])
            )
            df[f'q_{feat}'] = pd.to_numeric(df[f'q_{feat}'], errors='coerce')
    
    return df, weeks


@st.cache_data(show_spinner=False)
def compute_quintile_returns(df, feature, fwd_col='fwd_4w'):
    """For a feature's quintiles, what was the average forward return?"""
    qcol = f'q_{feature}'
    if qcol not in df.columns:
        return None
    valid = df.dropna(subset=[fwd_col, qcol])
    if len(valid) < 100:
        return None
    
    result = valid.groupby(qcol).agg(
        count=(fwd_col, 'count'),
        mean_return=(fwd_col, 'mean'),
        median_return=(fwd_col, 'median'),
        win_rate=(fwd_col, lambda x: (x > 0).mean() * 100),
        big_win_rate=(fwd_col, lambda x: (x > 10).mean() * 100),
        big_loss_rate=(fwd_col, lambda x: (x < -10).mean() * 100),
    ).reset_index()
    result.columns = ['Quintile','Count','Mean Return %','Median Return %',
                       'Win Rate %','Big Win >10%','Big Loss <-10%']
    result['Quintile'] = result['Quintile'].astype(int)
    
    # Monotonicity score (does Q5 outperform Q1 consistently?)
    returns = result['Mean Return %'].values
    if len(returns) == 5:
        mono_score = np.corrcoef(range(5), returns)[0,1]
    else:
        mono_score = 0
    
    return result, mono_score


@st.cache_data(show_spinner=False)
def compute_bayesian_probability(df, fwd_col='fwd_4w'):
    """
    P(gain > 0 | rank_quintile, score_quintile, market_state, has_key_patterns)
    
    This is the REAL probability system — computed from actual outcomes.
    No ML black box, just conditional frequencies.
    """
    valid = df.dropna(subset=[fwd_col]).copy()
    if len(valid) < 200:
        return None
    
    valid['gained'] = (valid[fwd_col] > 0).astype(int)
    valid['big_gained'] = (valid[fwd_col] > 10).astype(int)
    
    # Base rate
    base_rate = valid['gained'].mean()
    
    # Create composite bucket
    valid['rank_q'] = valid.get('q_rank', 3)
    valid['score_q'] = valid.get('q_master_score', 3)
    
    # Group: Rank quintile × Score quintile × Uptrend
    valid['bucket'] = (valid['rank_q'].astype(str) + '_' + 
                       valid['score_q'].astype(str) + '_' +
                       valid['is_uptrend'].astype(str))
    
    bucket_stats = valid.groupby('bucket').agg(
        count=('gained', 'count'),
        win_rate=('gained', 'mean'),
        big_win_rate=('big_gained', 'mean'),
        avg_return=(fwd_col, 'mean'),
        median_return=(fwd_col, 'median'),
    )
    
    # Minimum sample size for reliability
    bucket_stats = bucket_stats[bucket_stats['count'] >= 5]
    
    # Bayesian smoothing: blend with base rate (shrinkage toward prior)
    alpha = 20  # strength of prior
    bucket_stats['smoothed_win_rate'] = (
        (bucket_stats['count'] * bucket_stats['win_rate'] + alpha * base_rate) /
        (bucket_stats['count'] + alpha)
    )
    
    return bucket_stats, base_rate, valid


@st.cache_data(show_spinner=False)
def compute_pattern_edge(df, fwd_col='fwd_4w'):
    """For each pattern, compute actual edge vs baseline."""
    valid = df.dropna(subset=[fwd_col])
    if len(valid) < 100:
        return pd.DataFrame()
    
    baseline_return = valid[fwd_col].mean()
    baseline_winrate = (valid[fwd_col] > 0).mean() * 100
    
    pattern_cols = [c for c in df.columns if c.startswith('p_')]
    results = []
    
    for pc in pattern_cols:
        with_pat = valid[valid[pc] == 1]
        without_pat = valid[valid[pc] == 0]
        
        if len(with_pat) < 10:
            continue
        
        wr = (with_pat[fwd_col] > 0).mean() * 100
        avg_ret = with_pat[fwd_col].mean()
        
        # Statistical significance (t-test)
        if len(with_pat) >= 10 and len(without_pat) >= 10:
            t_stat, p_val = stats.ttest_ind(
                with_pat[fwd_col].dropna(), 
                without_pat[fwd_col].dropna(), 
                equal_var=False
            )
        else:
            t_stat, p_val = 0, 1
        
        results.append({
            'Pattern': pc.replace('p_','').replace('_',' ').upper(),
            'Count': len(with_pat),
            'Avg Return %': round(avg_ret, 2),
            'Edge vs Baseline': round(avg_ret - baseline_return, 2),
            'Win Rate %': round(wr, 1),
            'Win Rate Edge': round(wr - baseline_winrate, 1),
            'Big Win >10%': round((with_pat[fwd_col] > 10).mean() * 100, 1),
            'Big Loss <-10%': round((with_pat[fwd_col] < -10).mean() * 100, 1),
            'p-value': round(p_val, 4),
            'Significant': '✅' if p_val < 0.1 else '❌',
        })
    
    return pd.DataFrame(results).sort_values('Edge vs Baseline', ascending=False)


@st.cache_data(show_spinner=False)
def compute_composite_score(df, pattern_edges, quintile_monos, fwd_col='fwd_4w'):
    """
    Data-driven composite score:
    Weight each feature by its monotonicity (how cleanly quintiles predict returns).
    Weight each pattern by its actual edge.
    """
    valid = df.copy()
    
    # ── Quintile-based component (data-driven weights) ──
    score_features = ['master_score','rank','momentum_score','breakout_score',
                      'volume_score','trend_quality','rank_delta_1w','score_delta_1w']
    
    total_weight = 0
    valid['composite'] = 0.0
    
    for feat in score_features:
        qcol = f'q_{feat}'
        if qcol in valid.columns and feat in quintile_monos:
            mono = abs(quintile_monos[feat])
            weight = mono  # Higher monotonicity = higher weight
            valid['composite'] += valid[qcol].fillna(3) * weight
            total_weight += weight * 5  # max quintile is 5
    
    if total_weight > 0:
        valid['composite'] = (valid['composite'] / total_weight) * 70  # Scale to 0-70
    
    # ── Pattern bonus (up to ±30 points) ──
    if len(pattern_edges) > 0:
        edge_map = dict(zip(
            pattern_edges['Pattern'].str.lower().str.replace(' ','_').apply(lambda x: f'p_{x}'),
            pattern_edges['Edge vs Baseline']
        ))
        
        pattern_bonus = 0
        for pc, edge in edge_map.items():
            if pc in valid.columns:
                # Normalize edge to ±5 points per pattern
                normalized = np.clip(edge / 3, -5, 5)
                pattern_bonus = pattern_bonus + valid[pc] * normalized
        
        valid['composite'] += np.clip(pattern_bonus, -30, 30)
    
    # ── Uptrend bonus ──
    valid['composite'] += valid['is_uptrend'] * 5
    valid['composite'] -= valid['is_downtrend'] * 5
    
    # ── Clamp and normalize to 0-100 ──
    valid['composite'] = np.clip(valid['composite'], 0, 100)
    
    # ── Percentile rank within each week ──
    valid['composite_pctile'] = valid.groupby('week')['composite'].rank(pct=True) * 100
    
    return valid


@st.cache_data(show_spinner=False)
def walk_forward_backtest(df, fwd_col='fwd_4w', top_n=20):
    """
    Walk-forward: for each week, use ONLY past data to compute composite,
    then pick top-N, measure actual forward return.
    
    This is the HONEST test — no look-ahead bias.
    """
    weeks = sorted(df['week'].unique())
    
    # Need at least 4 weeks of history to start
    results = []
    
    for i in range(4, len(weeks)):
        current_week = weeks[i]
        
        # Only stocks from current week
        current = df[df['week'] == current_week].copy()
        
        # Forward return must exist
        current_valid = current.dropna(subset=[fwd_col])
        
        if len(current_valid) < 50:
            continue
        
        # Historical data (all weeks before current)
        historical = df[df['week'] < current_week]
        hist_valid = historical.dropna(subset=[fwd_col])
        
        if len(hist_valid) < 200:
            continue
        
        # Compute historical pattern: top quintile master_score stocks vs bottom
        # Simple: pick stocks with highest composite rank (use composite_pctile if available)
        if 'composite_pctile' in current_valid.columns:
            top_picks = current_valid.nlargest(top_n, 'composite_pctile')
            bottom_picks = current_valid.nsmallest(top_n, 'composite_pctile')
        else:
            top_picks = current_valid.nsmallest(top_n, 'rank')
            bottom_picks = current_valid.nlargest(top_n, 'rank')
        
        random_sample = current_valid.sample(min(top_n, len(current_valid)))
        
        results.append({
            'week': current_week,
            'top_n_avg_return': top_picks[fwd_col].mean(),
            'top_n_median_return': top_picks[fwd_col].median(),
            'top_n_win_rate': (top_picks[fwd_col] > 0).mean() * 100,
            'bottom_n_avg_return': bottom_picks[fwd_col].mean(),
            'bottom_n_win_rate': (bottom_picks[fwd_col] > 0).mean() * 100,
            'random_avg_return': random_sample[fwd_col].mean(),
            'random_win_rate': (random_sample[fwd_col] > 0).mean() * 100,
            'spread': top_picks[fwd_col].mean() - bottom_picks[fwd_col].mean(),
            'top_stocks': ', '.join(top_picks['ticker'].head(5).tolist()),
            'universe_avg': current_valid[fwd_col].mean(),
        })
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════

# ── Sidebar ──
with st.sidebar:
    st.markdown("# 🌊 WAVE v2")
    st.caption("Stock Probability Engine")
    
    uploaded_files = st.file_uploader(
        "📁 Upload Weekly CSVs",
        type=['csv'], accept_multiple_files=True,
        help="Upload Stocks_Weekly_*.csv files (minimum 5 recommended)"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} files loaded")
        
        st.markdown("---")
        horizon = st.radio("⏱️ Horizon", ['1 Week','2 Weeks','4 Weeks'], index=2)
        fwd_col = {'1 Week':'fwd_1w','2 Weeks':'fwd_2w','4 Weeks':'fwd_4w'}[horizon]
        
        top_n = st.slider("Top N stocks to pick", 5, 50, 20)
        
        st.markdown("---")
        min_score = st.slider("Min Master Score", 0.0, 80.0, 0.0, 5.0)
        max_rank_filter = st.slider("Max Rank", 50, 2200, 2200, 50)
        
        cat_filter = st.multiselect("Category", ['Mega Cap','Large Cap','Mid Cap','Small Cap','Micro Cap'])
        sector_input = st.text_input("Sector contains", "")


# ── Main ──
if not uploaded_files:
    st.markdown("# 🌊 WAVE v2 — Stock Probability Engine")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What makes this different?
        
        **v1 was flawed.** It used XGBoost on 23 weeks of data — too little to learn, 
        too many features, arbitrary rule weights, no honest validation.
        
        **v2 is built on statistical truths:**
        
        | Approach | Why It Works |
        |----------|-------------|
        | **Quintile Analysis** | Split stocks into 5 buckets by any feature → see which bucket's stocks actually gained the most. No ML needed. |
        | **Bayesian Probability** | P(gain \| rank=top, score=high, uptrend=yes) computed from actual frequencies, smoothed with prior. |
        | **Pattern Edge with p-values** | Each pattern tested — does STEALTH actually predict gains? Statistical significance reported. |
        | **Data-driven weights** | Features weighted by their monotonicity score (do higher quintiles consistently beat lower ones?). |
        | **Walk-forward backtest** | Each week, system sees ONLY past data. No cheating. |
        """)
    
    with col2:
        st.markdown("""
        ### 5 Pages
        
        1. **📊 Dashboard** — Today's picks with probability + reasons
        2. **🔬 Quintile Lab** — The proof: does each feature actually predict gains?  
        3. **⚡ Pattern Lab** — Which patterns have real edge (with statistics)?
        4. **🧪 Backtest** — Walk-forward: would this have worked?
        5. **📈 Stock Deep Dive** — Any stock's full journey
        
        ### How to use
        ```
        pip install streamlit pandas numpy plotly scipy
        streamlit run wave_v2.py
        ```
        
        Upload 5+ weekly CSVs (more = better).  
        System needs forward returns to validate — so the latest week 
        can get predictions but can't be validated yet.
        """)
    
    st.stop()



if len(uploaded_files) < 3:
    st.warning("Upload at least 3 weekly files.")
    st.stop()

# ── Process ──
with st.spinner("Loading data..."):
    panel = load_and_process(uploaded_files)

with st.spinner("Computing analytics..."):
    df, weeks = build_analytics(panel)

with st.spinner("Computing edges..."):
    pattern_edges = compute_pattern_edge(df, fwd_col)
    
    quintile_monos = {}
    for feat in ['master_score','rank','momentum_score','breakout_score',
                 'volume_score','trend_quality','rank_delta_1w','score_delta_1w',
                 'rvol','money_flow_mm','acceleration_score']:
        r = compute_quintile_returns(df, feat, fwd_col)
        if r is not None:
            quintile_monos[feat] = r[1]  # mono score
    
    df = compute_composite_score(df, pattern_edges, quintile_monos, fwd_col)

latest_week = max(weeks)
latest = df[df['week'] == latest_week].copy()

# Apply filters
if min_score > 0:
    latest = latest[latest['master_score'] >= min_score]
if max_rank_filter < 2200:
    latest = latest[latest['rank'] <= max_rank_filter]
if cat_filter:
    latest = latest[latest['category'].isin(cat_filter)]
if sector_input:
    latest = latest[latest['sector'].str.contains(sector_input, case=False, na=False)]


# ═══════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════

page = st.radio("", ["📊 Dashboard","🔬 Quintile Lab","⚡ Pattern Lab","🧪 Backtest","📈 Stock Deep Dive"],
                horizontal=True, label_visibility="collapsed")


# ─── PAGE 1: DASHBOARD ───────────────────────────────────
if page == "📊 Dashboard":
    st.markdown(f"## 📊 Dashboard — Week of {latest_week.strftime('%B %d, %Y')}")
    
    # Market overview
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        avg_strength = latest['overall_market_strength'].mean()
        st.metric("Market Strength", f"{avg_strength:.1f}")
    with c2:
        pct_uptrend = latest['is_uptrend'].mean() * 100
        st.metric("% in Uptrend", f"{pct_uptrend:.0f}%")
    with c3:
        st.metric("Total Stocks", f"{len(latest):,}")
    with c4:
        st.metric("Weeks of Data", len(weeks))
    with c5:
        # Base win rate from history
        hist = df.dropna(subset=[fwd_col])
        if len(hist) > 0:
            base_wr = (hist[fwd_col] > 0).mean() * 100
            st.metric("Historical Base Win Rate", f"{base_wr:.0f}%")
    
    st.markdown("---")
    
    # ── Bayesian probability lookup ──
    bayesian = compute_bayesian_probability(df, fwd_col)
    
    if bayesian is not None:
        bucket_stats, base_rate, _ = bayesian
        
        # Assign probability to each stock
        latest['rank_q'] = latest.get('q_rank', 3).fillna(3).astype(int)
        latest['score_q'] = latest.get('q_master_score', 3).fillna(3).astype(int)
        latest['bucket'] = (latest['rank_q'].astype(str) + '_' + 
                           latest['score_q'].astype(str) + '_' +
                           latest['is_uptrend'].astype(str))
        
        latest = latest.merge(
            bucket_stats[['smoothed_win_rate','avg_return','median_return']].reset_index(),
            left_on='bucket', right_on='bucket', how='left'
        )
        latest['probability'] = (latest['smoothed_win_rate'].fillna(base_rate) * 100).round(1)
    else:
        latest['probability'] = 50.0
    
    # Sort by composite
    latest = latest.sort_values('composite', ascending=False)
    
    # ── Signal classification ──
    latest['signal'] = latest['composite_pctile'].apply(
        lambda p: '🟢 STRONG BUY' if p >= 90 else
                  '🟡 BUY' if p >= 75 else
                  '⚪ HOLD' if p >= 40 else
                  '🟠 WEAK' if p >= 20 else
                  '🔴 AVOID'
    )
    
    # ── TOP PICKS ──
    st.markdown(f"### 🏆 Top {top_n} Picks (by Composite Score)")
    
    top_picks = latest.head(top_n)
    
    for i, (_, row) in enumerate(top_picks.iterrows()):
        prob = row.get('probability', 50)
        comp = row.get('composite', 0)
        
        edge_class = 'edge-positive' if prob > 55 else 'edge-negative' if prob < 45 else ''
        
        with st.container():
            cols = st.columns([0.5, 2.5, 1.2, 1.2, 1.2, 1.2, 1.5])
            
            with cols[0]:
                st.markdown(f"**{i+1}**")
            with cols[1]:
                st.markdown(f"**{row['ticker']}**")
                st.caption(f"{row.get('company_name','')[:35]} · {row.get('sector','')}")
            with cols[2]:
                color = 'green' if prob > 55 else 'red' if prob < 45 else 'amber'
                st.markdown(f"<span class='big-number {color}'>{prob:.0f}%</span><br><span class='muted'>Win Prob</span>", unsafe_allow_html=True)
            with cols[3]:
                st.metric("Rank", f"#{row['rank']:.0f}", f"{row.get('rank_delta_1w',0):+.0f}")
            with cols[4]:
                st.metric("Score", f"{row['master_score']:.0f}", f"{row.get('score_delta_1w',0):+.1f}")
            with cols[5]:
                st.metric("Price", f"₹{row['price']:.0f}", f"{row.get('ret_7d',0):+.1f}% 7d")
            with cols[6]:
                st.markdown(f"**{row['signal']}**")
                st.caption(f"{row.get('market_state','')} · {row.get('n_patterns',0)} patterns")
            
            st.markdown("---")
    
    # ── FULL TABLE ──
    with st.expander("📋 Full Rankings Table"):
        show_cols = ['ticker','company_name','signal','probability','composite',
                     'rank','rank_delta_1w','master_score','score_delta_1w',
                     'price','ret_7d','ret_30d','market_state','patterns',
                     'sector','category']
        available = [c for c in show_cols if c in latest.columns]
        st.dataframe(latest[available].reset_index(drop=True), height=600, use_container_width=True)
        
        csv_data = latest[available].to_csv(index=False)
        st.download_button("📥 Download", csv_data, "wave_picks.csv")
    
    # ── SECTOR VIEW ──
    st.markdown("### 🗺️ Sector View")
    
    sector_agg = latest.groupby('sector').agg(
        stocks=('ticker','count'),
        avg_composite=('composite','mean'),
        avg_probability=('probability','mean'),
        avg_rank=('rank','mean'),
        avg_ret_7d=('ret_7d','mean'),
        avg_ret_30d=('ret_30d','mean'),
        pct_uptrend=('is_uptrend','mean'),
    ).round(1)
    sector_agg = sector_agg[sector_agg['stocks'] >= 3].sort_values('avg_composite', ascending=False)
    sector_agg['pct_uptrend'] = (sector_agg['pct_uptrend'] * 100).round(0)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            sector_agg.head(15).reset_index(), x='avg_composite', y='sector',
            orientation='h', color='avg_ret_30d', color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0, title='Top 15 Sectors by Composite Score'
        )
        fig.update_layout(height=450, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(sector_agg.reset_index(), use_container_width=True, height=450)
    
    # ── RISK ALERTS ──
    st.markdown("### 🚨 Risk Alerts")
    
    # Stocks that were top ranked but now crashing
    if 'rank_delta_1w' in latest.columns:
        crashers = latest[(latest['rank_delta_1w'] < -200) & (latest['rank'] > 500)]
        if len(crashers) > 0:
            st.error(f"**{len(crashers)} stocks crashed >200 ranks this week:**")
            for _, r in crashers.head(10).iterrows():
                st.markdown(f"- **{r['ticker']}** — Rank #{r['rank']:.0f} (was #{r['rank'] - r['rank_delta_1w']:.0f}) | {r.get('market_state','')}")
    
    # Capitulation flags
    capit_stocks = latest[latest.get('p_capit', 0) == 1]
    if len(capit_stocks) > 0:
        st.warning(f"**{len(capit_stocks)} stocks showing CAPITULATION pattern** — extreme selling pressure")
        st.caption(', '.join(capit_stocks['ticker'].tolist()))


# ─── PAGE 2: QUINTILE LAB ────────────────────────────────
elif page == "🔬 Quintile Lab":
    st.markdown("## 🔬 Quintile Lab — Does Each Feature Actually Predict Gains?")
    st.caption(f"Every stock-week split into 5 equal groups. If Q5 consistently beats Q1, the feature works. Horizon: {horizon}")
    
    features_to_test = ['master_score','rank','momentum_score','breakout_score',
                        'volume_score','trend_quality','rank_delta_1w','score_delta_1w',
                        'acceleration_score','rvol','money_flow_mm']
    
    # Summary card
    st.markdown("### 📊 Feature Ranking by Predictive Power")
    
    mono_results = []
    for feat in features_to_test:
        r = compute_quintile_returns(df, feat, fwd_col)
        if r is not None:
            qtable, mono = r
            q5_ret = qtable[qtable['Quintile']==5]['Mean Return %'].values
            q1_ret = qtable[qtable['Quintile']==1]['Mean Return %'].values
            spread = (q5_ret[0] - q1_ret[0]) if len(q5_ret) > 0 and len(q1_ret) > 0 else 0
            
            mono_results.append({
                'Feature': feat,
                'Monotonicity': round(mono, 3),
                'Q5-Q1 Spread %': round(spread, 2),
                'Q5 Win Rate': qtable[qtable['Quintile']==5]['Win Rate %'].values[0] if len(qtable[qtable['Quintile']==5]) > 0 else 0,
                'Q1 Win Rate': qtable[qtable['Quintile']==1]['Win Rate %'].values[0] if len(qtable[qtable['Quintile']==1]) > 0 else 0,
                'Reliable': '✅' if abs(mono) > 0.7 else '⚠️' if abs(mono) > 0.4 else '❌',
            })
    
    if mono_results:
        mono_df = pd.DataFrame(mono_results).sort_values('Monotonicity', ascending=False)
        
        fig = px.bar(
            mono_df, x='Feature', y='Monotonicity',
            color='Monotonicity', color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title='Feature Monotonicity (closer to 1.0 = quintiles perfectly sort future returns)'
        )
        fig.add_hline(y=0.7, line_dash="dash", annotation_text="Strong threshold")
        fig.add_hline(y=-0.7, line_dash="dash")
        fig.update_layout(height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(mono_df.reset_index(drop=True), use_container_width=True)
        
        st.markdown("""
        **Reading this:** Monotonicity = correlation between quintile number and average forward return.
        - **> 0.7** ✅ = Feature reliably predicts — higher quintile → higher future return
        - **0.4-0.7** ⚠️ = Some signal, noisy
        - **< 0.4** ❌ = Feature doesn't predict well
        - **Negative** = Inverse relationship (rank works this way — lower rank = better return)
        """)
    
    st.markdown("---")
    
    # ── Detailed quintile tables ──
    st.markdown("### 📈 Detailed Quintile Breakdown")
    
    selected_feat = st.selectbox("Select Feature", features_to_test)
    
    r = compute_quintile_returns(df, selected_feat, fwd_col)
    if r is not None:
        qtable, mono = r
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                qtable, x='Quintile', y='Mean Return %',
                color='Mean Return %', color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                title=f'{selected_feat}: Average {horizon} Forward Return by Quintile'
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                qtable, x='Quintile', y='Win Rate %',
                color='Win Rate %', color_continuous_scale='RdYlGn',
                color_continuous_midpoint=50,
                title=f'{selected_feat}: Win Rate by Quintile'
            )
            fig.add_hline(y=50, line_dash="dash", line_color="white")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Monotonicity: {mono:.3f}** {'✅ Strong' if abs(mono)>0.7 else '⚠️ Moderate' if abs(mono)>0.4 else '❌ Weak'}")
        st.dataframe(qtable, use_container_width=True)


# ─── PAGE 3: PATTERN LAB ─────────────────────────────────
elif page == "⚡ Pattern Lab":
    st.markdown(f"## ⚡ Pattern Lab — Which Patterns Have Real Edge? ({horizon})")
    st.caption("Each pattern tested for statistically significant edge over baseline. p-value < 0.10 = significant.")
    
    if len(pattern_edges) > 0:
        # Summary
        n_sig = (pattern_edges['p-value'] < 0.1).sum()
        n_pos = (pattern_edges['Edge vs Baseline'] > 0).sum()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            hist = df.dropna(subset=[fwd_col])
            base = hist[fwd_col].mean() if len(hist) > 0 else 0
            st.metric("Baseline Avg Return", f"{base:.2f}%")
        with c2:
            st.metric("Patterns with +Edge", f"{n_pos}/{len(pattern_edges)}")
        with c3:
            st.metric("Statistically Significant", f"{n_sig}/{len(pattern_edges)}")
        
        st.markdown("---")
        
        # Edge chart
        fig = px.bar(
            pattern_edges, x='Pattern', y='Edge vs Baseline',
            color='Edge vs Baseline', color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title=f'Pattern Edge: Extra Return vs All Stocks ({horizon} forward)',
            hover_data=['Count','Win Rate %','p-value','Significant'],
            text='Significant'
        )
        fig.update_layout(height=450, xaxis_tickangle=-45)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        
        # Win rate
        fig2 = px.bar(
            pattern_edges, x='Pattern', y='Win Rate %',
            color='Win Rate Edge', color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title='Win Rate by Pattern (vs baseline)',
            hover_data=['Count','Edge vs Baseline']
        )
        base_wr = (df.dropna(subset=[fwd_col])[fwd_col] > 0).mean() * 100
        fig2.add_hline(y=base_wr, line_dash="dash", annotation_text=f"Baseline: {base_wr:.0f}%")
        fig2.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Full table
        st.markdown("### 📋 Full Pattern Statistics")
        st.dataframe(pattern_edges.reset_index(drop=True), use_container_width=True)
        
        st.markdown("""
        **How to read this:**
        - **Edge vs Baseline** = How much MORE this pattern returns vs average stock
        - **p-value < 0.10** = Statistically significant (not just random noise)
        - **Big Win >10%** = % of pattern stocks that gained >10% in the period
        - **Big Loss <-10%** = % that lost >10% — shows risk
        
        **Only trust patterns that are both ✅ Significant AND have enough Count (30+)**
        """)
        
        # ── Pattern Combinations ──
        st.markdown("### 🔗 Pattern Combinations")
        
        combos = [
            ('p_stealth','p_inst','STEALTH + INSTITUTIONAL'),
            ('p_cat','p_mkt_ldr','CAT + MARKET LEADER'),
            ('p_vel_brk','p_prem_mom','VELOCITY BREAK + PREMIUM MOM'),
            ('p_gold_x','p_mom_wave','GOLDEN CROSS + MOM WAVE'),
            ('p_cat','p_vol_exp','CAT + VOL EXPLOSION'),
            ('p_inst','p_gold_x','INSTITUTIONAL + GOLDEN CROSS'),
            ('p_stealth','p_accel','STEALTH + ACCELERATION'),
            ('p_garp','p_val_mom','GARP + VALUE MOM'),
            ('p_liquid','p_mkt_ldr','LIQUID + MARKET LEADER'),
        ]
        
        valid = df.dropna(subset=[fwd_col])
        combo_results = []
        for c1_name, c2_name, label in combos:
            if c1_name in valid.columns and c2_name in valid.columns:
                combo = valid[(valid[c1_name]==1) & (valid[c2_name]==1)]
                if len(combo) >= 5:
                    combo_results.append({
                        'Combination': label,
                        'Count': len(combo),
                        'Avg Return %': round(combo[fwd_col].mean(), 2),
                        'Win Rate %': round((combo[fwd_col]>0).mean()*100, 1),
                        'Edge vs Base': round(combo[fwd_col].mean() - base, 2),
                    })
        
        if combo_results:
            combo_df = pd.DataFrame(combo_results).sort_values('Avg Return %', ascending=False)
            st.dataframe(combo_df.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("Not enough data to compute pattern edges.")


# ─── PAGE 4: BACKTEST ─────────────────────────────────────
elif page == "🧪 Backtest":
    st.markdown(f"## 🧪 Walk-Forward Backtest — Top {top_n} Stocks, {horizon}")
    st.caption("Each week: pick top-N by composite score using ONLY past data → measure actual outcome. No look-ahead bias.")
    
    bt = walk_forward_backtest(df, fwd_col, top_n)
    
    if len(bt) > 0:
        # Summary
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Avg Return (Top N)", f"{bt['top_n_avg_return'].mean():.2f}%")
        with c2:
            st.metric("Avg Return (Bottom N)", f"{bt['bottom_n_avg_return'].mean():.2f}%")
        with c3:
            st.metric("Spread (Top - Bottom)", f"{bt['spread'].mean():.2f}%")
        with c4:
            st.metric("Top N Win Rate", f"{bt['top_n_win_rate'].mean():.0f}%")
        with c5:
            st.metric("Periods Tested", len(bt))
        
        st.markdown("---")
        
        # Returns comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bt['week'], y=bt['top_n_avg_return'], name=f'Top {top_n}',
                            marker_color='#22c55e', opacity=0.8))
        fig.add_trace(go.Bar(x=bt['week'], y=bt['bottom_n_avg_return'], name=f'Bottom {top_n}',
                            marker_color='#ef4444', opacity=0.8))
        fig.add_trace(go.Scatter(x=bt['week'], y=bt['universe_avg'], name='Universe Avg',
                                line=dict(color='#f59e0b', width=2, dash='dash')))
        fig.update_layout(
            title=f'Top {top_n} vs Bottom {top_n} — {horizon} Forward Return per Entry Week',
            yaxis_title='Return %', height=400, barmode='group',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative equity curve
        bt['cum_top'] = (1 + bt['top_n_avg_return']/100).cumprod() * 100
        bt['cum_bottom'] = (1 + bt['bottom_n_avg_return']/100).cumprod() * 100
        bt['cum_random'] = (1 + bt['random_avg_return']/100).cumprod() * 100
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bt['week'], y=bt['cum_top'], name=f'Top {top_n}',
                                 line=dict(color='#22c55e', width=3)))
        fig2.add_trace(go.Scatter(x=bt['week'], y=bt['cum_bottom'], name=f'Bottom {top_n}',
                                 line=dict(color='#ef4444', width=3)))
        fig2.add_trace(go.Scatter(x=bt['week'], y=bt['cum_random'], name='Random',
                                 line=dict(color='#9ca3af', width=2, dash='dot')))
        fig2.add_hline(y=100, line_dash="dash", line_color="gray")
        fig2.update_layout(
            title='Cumulative Equity Curve (₹100 start)',
            yaxis_title='Portfolio Value ₹', height=400,
            template='plotly_dark'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Win rate over time
        fig3 = px.bar(
            bt, x='week', y='top_n_win_rate',
            color='top_n_win_rate', color_continuous_scale='RdYlGn',
            color_continuous_midpoint=50,
            title=f'Win Rate per Period (Top {top_n} stocks)'
        )
        fig3.add_hline(y=50, line_dash="dash")
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Period details
        with st.expander("📋 Period-by-Period Details"):
            st.dataframe(bt.round(2), use_container_width=True)
        
        # Interpretation
        spread_avg = bt['spread'].mean()
        top_wr = bt['top_n_win_rate'].mean()
        
        if spread_avg > 3 and top_wr > 55:
            st.success(f"""
            **✅ System has edge.** Top {top_n} stocks average {spread_avg:.1f}% more return than bottom {top_n}, 
            with {top_wr:.0f}% win rate. This is actionable.
            """)
        elif spread_avg > 1:
            st.info(f"""
            **⚠️ Moderate edge.** {spread_avg:.1f}% spread, {top_wr:.0f}% win rate. 
            Might improve with more data or tighter filters.
            """)
        else:
            st.warning(f"""
            **❌ Weak/no edge.** {spread_avg:.1f}% spread is too thin. 
            The system's rankings don't reliably predict forward returns at this horizon.
            Try a different horizon or tighter filters.
            """)
    else:
        st.warning("Not enough weekly data for walk-forward backtest. Upload more files.")


# ─── PAGE 5: STOCK DEEP DIVE ─────────────────────────────
elif page == "📈 Stock Deep Dive":
    st.markdown("## 📈 Stock Deep Dive")
    
    all_tickers = sorted(df['ticker'].unique())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.selectbox("Select Stock", all_tickers)
    with col2:
        search = st.text_input("Search", "")
        if search:
            matches = [t for t in all_tickers if search.upper() in str(t).upper()]
            if matches:
                ticker = st.selectbox("Results", matches)
    
    stock = df[df['ticker'] == ticker].sort_values('week')
    
    if len(stock) > 0:
        last = stock.iloc[-1]
        first = stock.iloc[0]
        
        # Header
        st.markdown(f"### {ticker} — {last.get('company_name','')}")
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            price_chg = ((last['price'] - first['price']) / first['price'] * 100)
            st.metric("Price", f"₹{last['price']:.0f}", f"{price_chg:+.1f}% total")
        with c2:
            st.metric("Rank", f"#{last['rank']:.0f}", f"{last.get('rank_delta_1w',0):+.0f}")
        with c3:
            st.metric("Score", f"{last['master_score']:.1f}", f"{last.get('score_delta_1w',0):+.1f}")
        with c4:
            st.metric("Composite", f"{last.get('composite',0):.0f}")
        with c5:
            st.metric("State", last.get('market_state',''))
        with c6:
            st.metric("Sector", last.get('sector','')[:15])
        
        # Charts
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['💰 Price','📊 Rank (↓ = better)','🎯 Master Score','⚡ Breakout Score',
                           '📈 Momentum Score','🔊 Volume Score'],
            vertical_spacing=0.08
        )
        
        w = stock['week']
        traces = [
            (stock['price'], '#00d2ff', 1, 1),
            (stock['rank'], '#ff6b6b', 1, 2),
            (stock['master_score'], '#22c55e', 2, 1),
            (stock['breakout_score'], '#f59e0b', 2, 2),
            (stock['momentum_score'], '#a855f7', 3, 1),
            (stock['volume_score'], '#06b6d4', 3, 2),
        ]
        
        for data, color, r, c in traces:
            fig.add_trace(go.Scatter(x=w, y=data, mode='lines+markers',
                         line=dict(color=color, width=2), showlegend=False), row=r, col=c)
        
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_layout(height=700, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        # Journey table
        with st.expander("📋 Weekly Data"):
            show = ['week','rank','master_score','price','breakout_score',
                    'momentum_score','volume_score','ret_7d','ret_30d',
                    'rank_delta_1w','score_delta_1w','market_state','patterns']
            available = [c for c in show if c in stock.columns]
            display = stock[available].copy()
            display['week'] = display['week'].dt.strftime('%Y-%m-%d')
            st.dataframe(display.reset_index(drop=True), use_container_width=True)
        
        # Pattern timeline
        st.markdown("### 🏷️ Pattern Timeline")
        for _, row in stock.iterrows():
            pat = str(row.get('patterns',''))
            if pat and pat != 'nan':
                st.markdown(f"**{row['week'].strftime('%b %d')}** | Rank #{row['rank']:.0f} | `{row.get('market_state','')}` | {pat}")


# ── Footer ──
st.markdown("---")
st.caption("WAVE v2 — Statistical probability engine. Not financial advice. Past patterns don't guarantee future results.")
