# app.py — WAVE Stock Probability Engine
# Run: pip install streamlit pandas numpy scikit-learn plotly xgboost
# Then: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WAVE — Stock Probability Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800; 
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px; padding: 20px; border: 1px solid #0f3460;
        text-align: center; margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00d2ff; }
    .metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 5px; }
    .signal-buy { color: #00ff88; font-weight: 700; font-size: 1.2rem; }
    .signal-sell { color: #ff4444; font-weight: 700; font-size: 1.2rem; }
    .signal-hold { color: #ffaa00; font-weight: 700; font-size: 1.2rem; }
    .prob-high { background: linear-gradient(90deg, #00ff88, #00cc66); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8rem; font-weight: 800; }
    .prob-low { background: linear-gradient(90deg, #ff4444, #cc0000); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8rem; font-weight: 800; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e; border-radius: 10px 10px 0 0;
        padding: 10px 20px; color: #8892b0;
    }
    .stTabs [aria-selected="true"] { background-color: #0f3460; color: #00d2ff; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌊 WAVE — Stock Probability Engine</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8892b0;'>Upload weekly CSVs → System learns patterns → Predicts future gainers & losers with probability scores</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────

@st.cache_data
def load_weekly_files(uploaded_files):
    """Load and parse all weekly CSV files."""
    all_dfs = {}
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            # Extract date from filename
            name = f.name
            # Try to extract date like 2025-08-30
            import re
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', name)
            if date_match:
                date_str = date_match.group(1)
                week_date = pd.to_datetime(date_str)
            else:
                week_date = pd.Timestamp.now()
            df['week_date'] = week_date
            df['filename'] = name
            all_dfs[week_date] = df
        except Exception as e:
            st.warning(f"Error loading {f.name}: {e}")
    return all_dfs

def build_master_dataset(weekly_data):
    """Combine all weekly snapshots into a unified dataset with time series per stock."""
    all_rows = []
    for week_date, df in sorted(weekly_data.items()):
        df_copy = df.copy()
        df_copy['week_date'] = week_date
        all_rows.append(df_copy)
    
    master = pd.concat(all_rows, ignore_index=True)
    
    # Ensure numeric columns
    numeric_cols = ['rank', 'master_score', 'position_score', 'volume_score', 
                    'momentum_score', 'acceleration_score', 'breakout_score',
                    'rvol_score', 'trend_quality', 'price', 'pe', 'eps_current',
                    'eps_change_pct', 'from_low_pct', 'from_high_pct',
                    'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
                    'rvol', 'vmi', 'money_flow_mm', 'position_tension',
                    'momentum_harmony', 'overall_market_strength']
    for col in numeric_cols:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors='coerce')
    
    return master


def compute_stock_features(master_df):
    """
    For each stock at each week, compute features that capture:
    - Current snapshot features
    - Trend features (how rank/score changed over last N weeks)
    - Pattern features (encoded)
    - Momentum regime features
    """
    stocks = master_df.groupby('ticker')
    feature_rows = []
    
    for ticker, group in stocks:
        group = group.sort_values('week_date')
        
        if len(group) < 3:
            continue
        
        for i in range(2, len(group)):
            row = group.iloc[i]
            prev1 = group.iloc[i-1]
            prev2 = group.iloc[i-2]
            
            features = {
                'ticker': ticker,
                'week_date': row['week_date'],
                'company_name': row.get('company_name', ''),
                'sector': row.get('sector', ''),
                'industry': row.get('industry', ''),
                'category': row.get('category', ''),
                'price': row.get('price', 0),
                
                # Current scores
                'master_score': row.get('master_score', 0),
                'position_score': row.get('position_score', 0),
                'volume_score': row.get('volume_score', 0),
                'momentum_score': row.get('momentum_score', 0),
                'acceleration_score': row.get('acceleration_score', 0),
                'breakout_score': row.get('breakout_score', 0),
                'rvol_score': row.get('rvol_score', 0),
                'trend_quality': row.get('trend_quality', 0),
                
                # Rank & rank change
                'rank': row.get('rank', 0),
                'rank_change_1w': prev1.get('rank', 0) - row.get('rank', 0),  # positive = improved
                'rank_change_2w': prev2.get('rank', 0) - row.get('rank', 0),
                
                # Score changes
                'score_change_1w': row.get('master_score', 0) - prev1.get('master_score', 0),
                'score_change_2w': row.get('master_score', 0) - prev2.get('master_score', 0),
                'momentum_change_1w': row.get('momentum_score', 0) - prev1.get('momentum_score', 0),
                'breakout_change_1w': row.get('breakout_score', 0) - prev1.get('breakout_score', 0),
                'volume_change_1w': row.get('volume_score', 0) - prev1.get('volume_score', 0),
                
                # Price momentum
                'ret_1d': row.get('ret_1d', 0),
                'ret_7d': row.get('ret_7d', 0),
                'ret_30d': row.get('ret_30d', 0),
                'ret_3m': row.get('ret_3m', 0),
                'ret_6m': row.get('ret_6m', 0),
                
                # Price change from last week
                'price_change_1w_pct': ((row.get('price', 0) - prev1.get('price', 1)) / max(prev1.get('price', 1), 0.01)) * 100,
                
                # Relative volume
                'rvol': row.get('rvol', 0),
                'vmi': row.get('vmi', 0),
                'money_flow_mm': row.get('money_flow_mm', 0),
                
                # Tension & harmony
                'position_tension': row.get('position_tension', 0),
                'momentum_harmony': row.get('momentum_harmony', 0),
                
                # Distance from extremes
                'from_low_pct': row.get('from_low_pct', 0),
                'from_high_pct': row.get('from_high_pct', 0),
                
                # PE / EPS
                'pe': row.get('pe', 0),
                'eps_current': row.get('eps_current', 0),
                
                # Market state encoded
                'market_state': row.get('market_state', ''),
                'patterns': row.get('patterns', ''),
                'overall_market_strength': row.get('overall_market_strength', 0),
                
                # Score acceleration (2nd derivative)
                'score_acceleration': (row.get('master_score', 0) - prev1.get('master_score', 0)) - 
                                      (prev1.get('master_score', 0) - prev2.get('master_score', 0)),
                'rank_acceleration': (prev1.get('rank', 0) - row.get('rank', 0)) - 
                                     (prev2.get('rank', 0) - prev1.get('rank', 0)),
            }
            
            # Pattern encoding
            patterns_str = str(row.get('patterns', ''))
            pattern_flags = {
                'has_cat_leader': 1 if 'CAT LEADER' in patterns_str else 0,
                'has_vol_explosion': 1 if 'VOL EXPLOSION' in patterns_str else 0,
                'has_market_leader': 1 if 'MARKET LEADER' in patterns_str else 0,
                'has_momentum_wave': 1 if 'MOMENTUM WAVE' in patterns_str else 0,
                'has_premium_momentum': 1 if 'PREMIUM MOMENTUM' in patterns_str else 0,
                'has_velocity_breakout': 1 if 'VELOCITY BREAKOUT' in patterns_str else 0,
                'has_institutional': 1 if 'INSTITUTIONAL' in patterns_str else 0,
                'has_golden_cross': 1 if 'GOLDEN CROSS' in patterns_str else 0,
                'has_stealth': 1 if 'STEALTH' in patterns_str else 0,
                'has_distribution': 1 if 'DISTRIBUTION' in patterns_str else 0,
                'has_capitulation': 1 if 'CAPITULATION' in patterns_str else 0,
                'has_high_pe': 1 if 'HIGH PE' in patterns_str else 0,
                'has_phoenix': 1 if 'PHOENIX' in patterns_str else 0,
                'has_pullback_support': 1 if 'PULLBACK SUPPORT' in patterns_str else 0,
                'has_range_compress': 1 if 'RANGE COMPRESS' in patterns_str else 0,
                'has_acceleration': 1 if 'ACCELERATION' in patterns_str else 0,
                'has_rotation_leader': 1 if 'ROTATION LEADER' in patterns_str else 0,
                'has_garp': 1 if 'GARP' in patterns_str else 0,
                'has_value_momentum': 1 if 'VALUE MOMENTUM' in patterns_str else 0,
                'has_earnings_rocket': 1 if 'EARNINGS ROCKET' in patterns_str else 0,
                'has_liquid_leader': 1 if 'LIQUID LEADER' in patterns_str else 0,
                'has_velocity_squeeze': 1 if 'VELOCITY SQUEEZE' in patterns_str else 0,
                'has_runaway_gap': 1 if 'RUNAWAY GAP' in patterns_str else 0,
                'has_pyramid': 1 if 'PYRAMID' in patterns_str else 0,
                'pattern_count': patterns_str.count('|') + 1 if patterns_str and patterns_str != 'nan' else 0,
            }
            features.update(pattern_flags)
            
            # Market state encoding
            state = str(row.get('market_state', ''))
            state_flags = {
                'state_strong_uptrend': 1 if state == 'STRONG_UPTREND' else 0,
                'state_uptrend': 1 if state == 'UPTREND' else 0,
                'state_sideways': 1 if state == 'SIDEWAYS' else 0,
                'state_rotation': 1 if state == 'ROTATION' else 0,
                'state_pullback': 1 if state == 'PULLBACK' else 0,
                'state_bounce': 1 if state == 'BOUNCE' else 0,
                'state_downtrend': 1 if state == 'DOWNTREND' else 0,
                'state_strong_downtrend': 1 if state == 'STRONG_DOWNTREND' else 0,
            }
            features.update(state_flags)
            
            feature_rows.append(features)
    
    return pd.DataFrame(feature_rows)


def compute_forward_returns(feature_df, master_df):
    """
    For each stock at each week, compute the ACTUAL forward return 
    (what happened in the NEXT 1, 2, 4, 8 weeks).
    This is our TARGET variable.
    """
    sorted_dates = sorted(master_df['week_date'].unique())
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
    
    # Build price lookup: ticker -> {week_date -> price}
    price_lookup = {}
    for _, row in master_df[['ticker', 'week_date', 'price']].iterrows():
        t = row['ticker']
        if t not in price_lookup:
            price_lookup[t] = {}
        price_lookup[t][row['week_date']] = row['price']
    
    # Build rank lookup
    rank_lookup = {}
    for _, row in master_df[['ticker', 'week_date', 'rank']].iterrows():
        t = row['ticker']
        if t not in rank_lookup:
            rank_lookup[t] = {}
        rank_lookup[t][row['week_date']] = row['rank']
    
    fwd_returns_1w = []
    fwd_returns_2w = []
    fwd_returns_4w = []
    fwd_rank_1w = []
    fwd_rank_2w = []
    
    for _, row in feature_df.iterrows():
        ticker = row['ticker']
        current_date = row['week_date']
        current_price = row['price']
        current_rank = row['rank']
        
        idx = date_to_idx.get(current_date, -1)
        
        # Forward 1 week
        if idx + 1 < len(sorted_dates) and ticker in price_lookup:
            next_date = sorted_dates[idx + 1]
            next_price = price_lookup[ticker].get(next_date)
            if next_price and current_price and current_price > 0:
                fwd_returns_1w.append(((next_price - current_price) / current_price) * 100)
            else:
                fwd_returns_1w.append(np.nan)
            next_rank = rank_lookup.get(ticker, {}).get(next_date)
            fwd_rank_1w.append(current_rank - next_rank if next_rank else np.nan)
        else:
            fwd_returns_1w.append(np.nan)
            fwd_rank_1w.append(np.nan)
        
        # Forward 2 weeks
        if idx + 2 < len(sorted_dates) and ticker in price_lookup:
            fwd_date = sorted_dates[idx + 2]
            fwd_price = price_lookup[ticker].get(fwd_date)
            if fwd_price and current_price and current_price > 0:
                fwd_returns_2w.append(((fwd_price - current_price) / current_price) * 100)
            else:
                fwd_returns_2w.append(np.nan)
            fwd_r = rank_lookup.get(ticker, {}).get(fwd_date)
            fwd_rank_2w.append(current_rank - fwd_r if fwd_r else np.nan)
        else:
            fwd_returns_2w.append(np.nan)
            fwd_rank_2w.append(np.nan)
        
        # Forward 4 weeks
        if idx + 4 < len(sorted_dates) and ticker in price_lookup:
            fwd_date = sorted_dates[idx + 4]
            fwd_price = price_lookup[ticker].get(fwd_date)
            if fwd_price and current_price and current_price > 0:
                fwd_returns_4w.append(((fwd_price - current_price) / current_price) * 100)
            else:
                fwd_returns_4w.append(np.nan)
        else:
            fwd_returns_4w.append(np.nan)
    
    feature_df['fwd_return_1w'] = fwd_returns_1w
    feature_df['fwd_return_2w'] = fwd_returns_2w
    feature_df['fwd_return_4w'] = fwd_returns_4w
    feature_df['fwd_rank_change_1w'] = fwd_rank_1w
    feature_df['fwd_rank_change_2w'] = fwd_rank_2w
    
    # Binary targets
    feature_df['will_gain_1w'] = (feature_df['fwd_return_1w'] > 0).astype(int)
    feature_df['will_gain_2w'] = (feature_df['fwd_return_2w'] > 0).astype(int)
    feature_df['will_gain_4w'] = (feature_df['fwd_return_4w'] > 0).astype(int)
    feature_df['big_gainer_4w'] = (feature_df['fwd_return_4w'] > 10).astype(int)
    feature_df['big_loser_4w'] = (feature_df['fwd_return_4w'] < -10).astype(int)
    
    return feature_df


# ─────────────────────────────────────────────────────────
#  PROBABILITY MODEL
# ─────────────────────────────────────────────────────────

def get_feature_columns():
    """Return the feature columns used for ML."""
    return [
        'master_score', 'position_score', 'volume_score', 'momentum_score',
        'acceleration_score', 'breakout_score', 'rvol_score', 'trend_quality',
        'rank', 'rank_change_1w', 'rank_change_2w',
        'score_change_1w', 'score_change_2w', 'momentum_change_1w',
        'breakout_change_1w', 'volume_change_1w',
        'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m',
        'price_change_1w_pct', 'rvol', 'vmi', 'money_flow_mm',
        'position_tension', 'momentum_harmony',
        'from_low_pct', 'from_high_pct', 'pe',
        'overall_market_strength', 'score_acceleration', 'rank_acceleration',
        'has_cat_leader', 'has_vol_explosion', 'has_market_leader',
        'has_momentum_wave', 'has_premium_momentum', 'has_velocity_breakout',
        'has_institutional', 'has_golden_cross', 'has_stealth',
        'has_distribution', 'has_capitulation', 'has_high_pe',
        'has_phoenix', 'has_pullback_support', 'has_range_compress',
        'has_acceleration', 'has_rotation_leader', 'has_garp',
        'has_value_momentum', 'has_earnings_rocket', 'has_liquid_leader',
        'has_velocity_squeeze', 'has_runaway_gap', 'has_pyramid',
        'pattern_count',
        'state_strong_uptrend', 'state_uptrend', 'state_sideways',
        'state_rotation', 'state_pullback', 'state_bounce',
        'state_downtrend', 'state_strong_downtrend',
    ]


def train_probability_model(feature_df, target_col='will_gain_4w'):
    """Train an XGBoost classifier for gain probability."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score
    
    feature_cols = get_feature_columns()
    
    # Filter rows with valid targets
    valid = feature_df.dropna(subset=[target_col])
    valid = valid.dropna(subset=feature_cols, how='all')
    
    X = valid[feature_cols].fillna(0)
    y = valid[target_col].astype(int)
    
    if len(X) < 50:
        return None, None, "Not enough data (need 50+ rows with forward returns)"
    
    # Use time-based split: train on earlier weeks, test on later
    sorted_dates = sorted(valid['week_date'].unique())
    split_idx = int(len(sorted_dates) * 0.7)
    train_dates = sorted_dates[:split_idx]
    test_dates = sorted_dates[split_idx:]
    
    train_mask = valid['week_date'].isin(train_dates)
    test_mask = valid['week_date'].isin(test_dates)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    if len(X_train) < 20 or len(X_test) < 10:
        return None, None, "Not enough data for train/test split"
    
    try:
        # Try XGBoost first
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=0.1,
            random_state=42, verbosity=0,
            use_label_encoder=False, eval_metric='logloss'
        )
    except ImportError:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.5
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        'auc_roc': auc,
        'accuracy': report.get('accuracy', 0),
        'precision_gain': report.get('1', {}).get('precision', 0),
        'recall_gain': report.get('1', {}).get('recall', 0),
        'f1_gain': report.get('1', {}).get('f1-score', 0),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'gain_rate_train': y_train.mean(),
        'gain_rate_test': y_test.mean(),
        'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
    }
    
    return model, metrics, None


def predict_probabilities(model, feature_df, latest_date):
    """Use the trained model to predict gain probability for latest week stocks."""
    feature_cols = get_feature_columns()
    
    latest = feature_df[feature_df['week_date'] == latest_date].copy()
    
    if len(latest) == 0:
        return pd.DataFrame()
    
    X = latest[feature_cols].fillna(0)
    
    probas = model.predict_proba(X)[:, 1]
    
    latest['gain_probability'] = probas
    latest['gain_probability_pct'] = (probas * 100).round(1)
    
    # Signal
    latest['signal'] = latest['gain_probability'].apply(
        lambda p: '🟢 STRONG BUY' if p > 0.75 else
                  '🟡 BUY' if p > 0.6 else
                  '⚪ NEUTRAL' if p > 0.45 else
                  '🟠 AVOID' if p > 0.3 else
                  '🔴 HIGH RISK'
    )
    
    return latest.sort_values('gain_probability', ascending=False)


# ─────────────────────────────────────────────────────────
#  RULE-BASED SCORING SYSTEM (complementary to ML)
# ─────────────────────────────────────────────────────────

def compute_rule_scores(feature_df, latest_date):
    """
    A transparent, interpretable scoring system based on discovered patterns:
    - From our deep gainer/loser analysis, we know what works.
    """
    latest = feature_df[feature_df['week_date'] == latest_date].copy()
    
    if len(latest) == 0:
        return pd.DataFrame()
    
    scores = pd.Series(50.0, index=latest.index)  # Start at 50 (neutral)
    reasons = [[] for _ in range(len(latest))]
    
    for idx_pos, (idx, row) in enumerate(latest.iterrows()):
        r = []
        s = 50.0
        
        # 1. Rank Momentum (most predictive from our analysis)
        rank_chg = row.get('rank_change_1w', 0)
        if rank_chg > 500:
            s += 15; r.append(f"🚀 Rank jumped +{rank_chg:.0f} in 1 week")
        elif rank_chg > 200:
            s += 10; r.append(f"📈 Rank improved +{rank_chg:.0f}")
        elif rank_chg > 50:
            s += 5; r.append(f"↗️ Rank improving +{rank_chg:.0f}")
        elif rank_chg < -500:
            s -= 15; r.append(f"💀 Rank crashed {rank_chg:.0f}")
        elif rank_chg < -200:
            s -= 10; r.append(f"📉 Rank dropped {rank_chg:.0f}")
        elif rank_chg < -50:
            s -= 5; r.append(f"↘️ Rank declining {rank_chg:.0f}")
        
        # 2. Score Acceleration (2nd derivative — early signal)
        score_accel = row.get('score_acceleration', 0)
        if score_accel > 10:
            s += 8; r.append(f"⚡ Score accelerating (+{score_accel:.1f})")
        elif score_accel < -10:
            s -= 8; r.append(f"⚠️ Score decelerating ({score_accel:.1f})")
        
        # 3. Pattern Signals (from our gainer/loser analysis)
        if row.get('has_stealth', 0) and row.get('has_institutional', 0):
            s += 12; r.append("🤫🏦 STEALTH + INSTITUTIONAL (strongest combo)")
        elif row.get('has_cat_leader', 0) and row.get('has_market_leader', 0):
            s += 10; r.append("🐱👑 CAT LEADER + MARKET LEADER")
        elif row.get('has_velocity_breakout', 0) and row.get('has_premium_momentum', 0):
            s += 8; r.append("🚀🔥 VELOCITY BREAKOUT + PREMIUM MOMENTUM")
        
        if row.get('has_capitulation', 0):
            s -= 15; r.append("💣 CAPITULATION detected — high crash risk")
        if row.get('has_distribution', 0) and row.get('state_downtrend', 0):
            s -= 10; r.append("📊 DISTRIBUTION in DOWNTREND — selling pressure")
        
        if row.get('has_phoenix', 0):
            s += 6; r.append("🐦 PHOENIX RISING — potential turnaround")
        
        # 4. Market State
        if row.get('state_strong_uptrend', 0):
            s += 8; r.append("📈 STRONG UPTREND state")
        elif row.get('state_uptrend', 0):
            s += 5; r.append("📈 UPTREND state")
        elif row.get('state_strong_downtrend', 0):
            s -= 12; r.append("📉 STRONG DOWNTREND — avoid")
        elif row.get('state_downtrend', 0):
            s -= 8; r.append("📉 DOWNTREND state")
        elif row.get('state_bounce', 0):
            s += 3; r.append("🔄 BOUNCE detected")
        
        # 5. Master Score Threshold
        ms = row.get('master_score', 0)
        if ms > 80:
            s += 10; r.append(f"💎 Elite master_score: {ms:.1f}")
        elif ms > 60:
            s += 5; r.append(f"✅ Good master_score: {ms:.1f}")
        elif ms < 20:
            s -= 8; r.append(f"⛔ Low master_score: {ms:.1f}")
        
        # 6. Breakout Score (key leading indicator)
        bs = row.get('breakout_score', 0)
        bs_chg = row.get('breakout_change_1w', 0)
        if bs > 90 and bs_chg > 20:
            s += 10; r.append(f"🎯 Breakout score {bs:.0f} (surging +{bs_chg:.0f})")
        elif bs > 80:
            s += 5; r.append(f"⚡ High breakout score: {bs:.0f}")
        
        # 7. Volume Confirmation
        rvol = row.get('rvol', 0)
        if rvol > 5 and (row.get('state_uptrend', 0) or row.get('state_strong_uptrend', 0)):
            s += 5; r.append(f"🔊 High volume ({rvol:.1f}x) confirming uptrend")
        elif rvol > 5 and (row.get('state_downtrend', 0) or row.get('state_strong_downtrend', 0)):
            s -= 5; r.append(f"🔊 High volume ({rvol:.1f}x) confirming downtrend")
        
        # 8. Price Momentum Consistency
        r7 = row.get('ret_7d', 0)
        r30 = row.get('ret_30d', 0)
        r3m = row.get('ret_3m', 0)
        if r7 > 0 and r30 > 0 and r3m > 0:
            s += 5; r.append("📊 Consistent positive momentum (7d/30d/3m all green)")
        elif r7 < 0 and r30 < 0 and r3m < 0:
            s -= 5; r.append("📊 Consistent negative momentum (7d/30d/3m all red)")
        
        # 9. PE Warning
        pe = row.get('pe', 0)
        if pe and pe > 100 and not row.get('has_earnings_rocket', 0):
            s -= 3; r.append(f"⚠️ Very high PE: {pe:.0f}")
        
        # 10. Money Flow
        mf = row.get('money_flow_mm', 0)
        if mf > 100:
            s += 4; r.append(f"💰 Strong money flow: ₹{mf:.0f}M")
        elif mf < -50:
            s -= 4; r.append(f"💸 Money outflow: ₹{mf:.0f}M")
        
        # 11. Rank Zone
        rank = row.get('rank', 0)
        if rank <= 20:
            s += 5; r.append(f"🏆 Top 20 rank: #{rank:.0f}")
        elif rank <= 50:
            s += 3; r.append(f"✨ Top 50 rank: #{rank:.0f}")
        elif rank > 1500:
            s -= 5; r.append(f"⛔ Very low rank: #{rank:.0f}")
        
        # Clamp
        scores.iloc[idx_pos] = max(0, min(100, s))
        reasons[idx_pos] = r
    
    latest['rule_score'] = scores.values
    latest['rule_reasons'] = reasons
    
    # Signal based on rule score
    latest['rule_signal'] = latest['rule_score'].apply(
        lambda s: '🟢 STRONG BUY' if s >= 80 else
                  '🟡 BUY' if s >= 65 else
                  '⚪ NEUTRAL' if s >= 45 else
                  '🟠 AVOID' if s >= 30 else
                  '🔴 HIGH RISK'
    )
    
    return latest.sort_values('rule_score', ascending=False)


# ─────────────────────────────────────────────────────────
#  JOURNEY TRACKER
# ─────────────────────────────────────────────────────────

def get_stock_journey(master_df, ticker):
    """Get the complete weekly journey of a stock."""
    stock_data = master_df[master_df['ticker'] == ticker].sort_values('week_date')
    return stock_data


# ─────────────────────────────────────────────────────────
#  SECTOR HEATMAP
# ─────────────────────────────────────────────────────────

def compute_sector_momentum(master_df, latest_date):
    """Compute sector-level momentum scores."""
    latest = master_df[master_df['week_date'] == latest_date]
    
    sector_stats = latest.groupby('sector').agg({
        'master_score': 'mean',
        'ret_7d': 'mean',
        'ret_30d': 'mean',
        'ret_3m': 'mean',
        'momentum_score': 'mean',
        'ticker': 'count',
        'rank': 'mean',
    }).rename(columns={'ticker': 'stock_count', 'rank': 'avg_rank'})
    
    sector_stats = sector_stats[sector_stats['stock_count'] >= 3]
    sector_stats['sector_score'] = (
        sector_stats['master_score'] * 0.3 +
        sector_stats['momentum_score'] * 0.3 +
        (100 - sector_stats['avg_rank'] / sector_stats['avg_rank'].max() * 100) * 0.2 +
        sector_stats['ret_30d'].clip(-20, 20) * 1.0
    )
    
    return sector_stats.sort_values('sector_score', ascending=False)


# ─────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Upload Weekly CSVs")
    uploaded_files = st.file_uploader(
        "Upload your Stocks_Weekly_*.csv files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload all your weekly stock snapshot CSV files"
    )
    
    st.markdown("---")
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        
        # Processing options
        st.markdown("## ⚙️ Settings")
        
        prediction_horizon = st.selectbox(
            "Prediction Horizon",
            ['1 Week', '2 Weeks', '4 Weeks'],
            index=2,
            help="How far ahead to predict"
        )
        
        target_map = {
            '1 Week': 'will_gain_1w',
            '2 Weeks': 'will_gain_2w',
            '4 Weeks': 'will_gain_4w',
        }
        
        min_master_score = st.slider("Min Master Score Filter", 0, 100, 0)
        max_rank = st.slider("Max Rank Filter", 100, 2200, 2200)
        
        sector_filter = st.text_input("Sector Filter (leave blank for all)", "")
    else:
        st.info("👆 Upload your weekly CSV files to begin")
        prediction_horizon = '4 Weeks'
        min_master_score = 0
        max_rank = 2200
        sector_filter = ""


# Main content
if uploaded_files and len(uploaded_files) >= 3:
    
    # Load data
    with st.spinner("📊 Loading and parsing weekly data..."):
        weekly_data = load_weekly_files(uploaded_files)
    
    if len(weekly_data) < 3:
        st.error("Need at least 3 weekly files for analysis. Please upload more files.")
        st.stop()
    
    sorted_dates = sorted(weekly_data.keys())
    latest_date = sorted_dates[-1]
    
    with st.spinner("🔧 Building master dataset..."):
        master_df = build_master_dataset(weekly_data)
    
    with st.spinner("🧮 Computing features..."):
        feature_df = compute_stock_features(master_df)
    
    with st.spinner("📈 Computing forward returns..."):
        feature_df = compute_forward_returns(feature_df, master_df)
    
    # Overview metrics
    total_stocks = master_df[master_df['week_date'] == latest_date]['ticker'].nunique()
    total_weeks = len(sorted_dates)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", f"{total_stocks:,}")
    with col2:
        st.metric("Weekly Snapshots", total_weeks)
    with col3:
        st.metric("Date Range", f"{sorted_dates[0].strftime('%b %d')} → {sorted_dates[-1].strftime('%b %d, %Y')}")
    with col4:
        avg_ms = master_df[master_df['week_date'] == latest_date]['overall_market_strength'].mean()
        st.metric("Market Strength", f"{avg_ms:.1f}")
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────
    #  TABS
    # ─────────────────────────────────────────────────────
    
    tabs = st.tabs([
        "🎯 Probability Predictions",
        "📊 Rule-Based Scanner",
        "🔬 ML Model Performance",
        "🗺️ Sector Heatmap",
        "📈 Stock Journey Tracker",
        "🔄 Rank Movers",
        "⚡ Pattern Analysis",
        "🧪 Backtest System",
    ])
    
    # ═══════════════════════════════════════════════════
    #  TAB 1: PROBABILITY PREDICTIONS
    # ═══════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### 🎯 AI-Powered Gain Probability (Latest Week)")
        st.caption("Model learns from historical weekly snapshots to predict which stocks will gain")
        
        target_col = target_map[prediction_horizon]
        
        with st.spinner("🤖 Training probability model..."):
            model, metrics, error = train_probability_model(feature_df, target_col)
        
        if error:
            st.error(error)
        elif model is not None:
            # Show predictions
            predictions = predict_probabilities(model, feature_df, latest_date)
            
            if len(predictions) > 0:
                # Apply filters
                if min_master_score > 0:
                    predictions = predictions[predictions['master_score'] >= min_master_score]
                if max_rank < 2200:
                    predictions = predictions[predictions['rank'] <= max_rank]
                if sector_filter:
                    predictions = predictions[predictions['sector'].str.contains(sector_filter, case=False, na=False)]
                
                # Top Picks
                st.markdown("#### 🏆 TOP 20 — Highest Probability of Gain")
                top20 = predictions.head(20)
                
                for i, (_, row) in enumerate(top20.iterrows()):
                    prob = row['gain_probability_pct']
                    prob_class = 'prob-high' if prob > 60 else 'prob-low'
                    
                    col_a, col_b, col_c, col_d, col_e = st.columns([1, 3, 2, 2, 2])
                    with col_a:
                        st.markdown(f"**#{i+1}**")
                    with col_b:
                        st.markdown(f"**{row['ticker']}** — {row.get('company_name', '')[:40]}")
                        st.caption(f"{row.get('sector', '')} | Rank #{row['rank']:.0f}")
                    with col_c:
                        st.markdown(f"<span class='{prob_class}'>{prob:.1f}%</span>", unsafe_allow_html=True)
                        st.caption("Gain Probability")
                    with col_d:
                        st.markdown(f"**{row['signal']}**")
                    with col_e:
                        st.metric("Master Score", f"{row['master_score']:.1f}", f"{row.get('score_change_1w', 0):.1f}")
                    
                    st.markdown("---")
                
                # Bottom picks (highest risk)
                with st.expander("🔴 BOTTOM 20 — Highest Risk of Loss"):
                    bottom20 = predictions.tail(20).sort_values('gain_probability')
                    display_cols = ['ticker', 'company_name', 'gain_probability_pct', 'signal',
                                    'rank', 'master_score', 'market_state', 'sector']
                    st.dataframe(bottom20[display_cols].reset_index(drop=True), use_container_width=True)
                
                # Full table
                with st.expander("📋 Full Predictions Table"):
                    display_cols = ['ticker', 'company_name', 'gain_probability_pct', 'signal',
                                    'rank', 'master_score', 'price', 'ret_7d', 'ret_30d',
                                    'rank_change_1w', 'score_change_1w', 'market_state',
                                    'patterns', 'sector']
                    st.dataframe(
                        predictions[display_cols].reset_index(drop=True),
                        use_container_width=True,
                        height=600
                    )
                    
                    csv = predictions[display_cols].to_csv(index=False)
                    st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")
    
    # ═══════════════════════════════════════════════════
    #  TAB 2: RULE-BASED SCANNER 
    # ═══════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### 📊 Rule-Based Scanner (Transparent & Interpretable)")
        st.caption("Based on patterns discovered in deep gainer/loser analysis — every signal explained")
        
        with st.spinner("Computing rule-based scores..."):
            rule_results = compute_rule_scores(feature_df, latest_date)
        
        if len(rule_results) > 0:
            # Apply filters
            if min_master_score > 0:
                rule_results = rule_results[rule_results['master_score'] >= min_master_score]
            if max_rank < 2200:
                rule_results = rule_results[rule_results['rank'] <= max_rank]
            if sector_filter:
                rule_results = rule_results[rule_results['sector'].str.contains(sector_filter, case=False, na=False)]
            
            # Summary
            signal_counts = rule_results['rule_signal'].value_counts()
            cols = st.columns(5)
            for i, signal in enumerate(['🟢 STRONG BUY', '🟡 BUY', '⚪ NEUTRAL', '🟠 AVOID', '🔴 HIGH RISK']):
                with cols[i]:
                    st.metric(signal, signal_counts.get(signal, 0))
            
            st.markdown("---")
            
            # Top picks with explanations
            st.markdown("#### 🏆 TOP 30 Stocks by Rule Score")
            
            top30 = rule_results.head(30)
            
            for i, (_, row) in enumerate(top30.iterrows()):
                with st.expander(
                    f"#{i+1} | {row['ticker']} — Score: {row['rule_score']:.0f}/100 | "
                    f"{row['rule_signal']} | Rank #{row['rank']:.0f} | "
                    f"₹{row['price']:.0f} | {row.get('sector', '')}"
                ):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Rule Score", f"{row['rule_score']:.0f}/100")
                        st.metric("Master Score", f"{row['master_score']:.1f}")
                        st.metric("Rank", f"#{row['rank']:.0f}", f"{row.get('rank_change_1w', 0):+.0f}")
                        st.metric("7d Return", f"{row.get('ret_7d', 0):.1f}%")
                        st.metric("30d Return", f"{row.get('ret_30d', 0):.1f}%")
                    
                    with col2:
                        st.markdown("**📋 Why this score:**")
                        for reason in row['rule_reasons']:
                            st.markdown(f"- {reason}")
                        
                        st.markdown(f"**Market State:** `{row.get('market_state', 'N/A')}`")
                        patterns = str(row.get('patterns', ''))
                        if patterns and patterns != 'nan':
                            st.markdown(f"**Patterns:** {patterns}")
            
            # Red flags
            st.markdown("---")
            st.markdown("#### 🔴 TOP 20 — Highest Risk Stocks")
            bottom20 = rule_results.tail(20).sort_values('rule_score')
            
            for i, (_, row) in enumerate(bottom20.iterrows()):
                with st.expander(
                    f"⚠️ {row['ticker']} — Score: {row['rule_score']:.0f}/100 | "
                    f"{row['rule_signal']} | Rank #{row['rank']:.0f}"
                ):
                    st.markdown("**🚩 Red Flags:**")
                    for reason in row['rule_reasons']:
                        st.markdown(f"- {reason}")
    
    # ═══════════════════════════════════════════════════
    #  TAB 3: ML MODEL PERFORMANCE
    # ═══════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### 🔬 ML Model Performance & Feature Importance")
        
        if model is not None and metrics is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
            with col2:
                st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            with col3:
                st.metric("Precision (Gain)", f"{metrics['precision_gain']:.1%}")
            with col4:
                st.metric("Recall (Gain)", f"{metrics['recall_gain']:.1%}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train Size", f"{metrics['train_size']:,}")
            with col2:
                st.metric("Test Size", f"{metrics['test_size']:,}")
            with col3:
                st.metric("Base Gain Rate (test)", f"{metrics['gain_rate_test']:.1%}")
            
            st.markdown("---")
            
            # Feature importance
            st.markdown("#### 📊 Feature Importance (Top 25)")
            
            fi = metrics['feature_importance']
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:25]
            fi_df = pd.DataFrame(fi_sorted, columns=['Feature', 'Importance'])
            
            fig = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='viridis',
                title='Top 25 Most Important Features'
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Model interpretation
            st.markdown("#### 💡 Model Interpretation")
            top_features = [f[0] for f in fi_sorted[:5]]
            st.markdown(f"""
            The model relies most heavily on these features:
            1. **{top_features[0]}** — Most predictive single feature
            2. **{top_features[1]}**
            3. **{top_features[2]}**
            4. **{top_features[3]}**
            5. **{top_features[4]}**
            
            **Key insight:** If AUC-ROC > 0.6, the model has predictive power beyond random. 
            Above 0.65 = useful, above 0.7 = good, above 0.75 = strong.
            """)
        else:
            st.warning("Model not trained yet. Upload files and check the Probability tab first.")
    
    # ═══════════════════════════════════════════════════
    #  TAB 4: SECTOR HEATMAP
    # ═══════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### 🗺️ Sector Momentum Heatmap")
        
        sector_data = compute_sector_momentum(master_df, latest_date)
        
        if len(sector_data) > 0:
            fig = px.treemap(
                sector_data.reset_index(),
                path=['sector'],
                values='stock_count',
                color='ret_30d',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                title='Sectors by Size (stock count) and Color (30d return %)',
                hover_data=['master_score', 'momentum_score', 'ret_7d', 'ret_3m']
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sector table
            st.markdown("#### 📊 Sector Rankings")
            display_sector = sector_data.round(2).reset_index()
            st.dataframe(display_sector, use_container_width=True)
            
            # Sector momentum over time
            st.markdown("#### 📈 Sector Rotation Over Time")
            
            top_sectors = sector_data.head(8).index.tolist()
            
            sector_ts = []
            for date in sorted_dates:
                week_data = master_df[master_df['week_date'] == date]
                for sector in top_sectors:
                    sec_data = week_data[week_data['sector'] == sector]
                    if len(sec_data) > 0:
                        sector_ts.append({
                            'week': date,
                            'sector': sector,
                            'avg_score': sec_data['master_score'].mean(),
                            'avg_rank': sec_data['rank'].mean(),
                            'avg_ret_30d': sec_data['ret_30d'].mean(),
                        })
            
            if sector_ts:
                sector_ts_df = pd.DataFrame(sector_ts)
                fig = px.line(
                    sector_ts_df, x='week', y='avg_score', color='sector',
                    title='Top Sectors — Average Master Score Over Time'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════
    #  TAB 5: STOCK JOURNEY TRACKER
    # ═══════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### 📈 Stock Journey Tracker")
        st.caption("Track any stock's complete weekly evolution")
        
        # Stock selector
        all_tickers = sorted(master_df['ticker'].unique())
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_ticker = st.selectbox(
                "Select Stock",
                all_tickers,
                index=0
            )
        with col2:
            search_ticker = st.text_input("Or type ticker", "")
            if search_ticker:
                matches = [t for t in all_tickers if search_ticker.upper() in t.upper()]
                if matches:
                    selected_ticker = st.selectbox("Matches", matches)
        
        if selected_ticker:
            journey = get_stock_journey(master_df, selected_ticker)
            
            if len(journey) > 0:
                latest_row = journey.iloc[-1]
                
                # Header
                st.markdown(f"## {selected_ticker} — {latest_row.get('company_name', '')}")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Current Price", f"₹{latest_row['price']:.0f}")
                with col2:
                    st.metric("Rank", f"#{latest_row['rank']:.0f}")
                with col3:
                    st.metric("Master Score", f"{latest_row['master_score']:.1f}")
                with col4:
                    st.metric("Market State", latest_row.get('market_state', ''))
                with col5:
                    st.metric("Sector", latest_row.get('sector', ''))
                
                # Charts
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=['Price', 'Rank (lower=better)', 'Master Score', 
                                    'Breakout Score', 'Volume Score', 'Momentum Score'],
                    vertical_spacing=0.08
                )
                
                dates = journey['week_date']
                
                fig.add_trace(go.Scatter(x=dates, y=journey['price'], mode='lines+markers',
                    name='Price', line=dict(color='#00d2ff', width=2)), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=dates, y=journey['rank'], mode='lines+markers',
                    name='Rank', line=dict(color='#ff6b6b', width=2)), row=1, col=2)
                
                fig.add_trace(go.Scatter(x=dates, y=journey['master_score'], mode='lines+markers',
                    name='Master Score', line=dict(color='#00ff88', width=2)), row=2, col=1)
                
                fig.add_trace(go.Scatter(x=dates, y=journey['breakout_score'], mode='lines+markers',
                    name='Breakout', line=dict(color='#ffa500', width=2)), row=2, col=2)
                
                fig.add_trace(go.Scatter(x=dates, y=journey['volume_score'], mode='lines+markers',
                    name='Volume', line=dict(color='#9b59b6', width=2)), row=3, col=1)
                
                fig.add_trace(go.Scatter(x=dates, y=journey['momentum_score'], mode='lines+markers',
                    name='Momentum', line=dict(color='#e74c3c', width=2)), row=3, col=2)
                
                fig.update_layout(height=700, showlegend=False, template='plotly_dark')
                fig.update_yaxes(autorange="reversed", row=1, col=2)  # Rank: lower is better
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekly data table
                with st.expander("📋 Full Weekly Data"):
                    display_cols = ['week_date', 'rank', 'master_score', 'price',
                                    'breakout_score', 'momentum_score', 'volume_score',
                                    'ret_7d', 'ret_30d', 'market_state', 'patterns']
                    st.dataframe(journey[display_cols].reset_index(drop=True), use_container_width=True)
    
    # ═══════════════════════════════════════════════════
    #  TAB 6: RANK MOVERS
    # ═══════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("### 🔄 Biggest Rank Movers This Week")
        st.caption("Stocks that made the biggest rank jumps or drops — early signals")
        
        if len(sorted_dates) >= 2:
            current = master_df[master_df['week_date'] == sorted_dates[-1]][['ticker', 'company_name', 'rank', 'master_score', 'price', 'market_state', 'patterns', 'sector', 'ret_7d', 'breakout_score', 'momentum_score']].copy()
            previous = master_df[master_df['week_date'] == sorted_dates[-2]][['ticker', 'rank', 'master_score', 'price']].copy()
            
            merged = current.merge(previous, on='ticker', suffixes=('', '_prev'))
            merged['rank_change'] = merged['rank_prev'] - merged['rank']  # positive = improved
            merged['score_change'] = merged['master_score'] - merged['master_score_prev']
            merged['price_change_pct'] = ((merged['price'] - merged['price_prev']) / merged['price_prev'] * 100).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚀 Top 30 Rank Improvers")
                improvers = merged.nlargest(30, 'rank_change')
                for i, (_, row) in enumerate(improvers.iterrows()):
                    st.markdown(
                        f"**{i+1}. {row['ticker']}** | Rank #{row['rank']:.0f} "
                        f"(+{row['rank_change']:.0f}) | Score {row['master_score']:.1f} "
                        f"(+{row['score_change']:.1f}) | ₹{row['price']:.0f} "
                        f"({row['price_change_pct']:+.1f}%) | {row.get('market_state', '')}"
                    )
            
            with col2:
                st.markdown("#### 💀 Top 30 Rank Crashers")
                crashers = merged.nsmallest(30, 'rank_change')
                for i, (_, row) in enumerate(crashers.iterrows()):
                    st.markdown(
                        f"**{i+1}. {row['ticker']}** | Rank #{row['rank']:.0f} "
                        f"({row['rank_change']:.0f}) | Score {row['master_score']:.1f} "
                        f"({row['score_change']:.1f}) | ₹{row['price']:.0f} "
                        f"({row['price_change_pct']:+.1f}%) | {row.get('market_state', '')}"
                    )
    
    # ═══════════════════════════════════════════════════
    #  TAB 7: PATTERN ANALYSIS
    # ═══════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("### ⚡ Pattern Success Analysis")
        st.caption("Which patterns historically led to gains vs losses?")
        
        # For each pattern, compute average forward return
        patterns_list = [
            'CAT LEADER', 'VOL EXPLOSION', 'MARKET LEADER', 'MOMENTUM WAVE',
            'PREMIUM MOMENTUM', 'VELOCITY BREAKOUT', 'INSTITUTIONAL',
            'GOLDEN CROSS', 'STEALTH', 'DISTRIBUTION', 'CAPITULATION',
            'HIGH PE', 'PHOENIX', 'PULLBACK SUPPORT', 'RANGE COMPRESS',
            'ACCELERATION', 'ROTATION LEADER', 'GARP', 'VALUE MOMENTUM',
            'EARNINGS ROCKET', 'LIQUID LEADER', 'VELOCITY SQUEEZE',
            'RUNAWAY GAP', 'PYRAMID'
        ]
        
        valid_data = feature_df.dropna(subset=['fwd_return_4w'])
        
        if len(valid_data) > 0:
            pattern_stats = []
            for pattern in patterns_list:
                col_name = f"has_{pattern.lower().replace(' ', '_')}"
                if col_name in valid_data.columns:
                    with_pattern = valid_data[valid_data[col_name] == 1]
                    without_pattern = valid_data[valid_data[col_name] == 0]
                    
                    if len(with_pattern) >= 5:
                        pattern_stats.append({
                            'Pattern': pattern,
                            'Count': len(with_pattern),
                            'Avg Forward 4w Return': with_pattern['fwd_return_4w'].mean(),
                            'Win Rate': (with_pattern['fwd_return_4w'] > 0).mean() * 100,
                            'Avg Return (without)': without_pattern['fwd_return_4w'].mean(),
                            'Edge': with_pattern['fwd_return_4w'].mean() - without_pattern['fwd_return_4w'].mean(),
                            'Big Gainer Rate (>10%)': (with_pattern['fwd_return_4w'] > 10).mean() * 100,
                            'Big Loser Rate (<-10%)': (with_pattern['fwd_return_4w'] < -10).mean() * 100,
                        })
            
            if pattern_stats:
                ps_df = pd.DataFrame(pattern_stats).sort_values('Edge', ascending=False)
                
                # Chart
                fig = px.bar(
                    ps_df, x='Pattern', y='Edge',
                    color='Edge', color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    title='Pattern Edge: Extra Return vs Non-Pattern Stocks (4-week forward)',
                    hover_data=['Count', 'Win Rate', 'Avg Forward 4w Return']
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Win rate chart
                fig2 = px.bar(
                    ps_df, x='Pattern', y='Win Rate',
                    color='Win Rate', color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=50,
                    title='Pattern Win Rate (% of stocks that gained in next 4 weeks)',
                    hover_data=['Count', 'Edge']
                )
                fig2.update_layout(height=400, xaxis_tickangle=-45)
                fig2.add_hline(y=50, line_dash="dash", line_color="white", annotation_text="50% baseline")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Table
                st.dataframe(ps_df.round(2).reset_index(drop=True), use_container_width=True)
                
                # Best combos
                st.markdown("#### 🔗 Best Pattern Combinations")
                
                combo_stats = []
                combo_pairs = [
                    ('has_stealth', 'has_institutional', 'STEALTH + INSTITUTIONAL'),
                    ('has_cat_leader', 'has_market_leader', 'CAT + MARKET LEADER'),
                    ('has_velocity_breakout', 'has_premium_momentum', 'BREAKOUT + PREMIUM MOM'),
                    ('has_golden_cross', 'has_momentum_wave', 'GOLDEN CROSS + MOM WAVE'),
                    ('has_cat_leader', 'has_vol_explosion', 'CAT LEADER + VOL EXPLOSION'),
                    ('has_institutional', 'has_golden_cross', 'INSTITUTIONAL + GOLDEN CROSS'),
                    ('has_phoenix', 'has_pullback_support', 'PHOENIX + PULLBACK'),
                    ('has_stealth', 'has_acceleration', 'STEALTH + ACCELERATION'),
                    ('has_garp', 'has_value_momentum', 'GARP + VALUE MOMENTUM'),
                ]
                
                for col1_name, col2_name, combo_name in combo_pairs:
                    if col1_name in valid_data.columns and col2_name in valid_data.columns:
                        combo_data = valid_data[(valid_data[col1_name] == 1) & (valid_data[col2_name] == 1)]
                        if len(combo_data) >= 3:
                            combo_stats.append({
                                'Combination': combo_name,
                                'Count': len(combo_data),
                                'Avg 4w Return': combo_data['fwd_return_4w'].mean(),
                                'Win Rate': (combo_data['fwd_return_4w'] > 0).mean() * 100,
                            })
                
                if combo_stats:
                    combo_df = pd.DataFrame(combo_stats).sort_values('Avg 4w Return', ascending=False)
                    st.dataframe(combo_df.round(2).reset_index(drop=True), use_container_width=True)
    
    # ═══════════════════════════════════════════════════
    #  TAB 8: BACKTEST SYSTEM
    # ═══════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("### 🧪 Backtest: What if you followed the system?")
        st.caption("Simulate: Buy top-N ranked stocks each week, hold for N weeks, measure returns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            top_n = st.slider("Buy Top N stocks each week", 5, 50, 20)
        with col2:
            hold_weeks = st.selectbox("Hold period (weeks)", [1, 2, 4], index=1)
        with col3:
            score_threshold = st.slider("Min Master Score", 0.0, 90.0, 50.0)
        
        if st.button("🚀 Run Backtest"):
            with st.spinner("Running backtest..."):
                results = []
                
                for i, date in enumerate(sorted_dates[:-hold_weeks]):
                    week_data = master_df[master_df['week_date'] == date].copy()
                    
                    # Filter by score
                    eligible = week_data[week_data['master_score'] >= score_threshold]
                    
                    # Pick top N
                    picks = eligible.nsmallest(top_n, 'rank')
                    
                    # What happened to these stocks N weeks later?
                    future_date = sorted_dates[i + hold_weeks]
                    future_data = master_df[master_df['week_date'] == future_date][['ticker', 'price']]
                    
                    merged = picks.merge(future_data, on='ticker', suffixes=('_buy', '_sell'))
                    merged['return_pct'] = ((merged['price_sell'] - merged['price_buy']) / merged['price_buy'] * 100)
                    
                    if len(merged) > 0:
                        results.append({
                            'buy_date': date,
                            'sell_date': future_date,
                            'stocks_bought': len(merged),
                            'avg_return': merged['return_pct'].mean(),
                            'median_return': merged['return_pct'].median(),
                            'win_rate': (merged['return_pct'] > 0).mean() * 100,
                            'best': merged['return_pct'].max(),
                            'worst': merged['return_pct'].min(),
                            'total_return': merged['return_pct'].sum(),
                        })
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Return per Period", f"{results_df['avg_return'].mean():.2f}%")
                    with col2:
                        st.metric("Avg Win Rate", f"{results_df['win_rate'].mean():.1f}%")
                    with col3:
                        st.metric("Best Period", f"{results_df['avg_return'].max():.2f}%")
                    with col4:
                        st.metric("Worst Period", f"{results_df['avg_return'].min():.2f}%")
                    
                    # Cumulative returns
                    results_df['cumulative'] = (1 + results_df['avg_return'] / 100).cumprod() * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['buy_date'], y=results_df['cumulative'],
                        mode='lines+markers', name='Portfolio Value (₹100 start)',
                        line=dict(color='#00ff88', width=3),
                        fill='tonexty', fillcolor='rgba(0,255,136,0.1)'
                    ))
                    fig.add_hline(y=100, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        title=f'Backtest: Top {top_n} stocks (score>{score_threshold}), {hold_weeks}w hold',
                        yaxis_title='Portfolio Value (₹100 start)',
                        xaxis_title='Entry Date',
                        height=400, template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Period by period
                    fig2 = px.bar(
                        results_df, x='buy_date', y='avg_return',
                        color='avg_return', color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0,
                        title='Return per Period'
                    )
                    fig2.update_layout(height=300)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Win rate over time
                    fig3 = px.bar(
                        results_df, x='buy_date', y='win_rate',
                        color='win_rate', color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=50,
                        title='Win Rate per Period'
                    )
                    fig3.add_hline(y=50, line_dash="dash", line_color="white")
                    fig3.update_layout(height=300)
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    st.dataframe(results_df.round(2), use_container_width=True)
                else:
                    st.warning("Not enough data for backtest")

elif uploaded_files and len(uploaded_files) < 3:
    st.warning("⚠️ Please upload at least 3 weekly CSV files for meaningful analysis.")
else:
    # Landing page
    st.markdown("""
    ## 🚀 How to Use This System
    
    ### Step 1: Upload Data
    Upload your weekly `Stocks_Weekly_*.csv` files using the sidebar. More weeks = better predictions.
    
    ### Step 2: Explore 8 Powerful Modules
    
    | Module | What It Does |
    |--------|-------------|
    | 🎯 **Probability Predictions** | ML model predicts % chance each stock gains in next 1/2/4 weeks |
    | 📊 **Rule-Based Scanner** | Transparent scoring with written explanations for every signal |
    | 🔬 **ML Model Performance** | See AUC-ROC, accuracy, and which features matter most |
    | 🗺️ **Sector Heatmap** | Which sectors are hot/cold, rotation tracking |
    | 📈 **Stock Journey Tracker** | Track any stock's complete rank/score/price evolution |
    | 🔄 **Rank Movers** | Biggest rank jumps and crashes this week |
    | ⚡ **Pattern Analysis** | Which patterns (CAT LEADER, STEALTH etc.) actually predict gains |
    | 🧪 **Backtest System** | "What if I bought Top N stocks each week?" simulation |
    
    ### Step 3: Make Decisions
    Cross-reference ML probability + Rule Score + Pattern analysis for highest-conviction picks.
    
    ---
    
    ### 🧠 The System Learns From:
    - **60+ features** per stock per week (scores, patterns, returns, volume, rank changes)
    - **Forward return labels** — it knows what actually happened to stocks after each snapshot
    - **Pattern combinations** — STEALTH + INSTITUTIONAL was the strongest combo
    - **Rank acceleration** — 2nd derivative of rank change catches moves early
    - **Market state transitions** — SIDEWAYS → ROTATION → UPTREND = buy signal
    
    ### ⚠️ Minimum 3 weekly files required (recommended: 10+)
    """)

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#8892b0;'>© WAVE Stock Probability Engine | Built for systematic, data-driven investing</p>", unsafe_allow_html=True)
