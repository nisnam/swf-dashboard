"""
SIIP Dashboard — Data Loader
==============================
Cached loading of data and pre-computed model artifacts
"""

import json
import pickle
import pandas as pd
import streamlit as st
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"


@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    df = pd.read_csv(DATA_DIR / "SIIP_cleaned.csv")
    # Impute NaNs same as precompute
    df['esg_composite'] = df['esg_composite'].fillna(df['esg_composite'].median())
    df['esg_x_competition'] = df['esg_x_competition'].fillna(df['esg_x_competition'].median())
    return df


@st.cache_data
def load_regression_results():
    with open(MODELS_DIR / "regression_results.json") as f:
        return json.load(f)


@st.cache_data
def load_classification_results():
    with open(MODELS_DIR / "classification_results.json") as f:
        return json.load(f)


@st.cache_data
def load_classification_models():
    with open(MODELS_DIR / "classification_models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_shap_values():
    with open(MODELS_DIR / "shap_values.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_decision_tree():
    with open(MODELS_DIR / "decision_tree.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_clustering_results():
    with open(MODELS_DIR / "clustering_results.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_pca_results():
    with open(MODELS_DIR / "pca_results.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_anomaly_results():
    with open(MODELS_DIR / "anomaly_results.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_arm_rules():
    with open(MODELS_DIR / "arm_rules.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_arima_results():
    with open(MODELS_DIR / "arima_results.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_frameworks():
    with open(MODELS_DIR / "frameworks.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_interpretations():
    with open(MODELS_DIR / "interpretations.json") as f:
        return json.load(f)


@st.cache_data
def load_prep_artifacts():
    with open(MODELS_DIR / "prep_artifacts.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_scaler():
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_threshold_analysis():
    with open(MODELS_DIR / "threshold_analysis.json") as f:
        return json.load(f)


@st.cache_data
def load_sentiment_results():
    with open(MODELS_DIR / "sentiment_results.pkl", "rb") as f:
        return pickle.load(f)


# Feature name mappings for display
FEATURE_DISPLAY_NAMES = {
    'funding_stage_num': 'Funding Stage',
    'revenue_growth_pct': 'Revenue Growth (%)',
    'gross_margin_pct': 'Gross Margin (%)',
    'company_age': 'Company Age (yrs)',
    'country_risk_composite': 'Country Risk',
    'bilateral_composite': 'Bilateral Alignment',
    'competitive_intensity': 'Competition',
    'team_size': 'Team Size',
    'patent_count': 'Patents',
    'ip_protection_score': 'IP Protection',
    'tech_transfer_proxy': 'Tech Transfer',
    'esg_composite': 'ESG Score',
    'national_strategy_alignment': 'Strategy Alignment',
    'is_pre_revenue': 'Pre-Revenue',
    'stability_x_stage': 'Stability x Stage',
    'runway_months': 'Runway (months)',
    'regulatory_moat': 'Regulatory Moat',
    'total_capital_raised': 'Capital Raised',
    'last_valuation': 'Last Valuation',
    'burn_rate_monthly': 'Monthly Burn',
    'outcome_binary': 'Success (Binary)',
    'outcome_numeric': 'Outcome (0-3)',
}


def display_name(feat):
    """Get display name for a feature"""
    return FEATURE_DISPLAY_NAMES.get(feat, feat.replace('_', ' ').title())


# SWF Priority dimension groupings
SWF_DIMENSIONS = {
    'Financial Return': ['funding_stage_num', 'revenue_growth_pct', 'gross_margin_pct', 'runway_months'],
    'Strategic Value': ['patent_count', 'ip_protection_score', 'tech_transfer_proxy', 'competitive_intensity', 'regulatory_moat'],
    'Geopolitical Safety': ['country_risk_composite', 'bilateral_composite', 'stability_x_stage'],
    'Domestic Impact': ['national_strategy_alignment', 'team_size', 'is_pre_revenue'],
    'ESG': ['esg_composite', 'company_age']
}

# ── 3 SWF Mandates ─────────────────────────────────────────────────────────
MANDATES = {
    'GIC — Financial Return': {
        'fund': 'GIC (Singapore)',
        'icon': '🇸🇬',
        'weights': {
            'Financial Return': 0.35,
            'Strategic Value': 0.20,
            'Geopolitical Safety': 0.20,
            'Domestic Impact': 0.10,
            'ESG': 0.15,
        },
        'lens': 'Will this make money?',
        'description': 'Pure financial return maximization. Prioritizes revenue growth, margins, and exit potential.',
        'color': '#3498DB',
        'priority_features': ['revenue_growth_pct', 'gross_margin_pct', 'funding_stage_num', 'runway_months'],
    },
    'PIF — Strategic Alignment': {
        'fund': 'PIF (Saudi Arabia)',
        'icon': '🇸🇦',
        'weights': {
            'Financial Return': 0.15,
            'Strategic Value': 0.30,
            'Geopolitical Safety': 0.15,
            'Domestic Impact': 0.25,
            'ESG': 0.15,
        },
        'lens': 'Does this serve Vision 2030?',
        'description': 'Strategic national alignment. Prioritizes tech transfer, domestic impact, and bilateral ties.',
        'color': '#2ECC71',
        'priority_features': ['national_strategy_alignment', 'tech_transfer_proxy', 'team_size', 'bilateral_composite'],
    },
    'Mubadala — Balanced Portfolio': {
        'fund': 'Mubadala (UAE)',
        'icon': '🇦🇪',
        'weights': {
            'Financial Return': 0.20,
            'Strategic Value': 0.20,
            'Geopolitical Safety': 0.20,
            'Domestic Impact': 0.20,
            'ESG': 0.20,
        },
        'lens': 'Best risk-adjusted opportunity?',
        'description': 'Balanced across all dimensions. Equal weight to financial, strategic, geopolitical, domestic, and ESG.',
        'color': '#D4AF37',
        'priority_features': ['esg_composite', 'country_risk_composite', 'revenue_growth_pct', 'national_strategy_alignment'],
    },
}

DIMENSION_ICONS = {
    'Financial Return': '💰',
    'Strategic Value': '🎯',
    'Geopolitical Safety': '🌍',
    'Domestic Impact': '🏗️',
    'ESG': '🌱',
}


def compute_dimension_scores(row, df_stats=None):
    """
    Compute 0-100 scores for each of the 5 SWF dimensions for a venture row.
    Uses percentile ranking within the dataset.
    df_stats should be a dict of {feature: (min, max)} — if None, returns raw means.
    """
    scores = {}
    for dim, features in SWF_DIMENSIONS.items():
        dim_vals = []
        for f in features:
            if f in row.index:
                val = float(row[f])
                if df_stats and f in df_stats:
                    fmin, fmax = df_stats[f]
                    rng = fmax - fmin if fmax != fmin else 1.0
                    # For country_risk and competitive_intensity, higher = worse
                    if f in ['country_risk_composite', 'competitive_intensity', 'is_pre_revenue']:
                        norm = 1.0 - (val - fmin) / rng
                    else:
                        norm = (val - fmin) / rng
                    dim_vals.append(max(0, min(1, norm)))
                else:
                    dim_vals.append(val)
        scores[dim] = round(float(pd.Series(dim_vals).mean()) * 100, 1) if dim_vals else 50.0
    return scores


def compute_mandate_score(row, mandate_name, df_stats=None):
    """
    Compute a weighted 0-100 mandate score for a venture.
    """
    mandate = MANDATES[mandate_name]
    dim_scores = compute_dimension_scores(row, df_stats)
    weighted = sum(dim_scores[dim] * mandate['weights'][dim] for dim in dim_scores)
    return round(weighted, 1)


def get_df_stats(df):
    """Pre-compute min/max for each feature for normalization."""
    stats = {}
    for dim, features in SWF_DIMENSIONS.items():
        for f in features:
            if f in df.columns:
                stats[f] = (float(df[f].min()), float(df[f].max()))
    return stats


def get_mandate_insight(page, mandate_name):
    """Return mandate-specific contextual insight for each page."""
    mandate = MANDATES[mandate_name]
    fund = mandate['fund']

    insights = {
        'pipeline': {
            'GIC — Financial Return': f"<strong>{fund} Lens:</strong> Focus on sectors with highest exit multiples and revenue growth. Later-stage ventures de-risk returns but compress upside.",
            'PIF — Strategic Alignment': f"<strong>{fund} Lens:</strong> Evaluate pipeline for Vision 2030 alignment — prioritize Energy & Clean Tech, Digital Economy, and ventures with strong domestic job creation potential.",
            'Mubadala — Balanced Portfolio': f"<strong>{fund} Lens:</strong> Assess portfolio balance across all five dimensions. Flag concentrations in any single sector, geography, or risk profile.",
        },
        'risk': {
            'GIC — Financial Return': f"<strong>{fund} Lens:</strong> Which financial features (margins, growth, stage) carry the strongest linear signal for returns?",
            'PIF — Strategic Alignment': f"<strong>{fund} Lens:</strong> Do strategic alignment features (tech transfer, bilateral ties) show linear relationships with outcomes?",
            'Mubadala — Balanced Portfolio': f"<strong>{fund} Lens:</strong> Are any of the 5 SWF dimensions disproportionately represented in the regression signal?",
        },
        'prediction': {
            'GIC — Financial Return': f"<strong>{fund} Lens:</strong> Optimize the screening threshold for maximum financial return — a missed winner costs more than extra due diligence.",
            'PIF — Strategic Alignment': f"<strong>{fund} Lens:</strong> Prioritize recall over precision — strategic ventures that succeed create outsized national value even with lower probability.",
            'Mubadala — Balanced Portfolio': f"<strong>{fund} Lens:</strong> Use the balanced threshold that maximizes F1 — equalize the cost of false positives and false negatives across all dimensions.",
        },
        'segmentation': {
            'GIC — Financial Return': f"<strong>{fund} Lens:</strong> Which cluster archetypes have the highest exit rates? Over-allocate to proven financial performers.",
            'PIF — Strategic Alignment': f"<strong>{fund} Lens:</strong> Identify clusters with highest tech transfer and domestic impact scores — even if financial returns are moderate.",
            'Mubadala — Balanced Portfolio': f"<strong>{fund} Lens:</strong> Diversify across all archetypes proportional to their risk-adjusted composite scores.",
        },
        'patterns': {
            'GIC — Financial Return': f"<strong>{fund} Lens:</strong> What financial feature combinations (margins + stage + growth) most reliably predict successful exits?",
            'PIF — Strategic Alignment': f"<strong>{fund} Lens:</strong> Which rules involve strategic features (bilateral ties, tech transfer, national alignment)? These are your screening checklist.",
            'Mubadala — Balanced Portfolio': f"<strong>{fund} Lens:</strong> Look for rules that span multiple dimensions — cross-cutting patterns reveal the most robust opportunities.",
        },
        'deal': {
            'GIC — Financial Return': f"<strong>{fund} Lens:</strong> Score this deal primarily on financial metrics. How does it compare to the top financial performers in the portfolio?",
            'PIF — Strategic Alignment': f"<strong>{fund} Lens:</strong> Does this venture advance Vision 2030 priorities? Weight strategic and domestic impact dimensions heavily.",
            'Mubadala — Balanced Portfolio': f"<strong>{fund} Lens:</strong> Evaluate this venture holistically. Flag any dimension scoring below the portfolio average as a risk factor.",
        },
        'sentiment': {
            'GIC — Financial Return': f"<strong>{fund} Lens:</strong> Focus on financial sentiment signals — analyst language about margins, growth, and exit potential matters most.",
            'PIF — Strategic Alignment': f"<strong>{fund} Lens:</strong> Monitor sentiment around strategic themes — technology, bilateral relations, and domestic capability building language.",
            'Mubadala — Balanced Portfolio': f"<strong>{fund} Lens:</strong> Overall analyst sentiment provides a holistic check on quantitative scores. Flag where sentiment and models disagree.",
        },
    }
    return insights.get(page, {}).get(mandate_name, f"<strong>{fund}:</strong> {mandate['lens']}")
