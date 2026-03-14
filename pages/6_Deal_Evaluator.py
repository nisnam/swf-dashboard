"""
SIIP Dashboard — Page 6: Deal Evaluator
=========================================
Score a specific venture — combines all models into one recommendation.
Synthesis page bringing together classification, clustering, and similarity analysis.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from utils.data_loader import (
    load_data, load_classification_models, load_classification_results,
    load_scaler, load_clustering_results, load_frameworks,
    load_anomaly_results, load_prep_artifacts, load_interpretations,
    display_name, SWF_DIMENSIONS, MANDATES, get_mandate_insight,
    compute_mandate_score, compute_dimension_scores, get_df_stats, DIMENSION_ICONS,
)
from utils.styles import (
    inject_css, kpi_card, section_header, metric_highlight,
    swf_insight, mandate_badge, bridge_sentence, verdict_badge, verdict_card,
    format_currency, format_pct, COLORS, PLOTLY_COLORS,
)
from utils.charts import apply_theme, gauge_chart, radar_chart, bar_chart

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Deal Evaluator | SIIP", layout="wide")
st.html(inject_css())

# ── Mandate ──────────────────────────────────────────────────────────────────
mandate = st.session_state.get('mandate', 'GIC — Financial Return')
mandate_cfg = MANDATES[mandate]

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
clf_models = load_classification_models()
clf_results = load_classification_results()
scaler = load_scaler()
clustering = load_clustering_results()
frameworks = load_frameworks()
anomaly = load_anomaly_results()
prep = load_prep_artifacts()
interp = load_interpretations()

ivs = prep['ivs']  # 17 feature names in correct order
df_stats = get_df_stats(df)

st.title("Deal Evaluator")
st.markdown("*Score a specific venture — combines all models into one recommendation.*")
st.html(mandate_badge(mandate, mandate_cfg))
st.html(swf_insight(get_mandate_insight('deal', mandate)))

# ═════════════════════════════════════════════════════════════════════════════
# SECTION A: Venture Input
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("A. Venture Input"))

input_mode = st.radio(
    "Input Mode", ["Select from Dataset", "Manual Input"],
    horizontal=True, label_visibility="collapsed",
)

input_values = {}

if input_mode == "Select from Dataset":
    company_names = sorted(df['company_name'].dropna().unique().tolist())
    selected_company = st.selectbox("Select Company", company_names)
    row = df[df['company_name'] == selected_company].iloc[0]
    for feat in ivs:
        input_values[feat] = float(row[feat])

    # Show a summary of the selected venture
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.html(kpi_card("Company", selected_company))
    with col_info2:
        st.html(kpi_card("Sector", str(row.get('sector', 'N/A'))))
    with col_info3:
        st.html(kpi_card("Funding Stage", str(row.get('funding_stage', 'N/A'))))
    with col_info4:
        st.html(kpi_card("Country", str(row.get('country', 'N/A'))))

else:
    st.markdown("Adjust feature values below. Defaults are dataset medians.")
    for dimension, features in SWF_DIMENSIONS.items():
        dim_features = [f for f in features if f in ivs]
        if not dim_features:
            continue
        st.markdown(f"**{dimension}**")
        cols = st.columns(len(dim_features))
        for i, feat in enumerate(dim_features):
            with cols[i]:
                feat_min = float(df[feat].min())
                feat_max = float(df[feat].max())
                feat_median = float(df[feat].median())
                # Ensure min < max for slider
                if feat_min == feat_max:
                    feat_max = feat_min + 1.0
                step = (feat_max - feat_min) / 100.0
                input_values[feat] = st.slider(
                    display_name(feat),
                    min_value=feat_min,
                    max_value=feat_max,
                    value=feat_median,
                    step=step,
                    key=f"slider_{feat}",
                )

st.divider()

# ── Build input DataFrame in correct IV order ────────────────────────────────
input_row = pd.DataFrame([[input_values[f] for f in ivs]], columns=ivs)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION B: Intelligence Profile
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("B. Intelligence Profile"))

col_radar, col_cluster = st.columns([3, 2])

with col_radar:
    # Normalize input and dataset average to 0-1 using min-max from df
    radar_categories = [display_name(f) for f in ivs]
    input_norm = []
    avg_norm = []
    for feat in ivs:
        feat_min = float(df[feat].min())
        feat_max = float(df[feat].max())
        rng = feat_max - feat_min if feat_max != feat_min else 1.0
        input_norm.append((input_values[feat] - feat_min) / rng)
        avg_norm.append((float(df[feat].mean()) - feat_min) / rng)

    fig_radar = radar_chart(
        radar_categories,
        {"This Venture": input_norm, "Dataset Average": avg_norm},
        title="Feature Profile — Normalized (0-1)",
        height=480,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_cluster:
    # Scale input and find nearest K-Means centroid
    input_scaled = scaler.transform(input_row)
    best_k = str(clustering['best_k'])
    centroids = np.array(clustering['kmeans'][best_k]['centroids'])
    distances = [euclidean(input_scaled[0], c) for c in centroids]
    assigned_cluster = int(np.argmin(distances))

    cluster_personas = frameworks.get('cluster_personas', {})
    # Try int and string keys
    persona = cluster_personas.get(assigned_cluster, cluster_personas.get(str(assigned_cluster), {}))

    st.html(kpi_card("Assigned Cluster", f"Cluster {assigned_cluster}"))

    if persona:
        st.html(
            metric_highlight(f"<strong>{persona.get('name', 'Unknown')}</strong>"))
        st.markdown(f"**Description:** {persona.get('description', 'N/A')}")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            success_rate = persona.get('success_rate', 0)
            st.html(
                kpi_card("Cluster Success Rate", f"{success_rate:.1%}" if isinstance(success_rate, float) else str(success_rate)))
        with col_s2:
            st.html(
                kpi_card("Risk Level", persona.get('risk_level', 'N/A')))
        st.markdown(f"**Recommended Strategy:** {persona.get('strategy', 'N/A')}")
    else:
        st.info("No persona profile available for this cluster.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION C: Prediction
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("C. Prediction — Multi-Model Consensus"))

# Run all classification models on the scaled input
model_probs = {}
for model_name, model in clf_models.items():
    try:
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(input_scaled)[0][1]
        else:
            prob = float(model.predict(input_scaled)[0])
        model_probs[model_name] = float(prob)
    except Exception:
        pass

# Find best model by AUC-ROC from results
best_model_name = max(clf_results, key=lambda m: clf_results[m].get('auc_roc', 0))
best_prob = model_probs.get(best_model_name, 0.0)

# Verdict logic
if best_prob > 0.5:
    verdict = "INVEST"
elif best_prob > 0.3:
    verdict = "MONITOR"
else:
    verdict = "AVOID"

col_gauge, col_verdict = st.columns([2, 1])

with col_gauge:
    fig_gauge = gauge_chart(
        value=round(best_prob * 100, 1),
        title=f"Success Probability — {best_model_name}",
        max_val=100,
        height=300,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_verdict:
    st.html(verdict_badge(verdict))
    st.markdown("")
    st.html(
        metric_highlight(
            f"Best model: <strong>{best_model_name}</strong><br>"
            f"AUC-ROC: <strong>{clf_results[best_model_name]['auc_roc']:.4f}</strong><br>"
            f"Probability: <strong>{best_prob:.1%}</strong>"
        ))

# All models comparison bar chart
sorted_models = sorted(model_probs.items(), key=lambda x: x[1], reverse=True)
model_names_sorted = [m[0] for m in sorted_models]
model_probs_sorted = [round(m[1] * 100, 1) for m in sorted_models]

bar_colors = [
    COLORS['success'] if p > 50 else COLORS['warning'] if p > 30 else COLORS['danger']
    for p in model_probs_sorted
]

fig_models = bar_chart(
    x=model_names_sorted,
    y=model_probs_sorted,
    title="Success Probability by Model (%)",
    color=bar_colors,
    text=[f"{p:.1f}%" for p in model_probs_sorted],
    height=380,
)
st.plotly_chart(fig_models, use_container_width=True)

# ── Prescriptive: Verdict context ──
# Compute dimension scores for this venture
dim_scores = compute_dimension_scores(pd.Series(input_values), df_stats)
strong_dims = [d for d, s in dim_scores.items() if s >= 60]
weak_dims = [d for d, s in dim_scores.items() if s < 40]

if mandate == "GIC — Financial Return":
    verdict_context = (
        f"<b>GIC Assessment:</b> Success probability is <b>{best_prob:.1%}</b>. "
        + (f"Strong dimensions: <b>{', '.join(strong_dims)}</b>. " if strong_dims else "No dimensions score above 60. ")
        + (f"Weak dimensions: <b>{', '.join(weak_dims)}</b> — request mitigation plans for these. " if weak_dims else "")
        + f"<b>Action:</b> {'Proceed to IC review — financial metrics support investment.' if verdict == 'INVEST' else 'Place on watchlist — revisit when financials improve.' if verdict == 'MONITOR' else 'Pass — financial risk too high under GIC mandate.'}"
    )
elif mandate == "PIF — Strategic Alignment":
    strategic_score = dim_scores.get('Strategic Value', 50)
    domestic_score = dim_scores.get('Domestic Impact', 50)
    verdict_context = (
        f"<b>PIF Assessment:</b> Strategic Value score: <b>{strategic_score:.0f}/100</b>, "
        f"Domestic Impact: <b>{domestic_score:.0f}/100</b>. "
        f"<b>Action:</b> {'Strong strategic fit — advance for Vision 2030 alignment review.' if strategic_score > 60 else 'Strategic value below threshold — requires bilateral partner co-investment to justify.'}"
    )
else:
    avg_dim = sum(dim_scores.values()) / len(dim_scores)
    verdict_context = (
        f"<b>Mubadala Assessment:</b> Average dimension score: <b>{avg_dim:.0f}/100</b>. "
        + (f"Imbalanced: {', '.join(weak_dims)} below 40. " if weak_dims else "Well-balanced across all dimensions. ")
        + f"<b>Action:</b> {'Balanced profile supports investment. Proceed with standard diligence.' if verdict == 'INVEST' and not weak_dims else 'Address dimension imbalances before committing — request improvement plans for weak areas.' if weak_dims else 'Monitor and re-evaluate in 6 months.'}"
    )
st.html(swf_insight(verdict_context))

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION D: Similar Ventures
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("D. Similar Ventures — Nearest Neighbors"))

# Compute scaled features for entire dataset
df_features = df[ivs].copy()
df_scaled = scaler.transform(df_features)

# Euclidean distances from input to all rows
dists = np.linalg.norm(df_scaled - input_scaled[0], axis=1)
nearest_idx = np.argsort(dists)

# If we're selecting from dataset, skip the exact match (distance ~ 0)
neighbors = []
for idx in nearest_idx:
    if len(neighbors) >= 5:
        break
    if input_mode == "Select from Dataset" and dists[idx] < 1e-9:
        continue
    neighbors.append(idx)

# Get success probabilities for neighbors using the best model
neighbor_probs = []
for idx in neighbors:
    row_scaled = df_scaled[idx].reshape(1, -1)
    try:
        if hasattr(clf_models[best_model_name], 'predict_proba'):
            prob = clf_models[best_model_name].predict_proba(row_scaled)[0][1]
        else:
            prob = float(clf_models[best_model_name].predict(row_scaled)[0])
        neighbor_probs.append(f"{prob:.1%}")
    except Exception:
        neighbor_probs.append("N/A")

similar_df = pd.DataFrame({
    'Company': [df.iloc[i]['company_name'] for i in neighbors],
    'Sector': [df.iloc[i].get('sector', 'N/A') for i in neighbors],
    'Funding Stage': [df.iloc[i].get('funding_stage', 'N/A') for i in neighbors],
    'Investment Outcome': [df.iloc[i].get('investment_outcome', 'N/A') for i in neighbors],
    'Success Probability': neighbor_probs,
    'Distance': [f"{dists[i]:.2f}" for i in neighbors],
})

st.dataframe(similar_df, use_container_width=True, hide_index=True)

# ── Prescriptive: Similar ventures interpretation ──
if len(neighbors) > 0:
    outcomes_list = [df.iloc[i].get('investment_outcome', 'N/A') for i in neighbors]
    n_success_neighbors = sum(1 for o in outcomes_list if o == 'Successful Exit')
    n_total_neighbors = len(outcomes_list)
    neighbor_sr = n_success_neighbors / n_total_neighbors if n_total_neighbors > 0 else 0

    if neighbor_sr > 0.5:
        neighbor_signal = "strong historical support"
        neighbor_action = "Precedent is favorable — similar deals have mostly succeeded. Proceed with confidence."
    elif neighbor_sr > 0.2:
        neighbor_signal = "mixed historical precedent"
        neighbor_action = "Some similar deals succeeded, others failed. Dig into what differentiated the winners from losers."
    else:
        neighbor_signal = "weak historical precedent"
        neighbor_action = "Most similar ventures failed. Identify what makes this deal different — or walk away."

    st.html(swf_insight(
        f"<b>Historical Precedent:</b> Of the {n_total_neighbors} most similar past ventures, "
        f"<b>{n_success_neighbors} succeeded</b> ({neighbor_sr:.0%}) — {neighbor_signal}. "
        f"<b>Action:</b> {neighbor_action}"
    ))

deal_interp = interp.get('deal_evaluator', {})
if isinstance(deal_interp, dict):
    for key, text in deal_interp.items():
        st.caption(text)
elif isinstance(deal_interp, str):
    st.caption(deal_interp)
