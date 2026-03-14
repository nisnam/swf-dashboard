"""
SIIP Dashboard — Page 4: Segmentation Lab
==========================================
"What venture archetypes exist? How should we treat each?"
Clustering + PCA + Anomaly Detection
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    load_data, load_clustering_results, load_pca_results,
    load_anomaly_results, load_frameworks, load_interpretations,
    display_name, MANDATES, get_mandate_insight,
    compute_dimension_scores, get_df_stats, DIMENSION_ICONS,
)
from utils.styles import (
    inject_css, kpi_card, section_header, metric_highlight, COLORS, PLOTLY_COLORS,
    swf_insight, mandate_badge, bridge_sentence,
)
from utils.charts import apply_theme, scatter_chart, bar_chart, line_chart, radar_chart, heatmap_chart

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Segmentation Lab | SIIP", layout="wide")
st.html(inject_css())

# ── Mandate context ──────────────────────────────────────────────────────────
mandate = st.session_state.get('mandate', 'GIC — Financial Return')
mandate_config = MANDATES[mandate]

st.markdown("## Segmentation Lab")
st.html(
    '<span style="color:#95A5A6">Venture archetype discovery via clustering, '
    "dimensionality reduction, and anomaly detection</span>")

# Mandate badge and insight at the top
st.html(mandate_badge(mandate, mandate_config))
st.html(swf_insight(get_mandate_insight('segmentation', mandate)))

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
clust = load_clustering_results()
pca = load_pca_results()
anomaly = load_anomaly_results()
frameworks = load_frameworks()
interp = load_interpretations()

best_k = clust["best_k"]

# ─────────────────────────────────────────────────────────────────────────────
# Section A: Dimensionality Reduction (PCA)
# ─────────────────────────────────────────────────────────────────────────────
st.html(section_header("A — Dimensionality Reduction (PCA)"))

col_scree, col_loadings = st.columns([3, 2])

with col_scree:
    evr = pca["explained_variance_ratio"]
    cumvar = pca["cumulative_variance"]
    components = list(range(1, len(evr) + 1))

    fig_scree = go.Figure()
    fig_scree.add_trace(
        go.Bar(
            x=[f"PC{i}" for i in components],
            y=[v * 100 for v in evr],
            marker_color=COLORS["accent2"],
            name="Individual",
            hovertemplate="PC%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig_scree.add_trace(
        go.Scatter(
            x=[f"PC{i}" for i in components],
            y=[v * 100 for v in cumvar],
            mode="lines+markers",
            name="Cumulative",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=6),
            yaxis="y2",
        )
    )
    fig_scree.update_layout(
        title="Scree Plot — Explained Variance",
        yaxis=dict(title="Individual Variance (%)", side="left"),
        yaxis2=dict(
            title="Cumulative (%)",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            range=[0, 105],
        ),
        legend=dict(x=0.60, y=0.30),
    )
    apply_theme(fig_scree, height=400)
    st.plotly_chart(fig_scree, use_container_width=True)
    st.caption(
        f"PC1 explains {evr[0]*100:.1f}% of variance; "
        f"first 5 components capture {cumvar[min(4, len(cumvar)-1)]*100:.1f}% cumulatively."
    )

with col_loadings:
    loadings_2d = pca["loadings_2d"]
    feat_names = pca["feature_names"]

    # Build sorted loadings for PC1 and PC2
    rows = []
    for pc_label in ["PC1", "PC2"]:
        pc_dict = loadings_2d[pc_label]
        sorted_feats = sorted(pc_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for rank, (feat, val) in enumerate(sorted_feats, 1):
            rows.append(
                {"Component": pc_label, "Rank": rank, "Feature": display_name(feat), "Loading": round(val, 3)}
            )
    load_df = pd.DataFrame(rows)
    st.markdown("**Top-5 Feature Loadings per Component**")
    st.dataframe(load_df, use_container_width=True, hide_index=True, height=360)

if "pca" in interp.get("clustering", {}):
    st.caption(interp["clustering"]["pca"])

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section B: Optimal Clustering
# ─────────────────────────────────────────────────────────────────────────────
st.html(
    section_header("B — Optimal Clustering (K-Means / Hierarchical / DBSCAN)"))

col_elbow, col_sil = st.columns(2)

ks = list(range(2, 8))

with col_elbow:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(
        go.Scatter(
            x=ks,
            y=clust["inertias"],
            mode="lines+markers",
            line=dict(color=COLORS["accent2"], width=2),
            marker=dict(size=8),
        )
    )
    # Highlight best k
    best_idx = best_k - 2
    fig_elbow.add_trace(
        go.Scatter(
            x=[best_k],
            y=[clust["inertias"][best_idx]],
            mode="markers",
            marker=dict(color=COLORS["accent"], size=14, symbol="star"),
            name=f"Best k={best_k}",
        )
    )
    fig_elbow.update_layout(title="Elbow Plot — Inertia vs k", xaxis_title="k", yaxis_title="Inertia")
    apply_theme(fig_elbow, 370)
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.caption("The 'elbow' indicates where adding more clusters yields diminishing returns in inertia reduction.")

with col_sil:
    sil_colors = [COLORS["accent"] if k == best_k else COLORS["accent2"] for k in ks]
    fig_sil = go.Figure(
        data=[
            go.Bar(
                x=ks,
                y=clust["silhouettes"],
                marker_color=sil_colors,
                hovertemplate="k=%{x}: Silhouette=%{y:.3f}<extra></extra>",
            )
        ]
    )
    fig_sil.update_layout(title="Silhouette Score vs k", xaxis_title="k", yaxis_title="Silhouette Score")
    apply_theme(fig_sil, 370)
    st.plotly_chart(fig_sil, use_container_width=True)
    st.caption(f"Highest silhouette at k={best_k} ({clust['silhouettes'][best_idx]:.3f}), indicating best-separated clusters.")

# Slider for k
selected_k = st.slider("Select number of clusters (k)", min_value=2, max_value=7, value=best_k)

# Algorithm comparison table
st.markdown("**Algorithm Comparison — Silhouette Scores**")

comp_rows = []
# K-Means
km_sil = clust["kmeans"][str(selected_k)]["silhouette"]
comp_rows.append({"Algorithm": "K-Means", "k": selected_k, "Silhouette": round(km_sil, 4)})

# Hierarchical (available for k=3,4,5)
if str(selected_k) in clust["hierarchical"]:
    h_sil = clust["hierarchical"][str(selected_k)]["silhouette"]
    comp_rows.append({"Algorithm": "Hierarchical", "k": selected_k, "Silhouette": round(h_sil, 4)})
else:
    comp_rows.append({"Algorithm": "Hierarchical", "k": selected_k, "Silhouette": "N/A (only k=3-5)"})

# DBSCAN
comp_rows.append(
    {
        "Algorithm": "DBSCAN",
        "k": clust["dbscan"]["n_clusters"],
        "Silhouette": "—",
    }
)
st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

# DBSCAN stats
db = clust["dbscan"]
c1, c2, c3 = st.columns(3)
c1.metric("DBSCAN Clusters", db["n_clusters"])
c2.metric("DBSCAN Noise Points", f'{db["n_noise"]:,}')
c3.metric("Epsilon", f'{db["eps"]:.2f}')

if "overview" in interp.get("clustering", {}):
    st.caption(interp["clustering"]["overview"])

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section C: Cluster Map
# ─────────────────────────────────────────────────────────────────────────────
st.html(section_header("C — Cluster Map"))

# Prepare projections and labels
proj_2d = np.array(pca["projections_2d"])
km_labels = clust["kmeans"][str(selected_k)]["labels"]
anomaly_labels = anomaly["labels"]

map_df = pd.DataFrame(
    {
        "PC1": proj_2d[:, 0],
        "PC2": proj_2d[:, 1],
        "Cluster": [f"Cluster {l}" for l in km_labels],
        "Anomaly": ["Anomaly" if a == -1 else "Normal" for a in anomaly_labels],
    }
)

# Add venture details for hover
if "sector" in df.columns:
    map_df["Sector"] = df["sector"].values
if "funding_stage_num" in df.columns:
    map_df["Stage"] = df["funding_stage_num"].values

# Symbols: normal = circle, anomaly = x
map_df["Symbol"] = map_df["Anomaly"].map({"Normal": "circle", "Anomaly": "x"})

fig_map = go.Figure()

cluster_ids = sorted(map_df["Cluster"].unique())
for i, cid in enumerate(cluster_ids):
    for anomaly_status, symbol, sz in [("Normal", "circle", 5), ("Anomaly", "x", 9)]:
        mask = (map_df["Cluster"] == cid) & (map_df["Anomaly"] == anomaly_status)
        subset = map_df[mask]
        if subset.empty:
            continue
        hover_text = [
            f"Cluster: {cid}<br>Status: {anomaly_status}"
            + (f"<br>Sector: {row.get('Sector', 'N/A')}" if "Sector" in subset.columns else "")
            for _, row in subset.iterrows()
        ]
        fig_map.add_trace(
            go.Scatter(
                x=subset["PC1"],
                y=subset["PC2"],
                mode="markers",
                marker=dict(
                    color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)],
                    symbol=symbol,
                    size=sz,
                    opacity=0.6 if anomaly_status == "Normal" else 0.95,
                    line=dict(width=1, color="white") if anomaly_status == "Anomaly" else dict(width=0),
                ),
                name=f"{cid} {'(anomaly)' if anomaly_status == 'Anomaly' else ''}",
                text=hover_text,
                hoverinfo="text",
                showlegend=(anomaly_status == "Normal" or subset.shape[0] > 0),
            )
        )

fig_map.update_layout(
    title=f"PCA Cluster Map (k={selected_k}) — Anomalies marked with x",
    xaxis_title="PC1",
    yaxis_title="PC2",
    legend=dict(font=dict(size=10)),
)
apply_theme(fig_map, 520)
st.plotly_chart(fig_map, use_container_width=True)
st.caption(
    f"2-D PCA projection with k={selected_k} K-Means clusters. "
    f"{anomaly['n_anomalies']:,} anomalies (x markers) identified by Isolation Forest."
)

# Cluster profile heatmap
st.markdown("**Cluster Profile Heatmap**")

profiles = clust["kmeans"][str(selected_k)]["profiles"]
features = clust["features"]
feature_labels = [display_name(f) for f in features]

heatmap_z = []
cluster_labels_y = []
for cid in sorted(profiles.keys(), key=int):
    p = profiles[cid]
    mean_feats = p["mean_features"]
    row = [mean_feats.get(f, 0) for f in features]
    heatmap_z.append(row)
    cluster_labels_y.append(f"Cluster {cid}")

# Normalize columns (min-max per feature) for better visualization
z_arr = np.array(heatmap_z)
col_min = z_arr.min(axis=0)
col_max = z_arr.max(axis=0)
denom = col_max - col_min
denom[denom == 0] = 1
z_norm = (z_arr - col_min) / denom

fig_heat = heatmap_chart(
    z=z_norm.tolist(),
    x_labels=feature_labels,
    y_labels=cluster_labels_y,
    title="Cluster Feature Profiles (min-max normalized)",
    height=max(300, len(cluster_labels_y) * 60 + 100),
)
st.plotly_chart(fig_heat, use_container_width=True)
st.caption("Each cell shows the normalized mean feature value for the cluster. Warmer colors indicate higher relative values.")

# ── Prescriptive: Cluster allocation guidance ──
best_cluster = max(profiles.keys(), key=lambda cid: profiles[cid].get('success_rate', 0))
worst_cluster = min(profiles.keys(), key=lambda cid: profiles[cid].get('success_rate', 0))
best_sr = profiles[best_cluster]['success_rate']
worst_sr = profiles[worst_cluster]['success_rate']
best_persona_name = personas.get(best_cluster, personas.get(int(best_cluster), {})).get('name', f'Cluster {best_cluster}')
worst_persona_name = personas.get(worst_cluster, personas.get(int(worst_cluster), {})).get('name', f'Cluster {worst_cluster}')

if mandate == "GIC — Financial Return":
    cluster_action = (
        f"<b>Allocation Strategy (GIC):</b> <b>{best_persona_name}</b> (Cluster {best_cluster}) shows the highest success rate at "
        f"<b>{best_sr:.1%}</b>. <b>Action:</b> Overweight new deals matching this cluster's feature profile. "
        f"<b>{worst_persona_name}</b> (Cluster {worst_cluster}, {worst_sr:.1%} success) — cap at 10% of new commitments."
    )
elif mandate == "PIF — Strategic Alignment":
    cluster_action = (
        f"<b>Allocation Strategy (PIF):</b> Don't optimize solely for highest success rate. "
        f"<b>Action:</b> Check which cluster has highest tech transfer and bilateral scores in the heatmap — "
        f"that's your strategic priority cluster, even if success rate is lower. Use {best_persona_name} as your financial anchor."
    )
else:
    cluster_action = (
        f"<b>Allocation Strategy (Mubadala):</b> Diversify across all {selected_k} clusters proportional to "
        f"risk-adjusted returns. <b>Action:</b> No single cluster should exceed 40% of new deal flow. "
        f"Currently, Cluster {best_cluster} ({best_sr:.1%}) outperforms Cluster {worst_cluster} ({worst_sr:.1%}) — rebalance gradually."
    )
st.html(swf_insight(cluster_action))

# Cluster persona captions
personas = frameworks.get("cluster_personas", {})
for cid in sorted(profiles.keys(), key=int):
    p = profiles[cid]
    persona = personas.get(cid, personas.get(int(cid), {}))
    persona_name = persona.get("name", f"Cluster {cid}")
    sr = p.get("success_rate", 0)
    st.caption(f"Cluster {cid} — **{persona_name}** | Success rate: {sr*100:.1f}% | Size: {p['size']:,}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section D: Investment Strategy by Archetype
# ─────────────────────────────────────────────────────────────────────────────
st.html(section_header("D — Investment Strategy by Archetype"))

# Strategy cards
if personas:
    sorted_ids = sorted(personas.keys(), key=lambda x: int(x))
    n_cols = min(len(sorted_ids), 4)
    cols = st.columns(n_cols)

    for idx, cid in enumerate(sorted_ids):
        persona = personas[cid]
        with cols[idx % n_cols]:
            risk_color = {
                "Low": COLORS["success"],
                "Medium": COLORS["warning"],
                "High": COLORS["danger"],
            }.get(persona.get("risk_level", "Medium"), COLORS["muted"])

            top_sectors = persona.get("top_sectors", {})
            if isinstance(top_sectors, dict):
                sector_str = ", ".join(list(top_sectors.keys())[:3])
            elif isinstance(top_sectors, list):
                sector_str = ", ".join(top_sectors[:3])
            else:
                sector_str = str(top_sectors)

            top_stages = persona.get("top_stages", {})
            if isinstance(top_stages, dict):
                stage_str = ", ".join(list(top_stages.keys())[:2])
            elif isinstance(top_stages, list):
                stage_str = ", ".join(top_stages[:2])
            else:
                stage_str = str(top_stages)

            sr_pct = persona.get("success_rate", 0)
            if sr_pct <= 1:
                sr_pct = sr_pct * 100

            card_html = f"""
            <div class="strategy-card" style="min-height:320px;">
                <h4>{persona.get('name', f'Cluster {cid}')}</h4>
                <p style="color:#95A5A6;font-size:0.85em;">{persona.get('description', '')}</p>
                <div style="margin:8px 0;">
                    <span style="font-size:0.75em;color:#95A5A6;text-transform:uppercase;">Success Rate</span>
                    <div style="background:rgba(255,255,255,0.1);border-radius:4px;height:10px;margin:4px 0;">
                        <div style="background:{COLORS['accent']};width:{min(sr_pct, 100):.0f}%;height:100%;border-radius:4px;"></div>
                    </div>
                    <span style="color:{COLORS['accent']};font-weight:600;">{sr_pct:.1f}%</span>
                </div>
                <p style="font-size:0.82em;margin:6px 0;">
                    <span style="color:{risk_color};font-weight:600;">Risk: {persona.get('risk_level', 'N/A')}</span>
                </p>
                <p style="font-size:0.82em;color:#ECF0F1;margin:4px 0;">
                    <strong>Strategy:</strong> {persona.get('strategy', 'N/A')}
                </p>
                <p style="font-size:0.78em;color:#95A5A6;margin:4px 0;">
                    Sectors: {sector_str}<br>Stages: {stage_str}
                </p>
                <p style="font-size:0.78em;color:#95A5A6;">
                    Size: {persona.get('size', 'N/A'):,} ventures
                </p>
            </div>
            """
            st.html(card_html)

    # Success rate comparison bar chart
    st.markdown("**Success Rate by Archetype**")
    archetype_names = [personas[cid].get("name", f"Cluster {cid}") for cid in sorted_ids]
    archetype_sr = []
    for cid in sorted_ids:
        sr = personas[cid].get("success_rate", 0)
        archetype_sr.append(sr * 100 if sr <= 1 else sr)

    sr_colors = [PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i in range(len(sorted_ids))]

    fig_sr = go.Figure(
        data=[
            go.Bar(
                x=archetype_names,
                y=archetype_sr,
                marker_color=sr_colors,
                text=[f"{v:.1f}%" for v in archetype_sr],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        ]
    )
    fig_sr.update_layout(
        title="Success Rate Comparison Across Archetypes",
        yaxis_title="Success Rate (%)",
        xaxis_title="",
    )
    apply_theme(fig_sr, 400)
    st.plotly_chart(fig_sr, use_container_width=True)
    st.caption("Comparing venture success rates across discovered archetypes enables differentiated allocation strategies.")
else:
    st.info("Cluster persona data not available.")

# Anomaly summary at bottom
st.html(metric_highlight(
    f"Anomaly Detection: {anomaly['n_anomalies']:,} anomalies flagged | "
    f"Anomaly success rate: {anomaly['anomaly_success_rate']*100:.1f}% vs "
    f"Normal: {anomaly['normal_success_rate']*100:.1f}%"
))

if "anomaly" in interp.get("clustering", {}):
    st.caption(interp["clustering"]["anomaly"])

# ── Prescriptive: Anomaly action ──
anom_sr = anomaly['anomaly_success_rate']
norm_sr = anomaly['normal_success_rate']
if anom_sr > norm_sr:
    anom_insight = (
        f"<b>Anomaly Opportunity:</b> Outlier ventures actually <b>outperform</b> ({anom_sr:.1%} vs {norm_sr:.1%} normal). "
        f"These {anomaly['n_anomalies']:,} anomalies may be unconventional winners — breakthrough companies. "
        f"<b>Action:</b> Don't auto-reject outliers. Assign senior analysts to review each individually. "
        f"The best returns often come from pattern-breakers."
    )
else:
    anom_insight = (
        f"<b>Anomaly Risk:</b> Outliers <b>underperform</b> ({anom_sr:.1%} vs {norm_sr:.1%} normal). "
        f"These {anomaly['n_anomalies']:,} anomalies have extreme feature values correlating with higher failure. "
        f"<b>Action:</b> Flag any new deal classified as an anomaly for enhanced due diligence. "
        f"Require IC approval with explicit risk mitigation before committing capital."
    )
st.html(swf_insight(anom_insight))
