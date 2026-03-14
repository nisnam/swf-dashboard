"""
SIIP Dashboard — Page 1: Pipeline Intelligence
=================================================
Understand your venture universe — data quality, distributions,
correlations, ARIMA deal flow forecasting.
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as sp_stats

from utils.data_loader import (
    load_data, load_arima_results, load_prep_artifacts, load_interpretations,
    load_frameworks, display_name, SWF_DIMENSIONS,
    MANDATES, get_mandate_insight, compute_dimension_scores, get_df_stats, DIMENSION_ICONS
)
from utils.styles import (
    inject_css, kpi_card, section_header, metric_highlight, evidence_panel,
    swf_insight, format_currency, format_pct, COLORS, PLOTLY_COLORS,
    mandate_badge, bridge_sentence
)
from utils.charts import apply_theme, bar_chart, heatmap_chart, scatter_chart, line_chart, histogram_chart, donut_chart

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pipeline Intelligence | SIIP", layout="wide")
st.html(inject_css())

# ── Mandate ─────────────────────────────────────────────────────────────────
mandate = st.session_state.get('mandate', 'GIC — Financial Return')
mandate_config = MANDATES[mandate]

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
arima = load_arima_results()
prep = load_prep_artifacts()
interp = load_interpretations()
frameworks = load_frameworks()

ivs = prep['ivs']
fe_log = prep['fe_log']
class_dist = prep.get('class_distribution', {})

pipeline_interp = interp.get('pipeline', {})
dq_interp = pipeline_interp.get('data_quality', '')
arima_interp = pipeline_interp.get('arima', '')

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("# Pipeline Intelligence")
st.html(
    '<p style="color:#95A5A6;margin-top:-10px;">'
    'Venture universe diagnostics — data quality, distributional profiles, '
    'sector-stage dynamics, correlation structure, and ARIMA deal-flow forecasting</p>')
st.html(mandate_badge(mandate, mandate_config))
st.html(swf_insight(get_mandate_insight('pipeline', mandate)))

# Top-line KPI strip
numeric_cols = df.select_dtypes(include='number').columns.tolist()
n_rows, n_cols = df.shape
missing_pct = df.isnull().mean().mean()
n_sectors = df['sector'].nunique() if 'sector' in df.columns else 0
success_rate = df['outcome_binary'].mean() if 'outcome_binary' in df.columns else 0

k1, k2, k3, k4 = st.columns(4)
k1.html(kpi_card("Observations", f"{n_rows:,}"))
k2.html(kpi_card("Features", f"{n_cols}"))
k3.html(kpi_card("Missing Data", f"{missing_pct:.1%}"))
k4.html(kpi_card("Base Success Rate", f"{success_rate:.1%}"))

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — Data Quality & Engineering
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("A — Data Quality & Feature Engineering"))

with st.expander("Expand: Data Preparation Details", expanded=False):
    st.html(evidence_panel("Data Preparation & Feature Engineering"))

    col_log, col_table = st.columns([1, 1])

    # Feature engineering log
    with col_log:
        st.markdown("**Feature Engineering Log**")
        for item in fe_log:
            st.markdown(f"- {item}")

    # Composite features table
    with col_table:
        st.markdown("**Engineered Composite Features**")
        composites = {
            'esg_composite': 'Weighted combination of ESG sub-scores',
            'country_risk_composite': 'Aggregate political + economic risk',
            'bilateral_composite': 'Trade & diplomatic alignment index',
            'stability_x_stage': 'Country stability interacted with funding stage',
            'esg_x_competition': 'ESG score interacted with competitive intensity',
            'tech_transfer_proxy': 'IP + patent-based technology transfer measure',
            'regulatory_moat': 'Regulatory barrier to entry proxy',
        }
        comp_df = pd.DataFrame(
            [(k, v) for k, v in composites.items() if k in df.columns],
            columns=['Feature', 'Description'],
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Missing value summary
    st.markdown("**Missing Value Summary**")
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) > 0:
        miss_df = pd.DataFrame({
            'Feature': miss.index,
            'Missing Count': miss.values,
            'Missing %': (miss.values / len(df) * 100).round(2),
        })
        st.dataframe(miss_df, use_container_width=True, hide_index=True)
    else:
        st.success("No missing values remain after imputation.")

    if dq_interp:
        st.caption(dq_interp)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — Portfolio Distribution
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("B — Portfolio Distribution Analysis"))

col_hist, col_stats = st.columns([1, 1])

with col_hist:
    st.markdown("**Interactive Feature Distribution**")
    numeric_features = [c for c in numeric_cols if c not in ('outcome_binary',)]
    selected_feature = st.selectbox(
        "Select a numeric feature",
        options=numeric_features,
        format_func=display_name,
        key="dist_feature",
    )
    vals = df[selected_feature].dropna()
    fig_hist = histogram_chart(vals, title=f"Distribution of {display_name(selected_feature)}", nbins=40)
    # Add mean line
    mean_val = vals.mean()
    fig_hist.add_vline(x=mean_val, line_dash="dash", line_color=COLORS['accent'],
                       annotation_text=f"Mean: {mean_val:.2f}", annotation_font_color=COLORS['accent'])
    st.plotly_chart(fig_hist, use_container_width=True)

    # Stats overlay below the chart
    s_mean = vals.mean()
    s_med = vals.median()
    s_std = vals.std()
    s_skew = vals.skew()
    skew_label = "right-skewed" if s_skew > 0.5 else "left-skewed" if s_skew < -0.5 else "symmetric"
    st.html(
        metric_highlight(
            f"Mean: <b>{s_mean:.3f}</b> &nbsp;|&nbsp; "
            f"Median: <b>{s_med:.3f}</b> &nbsp;|&nbsp; "
            f"Std: <b>{s_std:.3f}</b> &nbsp;|&nbsp; "
            f"Skewness: <b>{s_skew:.3f}</b> ({skew_label})"
        ))
    st.caption(
        f"Distribution of {display_name(selected_feature)} across {len(vals):,} observations. "
        f"The dashed gold line marks the mean."
    )

with col_stats:
    st.markdown("**Summary Statistics — All Numeric Features**")
    desc = df[numeric_cols].describe().T
    desc.index = [display_name(c) for c in desc.index]
    desc = desc.round(3)
    st.dataframe(desc, use_container_width=True, height=500)
    st.caption("Standard descriptive statistics (count, mean, std, min, quartiles, max) for every numeric column.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — Sector x Stage Analysis
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("C — Sector x Funding Stage Success Analysis"))

if 'sector' in df.columns and 'funding_stage' in df.columns and 'outcome_binary' in df.columns:
    ct = pd.crosstab(df['sector'], df['funding_stage'], values=df['outcome_binary'], aggfunc='mean')
    ct = ct.fillna(0)

    fig_ct = heatmap_chart(
        z=ct.values.tolist(),
        x_labels=ct.columns.tolist(),
        y_labels=ct.index.tolist(),
        title="Success Rate by Sector x Funding Stage",
        height=max(400, len(ct.index) * 30),
        colorscale='RdYlGn',
    )
    st.plotly_chart(fig_ct, use_container_width=True)
    st.caption(
        "Heatmap displays the mean success rate (outcome_binary) at each sector-stage intersection. "
        "Green cells indicate higher historical success; red cells signal caution. "
        "Cell values represent the proportion of successful outcomes."
    )

    # ── Prescriptive: Sector-stage guidance ──
    # Find best and worst sector-stage combos
    ct_stacked = ct.stack()
    if len(ct_stacked) > 0:
        best_combo = ct_stacked.idxmax()
        worst_combo = ct_stacked[ct_stacked > 0].idxmin() if (ct_stacked > 0).any() else ("N/A", "N/A")
        best_val = ct_stacked.max()
        if mandate == "GIC — Financial Return":
            heatmap_action = (
                f"<b>Sourcing Priority (GIC):</b> Highest success rate is <b>{best_combo[0]}</b> at <b>{best_combo[1]}</b> stage ({best_val:.0%}). "
                f"<b>Action:</b> Direct sourcing teams to prioritize this sector-stage combination. "
                f"Avoid <b>{worst_combo[0]}</b> at <b>{worst_combo[1]}</b> unless valuation compensates for the risk."
            )
        elif mandate == "PIF — Strategic Alignment":
            heatmap_action = (
                f"<b>Sourcing Priority (PIF):</b> Green cells show proven success corridors. "
                f"<b>Action:</b> Cross-reference green cells with Vision 2030 priority sectors. "
                f"Where strategic sectors show red cells, engage bilateral partners to de-risk."
            )
        else:
            heatmap_action = (
                f"<b>Sourcing Priority (Mubadala):</b> Best risk-adjusted entry point is <b>{best_combo[0]}</b> at <b>{best_combo[1]}</b> ({best_val:.0%}). "
                f"<b>Action:</b> Allocate across multiple green cells for diversified exposure. Avoid concentration in any single cell."
            )
        st.html(swf_insight(heatmap_action))

    # Count matrix for context
    with st.expander("Show deal count matrix"):
        ct_count = pd.crosstab(df['sector'], df['funding_stage'])
        st.dataframe(ct_count, use_container_width=True)
        st.caption("Number of deals at each sector-stage intersection. Low-count cells should be interpreted with caution.")
else:
    st.warning("Required columns (sector, funding_stage, outcome_binary) not found for crosstab analysis.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — Correlation Intelligence
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("D — Correlation Intelligence"))

# Filter IVs that exist in the dataframe
iv_cols = [c for c in ivs if c in df.columns]

if len(iv_cols) >= 2:
    corr = df[iv_cols].corr()

    # Rename for display
    display_labels = [display_name(c) for c in iv_cols]

    fig_corr = heatmap_chart(
        z=corr.values.tolist(),
        x_labels=display_labels,
        y_labels=display_labels,
        title="Correlation Matrix — Independent Variables",
        height=550,
        colorscale='RdBu_r',
    )
    fig_corr.update_layout(
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Identify strongest correlations
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_pairs = corr.where(mask).stack().reset_index()
    corr_pairs.columns = ['Var1', 'Var2', 'Correlation']
    corr_pairs['AbsCorr'] = corr_pairs['Correlation'].abs()
    top_pairs = corr_pairs.nlargest(5, 'AbsCorr')

    if len(top_pairs) > 0:
        pairs_text = " | ".join(
            f"{display_name(r['Var1'])} & {display_name(r['Var2'])}: {r['Correlation']:.3f}"
            for _, r in top_pairs.iterrows()
        )
        st.caption(f"Strongest pairwise correlations: {pairs_text}")

    st.markdown("")

    # Scatter explorer
    st.markdown("**Scatter Explorer**")
    sc1, sc2 = st.columns(2)
    with sc1:
        x_feat = st.selectbox("X-axis feature", iv_cols, format_func=display_name, index=0, key="scatter_x")
    with sc2:
        default_y = min(1, len(iv_cols) - 1)
        y_feat = st.selectbox("Y-axis feature", iv_cols, format_func=display_name, index=default_y, key="scatter_y")

    if 'outcome_binary' in df.columns:
        color_col = df['outcome_binary'].map({0: 'Failure', 1: 'Success'})
        fig_scatter = scatter_chart(
            x=df[x_feat], y=df[y_feat],
            color=color_col,
            title=f"{display_name(x_feat)} vs {display_name(y_feat)}",
            labels={'x': display_name(x_feat), 'y': display_name(y_feat), 'color': 'Outcome'},
            color_discrete_map={'Failure': COLORS['danger'], 'Success': COLORS['success']},
            height=450,
        )
    else:
        fig_scatter = scatter_chart(
            x=df[x_feat], y=df[y_feat],
            title=f"{display_name(x_feat)} vs {display_name(y_feat)}",
            labels={'x': display_name(x_feat), 'y': display_name(y_feat)},
            height=450,
        )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Compute correlation for selected pair
    pair_corr = df[[x_feat, y_feat]].dropna().corr().iloc[0, 1]
    st.caption(
        f"Pearson r = {pair_corr:.3f} between {display_name(x_feat)} and {display_name(y_feat)}. "
        f"Points colored by investment outcome (green = success, red = failure)."
    )

    # ── Prescriptive: Correlation implication ──
    if abs(pair_corr) > 0.7:
        st.html(swf_insight(
            f"<b>Multicollinearity Alert:</b> {display_name(x_feat)} and {display_name(y_feat)} are highly correlated (r={pair_corr:.3f}). "
            f"<b>Action:</b> Use only one of these in screening criteria to avoid double-counting the same signal. "
            f"Choose the feature with stronger standalone relationship to outcomes."
        ))
    elif abs(pair_corr) < 0.2:
        st.html(swf_insight(
            f"<b>Independent Signals:</b> {display_name(x_feat)} and {display_name(y_feat)} are weakly correlated (r={pair_corr:.3f}). "
            f"<b>Action:</b> Screen on both independently — they capture different aspects of venture quality and should both be in your scorecard."
        ))
else:
    st.warning("Insufficient independent variables found for correlation analysis.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — Deal Flow Forecasting (ARIMA)
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("E — Deal Flow Forecasting (ARIMA)"))

if arima:
    hist = arima.get('historical', {})
    forecast = arima.get('forecast', {})
    model_summary = arima.get('model_summary', {})
    decomp = arima.get('decomposition', {})
    deal_flow_q = arima.get('deal_flow_quarterly', {})

    # ── Historical + Forecast combined chart ──
    fig_arima = go.Figure()

    # Historical series
    if hist.get('dates') and hist.get('values'):
        fig_arima.add_trace(go.Scatter(
            x=hist['dates'], y=hist['values'],
            mode='lines+markers', name='Historical',
            line=dict(color=COLORS['accent2'], width=2),
            marker=dict(size=4),
        ))

    # Forecast series
    if forecast.get('dates') and forecast.get('values'):
        fig_arima.add_trace(go.Scatter(
            x=forecast['dates'], y=forecast['values'],
            mode='lines+markers', name='Forecast',
            line=dict(color=COLORS['accent'], width=2, dash='dash'),
            marker=dict(size=4),
        ))

        # Confidence interval (shaded area)
        if forecast.get('upper_ci') and forecast.get('lower_ci'):
            fig_arima.add_trace(go.Scatter(
                x=list(forecast['dates']) + list(reversed(forecast['dates'])),
                y=list(forecast['upper_ci']) + list(reversed(forecast['lower_ci'])),
                fill='toself',
                fillcolor='rgba(212,175,55,0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% CI',
                showlegend=True,
            ))

    fig_arima.update_layout(
        title="Deal Volume — Historical & ARIMA Forecast",
        xaxis_title="Date",
        yaxis_title="Deal Count",
        legend=dict(x=0.01, y=0.99),
    )
    fig_arima = apply_theme(fig_arima, height=450)
    st.plotly_chart(fig_arima, use_container_width=True)

    st.caption(
        "Monthly deal volume with ARIMA out-of-sample forecast. "
        "The shaded gold region represents the 95% confidence interval. "
        "Widening bands indicate increasing forecast uncertainty over the horizon."
    )

    # ── Prescriptive: ARIMA deal flow action ──
    if forecast.get('values') and hist.get('values'):
        last_hist = hist['values'][-1] if hist['values'] else 0
        avg_forecast = sum(forecast['values']) / len(forecast['values']) if forecast['values'] else 0
        if avg_forecast > last_hist * 1.1:
            trend_word = "increasing"
            action = "Expand diligence team capacity and reserve dry powder for the surge. Consider pre-approved IC mandates for repeat sectors."
        elif avg_forecast < last_hist * 0.9:
            trend_word = "declining"
            action = "Tighten screening criteria — fewer deals means you can be more selective. Use downtime to deepen existing portfolio monitoring."
        else:
            trend_word = "stable"
            action = "Maintain current deployment pace. Focus on deal quality over volume."
        st.html(swf_insight(
            f"<b>Deal Flow Outlook:</b> ARIMA forecasts {trend_word} deal volume over the next 12 months. "
            f"<b>Action:</b> {action}"
        ))

    # ── Seasonal decomposition ──
    if decomp:
        st.markdown("**Seasonal Decomposition**")
        dc1, dc2 = st.columns(2)

        with dc1:
            if decomp.get('trend') and decomp.get('dates'):
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=decomp['dates'], y=decomp['trend'],
                    mode='lines', name='Trend',
                    line=dict(color=COLORS['accent2'], width=2),
                ))
                fig_trend.update_layout(title="Trend Component", xaxis_title="Date", yaxis_title="Trend")
                fig_trend = apply_theme(fig_trend, height=300)
                st.plotly_chart(fig_trend, use_container_width=True)
                st.caption("Long-run trend extracted via seasonal decomposition.")

        with dc2:
            if decomp.get('seasonal') and decomp.get('dates'):
                fig_seasonal = go.Figure()
                fig_seasonal.add_trace(go.Scatter(
                    x=decomp['dates'], y=decomp['seasonal'],
                    mode='lines', name='Seasonal',
                    line=dict(color=COLORS['success'], width=1.5),
                ))
                fig_seasonal.update_layout(title="Seasonal Component", xaxis_title="Date", yaxis_title="Seasonal")
                fig_seasonal = apply_theme(fig_seasonal, height=300)
                st.plotly_chart(fig_seasonal, use_container_width=True)
                st.caption("Recurring seasonal pattern in deal flow activity.")

    # ── ARIMA model stats ──
    if model_summary:
        st.markdown("**ARIMA Model Summary**")
        ms1, ms2, ms3 = st.columns(3)
        order = model_summary.get('order', 'N/A')
        aic = model_summary.get('aic', 'N/A')
        bic = model_summary.get('bic', 'N/A')
        ms1.html(kpi_card("Model Order", str(order)))
        ms2.html(kpi_card("AIC", f"{aic:.1f}" if isinstance(aic, (int, float)) else str(aic)))
        ms3.html(kpi_card("BIC", f"{bic:.1f}" if isinstance(bic, (int, float)) else str(bic)))

    if arima_interp:
        st.caption(arima_interp)
else:
    st.warning("ARIMA results not available. Run the precompute pipeline to generate forecasts.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.html(
    '<p style="text-align:center;color:#95A5A6;font-size:0.8em;">'
    'SIIP Pipeline Intelligence Module &mdash; Sovereign Investment Intelligence Platform'
    '</p>')
