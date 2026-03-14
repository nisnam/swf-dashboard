"""
SIIP — Risk Engine (Page 2)
============================
Regression analysis: Can structural features predict outcomes?
What are the linear risk drivers?
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import (
    load_data, load_regression_results, load_interpretations,
    load_prep_artifacts, display_name, SWF_DIMENSIONS,
    MANDATES, get_mandate_insight, DIMENSION_ICONS
)
from utils.styles import (
    inject_css, kpi_card, section_header, metric_highlight, COLORS, PLOTLY_COLORS,
    swf_insight, mandate_badge, bridge_sentence
)
from utils.charts import apply_theme, bar_chart, heatmap_chart, histogram_chart

st.set_page_config(page_title="Risk Engine | SIIP", layout="wide")
st.html(inject_css())

# --- Mandate ---
mandate = st.session_state.get('mandate', 'GIC — Financial Return')
mandate_config = MANDATES[mandate]

# --- Load data ---
df = load_data()
reg = load_regression_results()
interp = load_interpretations()
prep = load_prep_artifacts()

ols_stats = reg['ols_statsmodels']
models = reg['models']
interpretations = interp['regression']

# =====================================================================
# SECTION A — The Question
# =====================================================================
st.markdown("# Risk Engine")
st.caption("Can structural features predict venture outcomes? Identifying linear risk drivers across the SWF pipeline.")
st.html(mandate_badge(mandate, mandate_config))
st.html(swf_insight(get_mandate_insight('risk', mandate)))
st.markdown("---")

st.html(section_header("A. The Question — Dependent & Independent Variables"))

col_dv, col_iv = st.columns([1, 2])

with col_dv:
    st.markdown("##### Dependent Variable: `outcome_numeric` (0-3)")
    outcome_counts = df['outcome_numeric'].value_counts().sort_index()
    colors_dv = [COLORS['danger'], COLORS['warning'], COLORS['accent2'], COLORS['success']]
    fig_dv = go.Figure(data=[go.Bar(
        x=outcome_counts.index.astype(str).tolist(),
        y=outcome_counts.values.tolist(),
        marker_color=colors_dv[:len(outcome_counts)],
        text=outcome_counts.values.tolist(),
        textposition='auto',
        hovertemplate='Outcome %{x}: %{y:,} ventures<extra></extra>'
    )])
    fig_dv.update_layout(
        xaxis_title='Outcome Score',
        yaxis_title='Count'
    )
    fig_dv = apply_theme(fig_dv, 320)
    st.plotly_chart(fig_dv, use_container_width=True)
    st.caption("Distribution of the ordinal outcome variable (0 = failure, 3 = successful exit).")

with col_iv:
    st.markdown("##### Independent Variables — 17 Features by SWF Priority")
    for dimension, features in SWF_DIMENSIONS.items():
        iv_list = reg.get('ivs', prep.get('ivs', []))
        active = [f for f in features if f in iv_list]
        if active:
            names = ", ".join([display_name(f) for f in active])
            st.markdown(f"**{dimension}:** {names}")
    st.caption(f"Total IVs: {len(reg.get('ivs', prep.get('ivs', [])))} features spanning 5 SWF priority dimensions.")

st.caption(interpretations.get('overview', ''))
st.markdown("")

# =====================================================================
# SECTION B — OLS Findings
# =====================================================================
st.markdown("---")
st.html(section_header("B. Evidence: OLS Linear Regression"))

# KPI row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.html(kpi_card("R-Squared", f"{ols_stats['r2']:.4f}"))
with k2:
    st.html(kpi_card("Adjusted R²", f"{ols_stats['adj_r2']:.4f}"))
with k3:
    st.html(kpi_card("F-Statistic", f"{ols_stats['f_stat']:.2f}"))
with k4:
    f_pval = ols_stats['f_pvalue']
    pval_str = f"{f_pval:.2e}" if f_pval < 0.001 else f"{f_pval:.4f}"
    st.html(kpi_card("F p-value", pval_str))

st.markdown("")

# Coefficient table and chart
col_table, col_chart = st.columns([1, 1])

with col_table:
    st.markdown("##### OLS Coefficients")
    ols_coefs = models['OLS']['coefficients']
    coef_df = pd.DataFrame([
        {
            'Feature': display_name(feat),
            '\u03b2 (Coefficient)': round(coef, 4),
            'Direction': '\u2191' if coef > 0 else '\u2193'
        }
        for feat, coef in sorted(ols_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    ])
    st.dataframe(coef_df, use_container_width=True, hide_index=True, height=460)

with col_chart:
    st.markdown("##### Coefficient Magnitudes")
    sorted_coefs = sorted(ols_coefs.items(), key=lambda x: x[1])
    feat_names = [display_name(f) for f, _ in sorted_coefs]
    coef_vals = [c for _, c in sorted_coefs]
    coef_colors = [COLORS['success'] if c > 0 else COLORS['danger'] for c in coef_vals]

    fig_coef = go.Figure(data=[go.Bar(
        y=feat_names,
        x=coef_vals,
        orientation='h',
        marker_color=coef_colors,
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    )])
    fig_coef.update_layout(
        xaxis_title='\u03b2 Coefficient',
        title='OLS Coefficients (green = positive, red = negative)'
    )
    fig_coef = apply_theme(fig_coef, 460)
    st.plotly_chart(fig_coef, use_container_width=True)
    st.caption("Horizontal bar chart of all OLS coefficients. Green bars indicate positive association with outcome; red bars indicate negative.")

st.caption(interpretations.get('ols_finding', ''))

# ── Prescriptive: OLS coefficient interpretation ──
top3_feats = sorted(ols_coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
top3_names = [display_name(f) for f, _ in top3_feats]
if mandate == "GIC — Financial Return":
    ols_action = (
        f"<b>Key Linear Drivers (GIC):</b> The 3 strongest predictors are <b>{', '.join(top3_names)}</b>. "
        f"<b>Action:</b> Build these into your IC scorecard as primary screening criteria. "
        f"Any deal below the 25th percentile on {top3_names[0]} should trigger enhanced due diligence."
    )
elif mandate == "PIF — Strategic Alignment":
    strategic_feats = [f for f, c in sorted_coefs if f in ['tech_transfer_proxy', 'bilateral_composite', 'national_strategy_alignment']]
    strategic_names = [display_name(f) for f in strategic_feats[:3]] if strategic_feats else top3_names
    ols_action = (
        f"<b>Strategic Signal Check (PIF):</b> Look for strategic features in the top coefficients: "
        f"<b>{', '.join(strategic_names)}</b>. If strategic features rank low, it means linear models don't capture their value — "
        f"<b>Action:</b> Don't rely on regression alone for strategic deals. Use the Prediction Engine's non-linear models instead."
    )
else:
    ols_action = (
        f"<b>Balanced View (Mubadala):</b> Top 3 drivers (<b>{', '.join(top3_names)}</b>) span multiple SWF dimensions. "
        f"<b>Action:</b> Ensure your screening criteria don't over-index on any one dimension. "
        f"Weight all 5 dimensions equally in your preliminary deal assessment."
    )
st.html(swf_insight(ols_action))
st.markdown("")

# =====================================================================
# SECTION C — Regularization
# =====================================================================
st.markdown("---")
st.html(section_header("C. Evidence: Regularization — Ridge / Lasso / ElasticNet"))

col_compare, col_lasso = st.columns([1, 1])

with col_compare:
    st.markdown("##### Model Comparison")
    comparison_rows = []
    for model_name in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']:
        m = models[model_name]
        comparison_rows.append({
            'Model': model_name,
            'R\u00b2': round(m['R2'], 4),
            'MAE': round(m['MAE'], 4),
            'RMSE': round(m['RMSE'], 4),
        })
    compare_df = pd.DataFrame(comparison_rows)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)
    st.caption("Performance metrics across all four linear models. Regularization provides marginal improvement over vanilla OLS.")

with col_lasso:
    st.markdown("##### Lasso Feature Selection")
    lasso_coefs = models['Lasso']['coefficients']
    n_nonzero = reg.get('models.Lasso.n_nonzero',
                        sum(1 for v in lasso_coefs.values() if abs(v) > 1e-10))
    st.html(metric_highlight(
        f"Lasso retained <b>{n_nonzero}</b> of {len(lasso_coefs)} features — "
        f"the remaining {len(lasso_coefs) - n_nonzero} were zeroed out."
    ))

# Lasso coefficient chart — all 17 features
st.markdown("##### Lasso Coefficients — All Features")
lasso_sorted = sorted(lasso_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
lasso_feat_names = [display_name(f) for f, _ in lasso_sorted]
lasso_vals = [c for _, c in lasso_sorted]
lasso_colors = [COLORS['success'] if abs(c) > 1e-10 else COLORS['danger'] for c in lasso_vals]

fig_lasso = go.Figure(data=[go.Bar(
    y=lasso_feat_names[::-1],
    x=lasso_vals[::-1],
    orientation='h',
    marker_color=lasso_colors[::-1],
    hovertemplate='%{y}: %{x:.4f}<extra></extra>'
)])
fig_lasso.update_layout(
    xaxis_title='\u03b2 Coefficient',
    title='Lasso Coefficients (green = retained, red = zeroed)'
)
fig_lasso = apply_theme(fig_lasso, 450)
st.plotly_chart(fig_lasso, use_container_width=True)
st.caption("Lasso (L1) regularization zeroes out features with negligible linear contribution, revealing the minimal set of linear risk drivers.")

st.caption(interpretations.get('lasso_finding', ''))

# ── Prescriptive: Lasso feature selection action ──
retained_features = [k for k, v in lasso_coefs.items() if abs(v) > 1e-10]
retained_names = [display_name(f) for f in retained_features]
st.html(swf_insight(
    f"<b>Minimum Viable Scorecard:</b> Lasso retained only <b>{len(retained_features)}</b> features: "
    f"<b>{', '.join(retained_names)}</b>. These are the only features with meaningful linear signal. "
    f"<b>Action:</b> Use these {len(retained_features)} features as your 'quick screen' checklist for initial deal filtering. "
    f"A deal that fails on 2+ of these should be deprioritized before deeper analysis."
))
st.markdown("")

# =====================================================================
# SECTION D — Key Finding
# =====================================================================
st.markdown("---")
st.html(section_header("D. Key Finding & Bridge to Prediction"))

st.warning(interpretations.get('key_finding', 'Linear models explain limited variance in venture outcomes.'))

st.markdown("")
st.html(metric_highlight(
    interpretations.get('bridge', 'The low R-squared motivates non-linear classification approaches explored in the Prediction Engine.')
))

st.markdown("")
st.markdown("##### Residual Analysis")
st.markdown(
    "With an R\u00b2 of **{r2:.4f}**, approximately **{resid:.1f}%** of outcome variance remains unexplained by "
    "linear feature combinations. This signals that:".format(
        r2=ols_stats['r2'],
        resid=(1 - ols_stats['r2']) * 100
    )
)
st.markdown("""
- **Non-linear interactions** between features likely drive outcomes more than additive effects.
- **Threshold effects** (e.g., minimum viable margin, critical team size) cannot be captured by OLS.
- **Categorical outcome structure** (0-3 ordinal) is poorly suited to continuous regression — classification is the natural next step.
- The **statistically significant F-statistic** confirms that features *do* carry signal, but the linear functional form is insufficient to extract it.
""")
st.caption("These findings directly motivate the Prediction Engine (Page 3), where tree-based and ensemble classifiers capture non-linear patterns.")

# ── Prescriptive: R² action item ──
st.html(swf_insight(
    f"<b>Critical Insight:</b> Linear models explain only <b>{ols_stats['r2']:.1%}</b> of outcomes. "
    f"This means <b>{(1-ols_stats['r2'])*100:.0f}% of what drives success is non-linear</b> — threshold effects, feature interactions, "
    f"and combinatorial patterns. <b>Action:</b> Do NOT use regression coefficients as your primary deal-scoring tool. "
    f"Instead, use the <b>Prediction Engine</b> (non-linear ML classifiers) for deal screening, and treat this page's coefficients "
    f"as directional indicators only."
))
