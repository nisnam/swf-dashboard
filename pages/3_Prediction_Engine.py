"""
SIIP Dashboard — Page 3: Prediction Engine
============================================
Which ventures will succeed? How confident are we?
Classification + SHAP + Threshold Analysis
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import export_text

from utils.data_loader import (
    load_data, load_classification_results, load_shap_values,
    load_decision_tree, load_threshold_analysis,
    load_interpretations, load_prep_artifacts, display_name,
    MANDATES, get_mandate_insight, SWF_DIMENSIONS,
)
from utils.styles import (
    inject_css, kpi_card, section_header, metric_highlight, COLORS, PLOTLY_COLORS,
    swf_insight, mandate_badge, bridge_sentence,
)
from utils.charts import (
    apply_theme, roc_curves, confusion_matrix_chart,
    feature_importance_chart, bar_chart, line_chart,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prediction Engine | SIIP", layout="wide")
st.html(inject_css())

# ── Mandate ─────────────────────────────────────────────────────────────────
mandate = st.session_state.get('mandate', 'GIC — Financial Return')
mandate_config = MANDATES[mandate]

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()
clf_results = load_classification_results()
shap_data = load_shap_values()
tree_model = load_decision_tree()
threshold_analysis = load_threshold_analysis()
interp = load_interpretations()
prep = load_prep_artifacts()

st.title("Prediction Engine")
st.markdown("*Which ventures will succeed? How confident are we?*")
st.html(mandate_badge(mandate, mandate_config))
st.html(swf_insight(get_mandate_insight('prediction', mandate)))

# ═════════════════════════════════════════════════════════════════════════════
# SECTION A: The Challenge
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("A. The Challenge — Class Imbalance"))

class_counts = df['outcome_binary'].value_counts().sort_index()
total = len(df)
n_success = int(class_counts.get(1, 0))
n_failure = int(class_counts.get(0, 0))
success_rate = n_success / total if total > 0 else 0
class_ratio = f"{n_failure}:{n_success}"

col1, col2, col3 = st.columns(3)
with col1:
    st.html(kpi_card("Success Rate", f"{success_rate:.1%}"))
with col2:
    st.html(kpi_card("Class Ratio (Fail:Success)", class_ratio))
with col3:
    st.html(kpi_card("Total Ventures", f"{total:,}"))

# Class distribution bar chart
fig_class = bar_chart(
    x=['Failure (0)', 'Success (1)'],
    y=[n_failure, n_success],
    title="Class Distribution — outcome_binary",
    color=[COLORS['danger'], COLORS['success']],
    text=[f"{n_failure:,}", f"{n_success:,}"],
)
st.plotly_chart(fig_class, use_container_width=True)
st.caption(interp['classification']['imbalance'])

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION B: Model Arena
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("B. Model Arena — 6 Classifiers Compared"))

# ── Comparison table ─────────────────────────────────────────────────────────
model_names = list(clf_results.keys())
metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
table_data = []
for name in model_names:
    r = clf_results[name]
    table_data.append({
        'Model': name,
        'Accuracy': round(r['accuracy'], 4),
        'Precision': round(r['precision'], 4),
        'Recall': round(r['recall'], 4),
        'F1': round(r['f1'], 4),
        'AUC-ROC': round(r['auc_roc'], 4),
    })

comparison_df = pd.DataFrame(table_data).set_index('Model')

# Highlight best per metric
def highlight_best(s):
    is_max = s == s.max()
    return ['background-color: rgba(212, 175, 55, 0.25); font-weight: bold' if v else '' for v in is_max]

styled_df = comparison_df.style.apply(highlight_best, axis=0).format("{:.4f}")
st.dataframe(styled_df, use_container_width=True, height=260)
st.caption("Gold highlights indicate the best-performing model for each metric.")

# ── Prescriptive: Best model recommendation ──
best_model = max(clf_results, key=lambda k: clf_results[k]['auc_roc'])
best_auc = clf_results[best_model]['auc_roc']
best_recall = clf_results[best_model]['recall']
if mandate == "GIC — Financial Return":
    model_action = (
        f"<b>Model Selection (GIC):</b> Use <b>{best_model}</b> (AUC={best_auc:.3f}) as your primary screening model. "
        f"<b>Action:</b> Set a higher threshold to maximize precision — GIC prefers fewer, higher-confidence picks over broad coverage. "
        f"False positives waste diligence resources; optimize for return certainty."
    )
elif mandate == "PIF — Strategic Alignment":
    model_action = (
        f"<b>Model Selection (PIF):</b> Use <b>{best_model}</b> (AUC={best_auc:.3f}), but <b>prioritize recall ({best_recall:.1%})</b>. "
        f"<b>Action:</b> Lower the threshold to cast a wider net — missing a strategic winner costs more than investigating a few false positives. "
        f"A single transformative tech-transfer deal justifies 10 wasted diligence cycles."
    )
else:
    model_action = (
        f"<b>Model Selection (Mubadala):</b> Use <b>{best_model}</b> (AUC={best_auc:.3f}) with the F1-optimal threshold for balanced risk. "
        f"<b>Action:</b> Accept moderate false positive rates in exchange for not missing winners. "
        f"Combine this model's output with the Segmentation Lab to contextualize predictions by venture archetype."
    )
st.html(swf_insight(model_action))

# ── ROC curves ───────────────────────────────────────────────────────────────
fig_roc = roc_curves(clf_results)
st.plotly_chart(fig_roc, use_container_width=True)
st.caption("ROC curves for all six classifiers. The dashed line represents random chance (AUC = 0.5).")

# ── Per-model deep dive ─────────────────────────────────────────────────────
selected_model = st.selectbox("Select a model for detailed view", model_names, index=model_names.index('Gradient Boosting') if 'Gradient Boosting' in model_names else 0)

col_cm, col_fi = st.columns(2)

with col_cm:
    cm = clf_results[selected_model]['confusion_matrix']
    fig_cm = confusion_matrix_chart(cm, title=f"Confusion Matrix — {selected_model}")
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption(f"Confusion matrix for {selected_model}. Rows = actual, columns = predicted.")

with col_fi:
    importances = clf_results[selected_model].get('feature_importances', {})
    if importances:
        feature_names_list = list(importances.keys())
        display_names = [display_name(f) for f in feature_names_list]
        # Rebuild importances dict with display names
        display_importances = dict(zip(display_names, importances.values()))
        fig_fi = feature_importance_chart(
            display_importances, display_names,
            title=f"Feature Importances — {selected_model}"
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption(f"Feature importance scores from {selected_model}. Gold bars exceed the mean importance.")
    else:
        st.info(f"{selected_model} does not provide feature importances.")

st.caption(interp['classification']['best_model'])

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION C: Interpretability
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("C. Interpretability — Decision Tree + SHAP"))

col_tree, col_shap = st.columns(2)

# ── Decision tree rules ──────────────────────────────────────────────────────
with col_tree:
    st.subheader("Decision Tree Rules")
    feature_names_for_tree = prep.get('ivs', shap_data.get('feature_names', []))
    tree_display_names = [display_name(f) for f in feature_names_for_tree]
    tree_text = export_text(tree_model, feature_names=tree_display_names, max_depth=4)
    st.code(tree_text, language="text")
    st.caption("Decision tree rules (depth limited to 4 for readability). Each leaf shows class counts.")

# ── SHAP summary ─────────────────────────────────────────────────────────────
with col_shap:
    st.subheader("SHAP Feature Impact")
    shap_values_arr = np.array(shap_data['shap_values'])
    shap_feature_names = shap_data['feature_names']

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=0)

    # Sort descending
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_names = [display_name(shap_feature_names[i]) for i in sorted_idx]
    sorted_vals = [float(mean_abs_shap[i]) for i in sorted_idx]

    fig_shap = bar_chart(
        x=sorted_names, y=sorted_vals,
        title="Mean |SHAP Value| per Feature",
        color=COLORS['accent'],
        height=max(350, len(sorted_names) * 22),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption("Mean absolute SHAP values quantify each feature's average impact on model predictions.")

st.caption(interp['classification']['shap'])

# ── Prescriptive: SHAP-driven due diligence ──
top3_shap = sorted_names[:3]
st.html(swf_insight(
    f"<b>Due Diligence Priority:</b> The top 3 predictive drivers are <b>{', '.join(top3_shap)}</b>. "
    f"These features have the most influence on whether a venture is classified as a success or failure. "
    f"<b>Action:</b> Frontload IC review on these 3 dimensions. Request detailed data on each from the management team. "
    f"If a venture scores poorly on 2+ of these top drivers, it's a high-risk bet regardless of other metrics."
))

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION D: Threshold Optimization
# ═════════════════════════════════════════════════════════════════════════════
st.html(section_header("D. Threshold Optimization — Cost-Sensitive Learning"))

thresholds_data = threshold_analysis['thresholds']
optimal = threshold_analysis['optimal']

# Extract lists for charting
t_vals = [t['threshold'] for t in thresholds_data]
t_precision = [t['precision'] for t in thresholds_data]
t_recall = [t['recall'] for t in thresholds_data]
t_f1 = [t['f1'] for t in thresholds_data]

# Threshold slider
selected_threshold = st.slider(
    "Classification Threshold",
    min_value=0.05, max_value=0.95, value=float(optimal['threshold']),
    step=0.05, format="%.2f"
)

col_chart, col_metrics = st.columns([2, 1])

with col_chart:
    # Line chart: precision/recall/F1 vs threshold
    fig_thresh = line_chart(
        x=t_vals,
        y={'Precision': t_precision, 'Recall': t_recall, 'F1': t_f1},
        title="Precision / Recall / F1 vs. Threshold",
        height=420,
    )
    # Add vertical line at selected threshold
    fig_thresh.add_vline(
        x=selected_threshold, line_dash="dash",
        line_color=COLORS['accent'], opacity=0.8,
        annotation_text=f"t = {selected_threshold:.2f}",
        annotation_font_color=COLORS['accent'],
    )
    st.plotly_chart(fig_thresh, use_container_width=True)
    st.caption("Trade-off curves across thresholds. The dashed gold line marks the selected threshold.")

with col_metrics:
    # Find closest threshold entry
    closest_idx = int(np.argmin([abs(t - selected_threshold) for t in t_vals]))
    sel = thresholds_data[closest_idx]

    st.html(metric_highlight(f"Threshold: <strong>{sel['threshold']:.2f}</strong>"))

    m1, m2 = st.columns(2)
    with m1:
        st.html(kpi_card("Precision", f"{sel['precision']:.3f}"))
        st.html(kpi_card("Recall", f"{sel['recall']:.3f}"))
    with m2:
        st.html(kpi_card("F1 Score", f"{sel['f1']:.3f}"))
        st.html(kpi_card("Predicted Positive", f"{sel['n_predicted_positive']:,}"))

    # Optimal threshold callout
    st.html(
        metric_highlight(
            f"Optimal threshold: <strong>{optimal['threshold']:.2f}</strong> "
            f"(F1 = {optimal['f1']:.3f})"
        ))

st.caption(interp['classification']['threshold'])

# ── Prescriptive: Threshold action ──
opt_thresh = optimal['threshold']
opt_f1 = optimal['f1']
# Find recall/precision at optimal
closest = min(thresholds_data, key=lambda t: abs(t['threshold'] - opt_thresh))
st.html(swf_insight(
    f"<b>Recommended Threshold:</b> At <b>t={opt_thresh:.2f}</b> (optimal F1={opt_f1:.3f}), "
    f"precision is <b>{closest['precision']:.1%}</b> and recall is <b>{closest['recall']:.1%}</b>. "
    f"This means: of every 100 ventures flagged as 'likely winners', <b>{closest['precision']*100:.0f} actually succeed</b>, "
    f"and you catch <b>{closest['recall']*100:.0f}%</b> of all winners. "
    f"<b>Action:</b> Use this threshold as your default cutoff. Adjust down for strategic mandates (catch more), up for return mandates (higher precision)."
))

st.info(
    "**Cost framing:** False negatives (missing winners) cost more than false positives. "
    "A sovereign wealth fund that passes on the next transformative venture loses far more "
    "than the due-diligence cost of investigating a false positive."
)
