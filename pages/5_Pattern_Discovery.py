"""
SIIP — Pattern Discovery (Page 5)
===================================
What feature combinations predict success or failure?
Association Rule Mining (Apriori)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import (
    load_data, load_arm_rules, load_interpretations,
    MANDATES, get_mandate_insight, display_name, SWF_DIMENSIONS,
)
from utils.styles import (
    inject_css, kpi_card, section_header, metric_highlight,
    swf_insight, mandate_badge, bridge_sentence, COLORS, PLOTLY_COLORS,
)
from utils.charts import apply_theme, bar_chart, scatter_chart

st.set_page_config(page_title="Pattern Discovery | SIIP", layout="wide")
st.html(inject_css())

# --- Mandate ---
mandate = st.session_state.get('mandate', 'GIC — Financial Return')
mandate_config = MANDATES[mandate]

# --- Load data ---
df = load_data()
arm = load_arm_rules()
interp = load_interpretations()

all_rules = arm['all_rules']
outcome_rules = arm['outcome_rules']
success_rules = arm['success_rules']
failure_rules = arm['failure_rules']
freq_itemsets = arm['freq_itemsets']
disc_thresholds = arm['discretization_thresholds']

st.markdown("# Pattern Discovery")
st.caption("What feature combinations predict success or failure? Association Rule Mining via the Apriori algorithm.")
st.html(mandate_badge(mandate, mandate_config))
st.html(swf_insight(get_mandate_insight('pattern', mandate)))
st.markdown("---")

# =====================================================================
# SECTION A — Data Transformation (Discretization)
# =====================================================================
st.html(section_header("A. Data Transformation (Evidence: Discretization)"))

st.markdown(
    "Continuous features were discretized into **tercile bins** (Low / Med / High) based on the 33rd and 67th "
    "percentiles. This converts numeric data into categorical items suitable for association rule mining."
)

col_thresh, col_freq = st.columns([1, 1])

with col_thresh:
    st.markdown("##### Discretization Thresholds")
    thresh_rows = []
    for feat, edges in disc_thresholds.items():
        thresh_rows.append({
            'Feature': feat,
            'Bin Edges': ', '.join([f"{e:.2f}" for e in edges]) if isinstance(edges, list) else str(edges)
        })
    thresh_df = pd.DataFrame(thresh_rows)
    st.dataframe(thresh_df, use_container_width=True, hide_index=True, height=min(400, len(thresh_rows) * 40 + 40))
    st.caption("Tercile bin edges used to convert continuous features into Low/Med/High categories.")

with col_freq:
    st.markdown("##### Top 20 Frequent Items")
    # Sort frequent itemsets by support descending, take top 20
    sorted_itemsets = sorted(freq_itemsets, key=lambda x: x['support'], reverse=True)[:20]
    item_labels = []
    item_supports = []
    for fi in sorted_itemsets:
        label = ', '.join(fi['itemset']) if isinstance(fi['itemset'], list) else str(fi['itemset'])
        item_labels.append(label)
        item_supports.append(fi['support'])

    fig_freq = go.Figure(data=[go.Bar(
        y=item_labels[::-1],
        x=item_supports[::-1],
        orientation='h',
        marker_color=COLORS['accent2'],
        hovertemplate='%{y}: support=%{x:.4f}<extra></extra>'
    )])
    fig_freq.update_layout(
        xaxis_title='Support',
        title='Most Frequent Items (by Support)'
    )
    fig_freq = apply_theme(fig_freq, 450)
    st.plotly_chart(fig_freq, use_container_width=True)
    st.caption("Top 20 most frequently occurring items across all transactions in the discretized dataset.")

st.caption(interp.get('arm', {}).get('overview', ''))
st.markdown("")

# =====================================================================
# SECTION B — Rule Explorer (Apriori Algorithm)
# =====================================================================
st.markdown("---")
st.html(section_header("B. Rule Explorer (Evidence: Apriori Algorithm)"))

# KPI row
avg_lift = np.mean([r['lift'] for r in all_rules]) if all_rules else 0
avg_conf = np.mean([r['confidence'] for r in all_rules]) if all_rules else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.html(kpi_card("Total Rules", f"{arm['n_rules']:,}"))
with k2:
    st.html(kpi_card("Outcome Rules", f"{arm['n_outcome_rules']:,}"))
with k3:
    st.html(kpi_card("Avg Lift", f"{avg_lift:.2f}"))
with k4:
    st.html(kpi_card("Avg Confidence", f"{avg_conf:.1%}"))

st.markdown("")

# Filters
fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
with fcol1:
    min_support = st.slider("Min Support", 0.0, 1.0, 0.01, 0.01, key="arm_support")
with fcol2:
    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.1, 0.05, key="arm_confidence")
with fcol3:
    consequent_filter = st.selectbox("Consequent Filter", ["All Rules", "Outcome Rules Only"], key="arm_filter")

# Build filtered rules table
if consequent_filter == "Outcome Rules Only":
    display_rules = outcome_rules
else:
    display_rules = all_rules

filtered_rules = [
    r for r in display_rules
    if r['support'] >= min_support and r['confidence'] >= min_confidence
]

rules_table = pd.DataFrame([
    {
        'Antecedents': r.get('antecedents_str', ' + '.join(r['antecedents'])),
        'Consequents': r.get('consequents_str', ' + '.join(r['consequents'])),
        'Support': round(r['support'], 4),
        'Confidence': round(r['confidence'], 4),
        'Lift': round(r['lift'], 4),
    }
    for r in filtered_rules
])

if not rules_table.empty:
    rules_table = rules_table.sort_values('Lift', ascending=False).reset_index(drop=True)
    st.markdown(f"##### Association Rules ({len(rules_table):,} rules matching filters)")
    st.dataframe(rules_table, use_container_width=True, hide_index=True, height=450)
else:
    st.info("No rules match the current filter settings. Try lowering the thresholds.")

st.caption(
    "Rules are generated via the Apriori algorithm. **Support** = frequency of itemset; "
    "**Confidence** = P(consequent | antecedent); **Lift** > 1 indicates positive association."
)
st.markdown("")

# =====================================================================
# SECTION C — Support vs Confidence Scatter
# =====================================================================
st.markdown("---")
st.html(section_header("C. Support vs Confidence Scatter"))

col_scatter, col_top = st.columns([2, 1])

with col_scatter:
    if all_rules:
        scatter_df = pd.DataFrame([
            {
                'Support': r['support'],
                'Confidence': r['confidence'],
                'Lift': r['lift'],
                'Is Outcome Rule': 'outcome_' in ' '.join(r['consequents']),
                'Rule': f"{r.get('antecedents_str', ' + '.join(r['antecedents']))} => "
                        f"{r.get('consequents_str', ' + '.join(r['consequents']))}"
            }
            for r in all_rules
        ])
        scatter_df['Type'] = scatter_df['Is Outcome Rule'].map({True: 'Outcome Rule', False: 'General Rule'})

        fig_scatter = px.scatter(
            scatter_df, x='Support', y='Confidence',
            size='Lift', color='Type',
            color_discrete_map={'Outcome Rule': COLORS['accent'], 'General Rule': COLORS['accent2']},
            hover_data=['Rule', 'Lift'],
            title='Rule Landscape: Support vs Confidence'
        )
        fig_scatter.update_traces(marker=dict(opacity=0.7))
        fig_scatter = apply_theme(fig_scatter, 450)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Each point is one association rule. Size encodes lift; gold points target outcome variables.")
    else:
        st.info("No rules available for scatter plot.")

with col_top:
    st.markdown("##### Top 10 Rules by Lift")
    top_rules = sorted(all_rules, key=lambda x: x['lift'], reverse=True)[:10]
    top_df = pd.DataFrame([
        {
            'Rule': f"{r.get('antecedents_str', ' + '.join(r['antecedents']))} => "
                    f"{r.get('consequents_str', ' + '.join(r['consequents']))}",
            'Lift': round(r['lift'], 3),
            'Conf': round(r['confidence'], 3),
        }
        for r in top_rules
    ])
    if not top_df.empty:
        st.dataframe(top_df, use_container_width=True, hide_index=True, height=420)
    st.caption("Highest-lift rules represent the strongest statistical associations in the data.")

st.caption(interp.get('arm', {}).get('success_patterns', ''))
st.markdown("")

# =====================================================================
# SECTION D — Screening Patterns
# =====================================================================
st.markdown("---")
st.html(section_header("D. Screening Patterns — Outcome-Linked Rules"))

col_success, col_failure = st.columns(2)

with col_success:
    st.markdown("##### Success Patterns")
    if success_rules:
        success_sorted = sorted(success_rules, key=lambda x: x['lift'], reverse=True)
        success_df = pd.DataFrame([
            {
                'Antecedents': r.get('antecedents_str', ' + '.join(r['antecedents'])),
                'Consequent': r.get('consequents_str', ' + '.join(r['consequents'])),
                'Lift': round(r['lift'], 3),
                'Confidence': round(r['confidence'], 3),
                'Support': round(r['support'], 4),
            }
            for r in success_sorted[:15]
        ])
        st.dataframe(success_df, use_container_width=True, hide_index=True, height=400)
        st.caption("Feature combinations most associated with successful outcomes, ranked by lift.")

        # ── Prescriptive: Success screening checklist ──
        top3_success = success_sorted[:3]
        checklist_items = []
        for r in top3_success:
            ant = r.get('antecedents_str', ' + '.join(r['antecedents']))
            checklist_items.append(f"<b>{ant}</b> (lift={r['lift']:.2f}, conf={r['confidence']:.0%})")
        st.html(swf_insight(
            f"<b>Positive Screening Checklist:</b> New deals matching any of these patterns have historically "
            f"higher success rates:<br>" + "<br>".join([f"&bull; {item}" for item in checklist_items]) +
            f"<br><b>Action:</b> Add these as 'green flags' in your deal screening template. "
            f"Ventures matching 2+ patterns should be fast-tracked to IC."
        ))
    else:
        st.html(metric_highlight(
            "No rules with <b>outcome_Success</b> as consequent were found at current thresholds. "
            "This may indicate success is driven by complex multi-factor interactions not captured at this support level."
        ))

with col_failure:
    st.markdown("##### Failure / Write-off Patterns")
    if failure_rules:
        failure_sorted = sorted(failure_rules, key=lambda x: x['lift'], reverse=True)
        failure_df = pd.DataFrame([
            {
                'Antecedents': r.get('antecedents_str', ' + '.join(r['antecedents'])),
                'Consequent': r.get('consequents_str', ' + '.join(r['consequents'])),
                'Lift': round(r['lift'], 3),
                'Confidence': round(r['confidence'], 3),
                'Support': round(r['support'], 4),
            }
            for r in failure_sorted[:15]
        ])
        st.dataframe(failure_df, use_container_width=True, hide_index=True, height=400)
        st.caption("Feature combinations most associated with write-off / failure outcomes, ranked by lift.")

        # ── Prescriptive: Failure red-flag checklist ──
        top3_failure = failure_sorted[:3]
        redflag_items = []
        for r in top3_failure:
            ant = r.get('antecedents_str', ' + '.join(r['antecedents']))
            redflag_items.append(f"<b>{ant}</b> (lift={r['lift']:.2f}, conf={r['confidence']:.0%})")
        st.html(swf_insight(
            f"<b>Red-Flag Screening List:</b> These patterns are the strongest predictors of write-offs:<br>" +
            "<br>".join([f"&bull; {item}" for item in redflag_items]) +
            f"<br><b>Action:</b> Any new deal matching these patterns should be automatically flagged for enhanced diligence. "
            f"Block deals matching 2+ red-flag patterns from proceeding without senior IC override."
        ))
    else:
        st.html(metric_highlight(
            "No rules with <b>outcome_Write-off</b> as consequent were found at current thresholds. "
            "Failure patterns may require lower support thresholds to surface."
        ))

st.markdown("")

# Outcome rules overview
if outcome_rules:
    st.markdown("##### All Outcome-Linked Rules (sorted by Lift)")
    outcome_sorted = sorted(outcome_rules, key=lambda x: x['lift'], reverse=True)
    outcome_df = pd.DataFrame([
        {
            'Antecedents': r.get('antecedents_str', ' + '.join(r['antecedents'])),
            'Consequent': r.get('consequents_str', ' + '.join(r['consequents'])),
            'Lift': round(r['lift'], 3),
            'Confidence': round(r['confidence'], 3),
            'Support': round(r['support'], 4),
        }
        for r in outcome_sorted
    ])
    st.dataframe(outcome_df, use_container_width=True, hide_index=True, height=350)
    st.caption("Complete set of rules where the consequent contains an outcome category. These rules reveal which feature combinations screen for specific outcomes.")

# Anti-pattern alerts
st.markdown("")
st.markdown("##### Anti-Pattern Alerts")
anti_patterns_text = interp.get('arm', {}).get('anti_patterns', '')
if anti_patterns_text:
    st.warning(anti_patterns_text)
else:
    # Surface high-lift failure rules as anti-patterns
    if failure_rules:
        st.warning(
            "The following feature combinations are strong predictors of failure/write-off. "
            "Screen deals exhibiting these patterns with heightened diligence."
        )
        for r in sorted(failure_rules, key=lambda x: x['lift'], reverse=True)[:5]:
            antecedent_str = r.get('antecedents_str', ' + '.join(r['antecedents']))
            st.markdown(f"- **{antecedent_str}** (lift={r['lift']:.2f}, confidence={r['confidence']:.1%})")
    else:
        st.info("No anti-patterns identified at current mining thresholds.")

st.caption(interp.get('arm', {}).get('anti_patterns', ''))

# ── Prescriptive: Mandate-specific pattern guidance ──
if mandate == "GIC — Financial Return":
    pattern_action = (
        "<b>Pattern Application (GIC):</b> Focus on rules involving financial features — revenue growth, "
        "capital raised, valuation, margins. Ignore rules driven primarily by strategic or ESG features for this mandate. "
        "<b>Action:</b> Extract the top 5 financial-only rules and encode them into your automated deal screening pipeline."
    )
elif mandate == "PIF — Strategic Alignment":
    pattern_action = (
        "<b>Pattern Application (PIF):</b> Prioritize rules containing bilateral alignment, tech transfer, or "
        "national strategy features — these are your Vision 2030 screening patterns. "
        "<b>Action:</b> Even if these rules have lower lift than financial rules, they serve PIF's mandate. "
        "Cross-reference with sector filters to find strategic sector + feature combinations."
    )
else:
    pattern_action = (
        "<b>Pattern Application (Mubadala):</b> The most robust rules span multiple SWF dimensions — "
        "they combine financial, strategic, and risk features. "
        "<b>Action:</b> Prioritize cross-dimensional rules (lift > 1.5) over single-dimension rules. "
        "These multi-factor patterns capture the balanced risk-return profile Mubadala targets."
    )
st.html(swf_insight(pattern_action))
