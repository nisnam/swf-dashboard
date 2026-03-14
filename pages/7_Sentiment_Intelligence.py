"""
SIIP Dashboard — Page 7: Sentiment Intelligence
==================================================
NLP-based sentiment analysis of investment analyst memos.
Three methods: Lexicon (TextBlob), VADER, Naive Bayes.
Head-to-head comparisons and sentiment visualizations.
Mandate-aware framing and disagreement detection.
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

from utils.data_loader import (
    load_data, load_sentiment_results,
    MANDATES, get_mandate_insight,
)
from utils.styles import (
    inject_css, section_header, metric_highlight,
    kpi_card, swf_insight, mandate_badge, bridge_sentence,
    COLORS, PLOTLY_COLORS,
)
from utils.charts import apply_theme, donut_chart, bar_chart

st.set_page_config(
    page_title="SIIP — Sentiment Intelligence",
    page_icon="💬",
    layout="wide"
)

st.html(inject_css())

# --- Mandate ---
mandate = st.session_state.get('mandate', 'GIC — Financial Return')
mandate_cfg = MANDATES[mandate]

# --- Load data ---
df = load_data()
sentiment = load_sentiment_results()

# --- Header ---
st.markdown("# Sentiment Intelligence")
st.caption("Quantifying analyst conviction from investment memos — an NLP layer for SWF deal screening")
st.html(mandate_badge(mandate, mandate_cfg))
st.html(swf_insight(get_mandate_insight('sentiment', mandate)))

# ================================================================
# SECTION A: METHODOLOGY & KPIs
# ================================================================
st.html(section_header("Methodology: Text Generation & Analysis Pipeline"))

st.html(metric_highlight(
    "<strong>SWF Application:</strong> Sovereign wealth funds process thousands of analyst memos, IC reports, and due diligence notes. "
    "Sentiment analysis automates the extraction of analyst conviction signals from unstructured text — flagging ventures where qualitative "
    "assessments diverge from quantitative metrics and enabling systematic screening at scale."
))

with st.expander("How are analyst memos generated?", expanded=False):
    st.markdown("""
    **Investment Analyst Memos** are synthetically generated for each of the 6,798 ventures based on their actual feature values:
    - **Financial metrics** (revenue growth, margins, burn rate) → positive/negative financial language
    - **Risk indicators** (country risk, competitive intensity, sanctions) → risk assessment language
    - **Strategic factors** (bilateral alignment, ESG, IP portfolio) → strategic positioning language
    - **Outcome alignment** → analyst conclusion partially reflects actual investment outcome (with noise to simulate real-world uncertainty)

    This mirrors real-world SWF practice where analyst notes accompany quantitative scoring.
    Each memo is analyzed by **three independent sentiment methods** to demonstrate different NLP approaches.

    | Method | Type | How It Works | SWF Use Case |
    |--------|------|-------------|--------------|
    | **TextBlob** | Lexicon-based | Scores words against a pre-built polarity dictionary | Quick screening of new memos |
    | **VADER** | Rule-based | Handles financial language, negations, intensifiers | Real-time sentiment monitoring |
    | **Naive Bayes** | Supervised ML | Trained on labeled outcome data using TF-IDF features | Predictive sentiment classification |
    """)

st.markdown("")

# --- Mandate framing after methodology ---
mandate_framing = {
    'GIC — Financial Return': (
        "For a **financial-return-focused** fund like GIC, sentiment analysis is most valuable when it captures "
        "analyst language about **margins, growth trajectories, and exit potential**. Pay particular attention to "
        "ventures where positive financial sentiment aligns with strong quantitative metrics — and flag those where "
        "analyst optimism is not backed by the numbers."
    ),
    'PIF — Strategic Alignment': (
        "For a **strategy-focused** fund like PIF, sentiment analysis should emphasize language around **technology transfer, "
        "bilateral relations, domestic capability building, and Vision 2030 alignment**. Analyst conviction about strategic "
        "value may diverge from pure financial metrics — and that divergence is the signal."
    ),
    'Mubadala — Balanced Portfolio': (
        "For a **balanced** fund like Mubadala, sentiment analysis serves as a **holistic cross-check** on quantitative scores. "
        "The most actionable signal comes from disagreements: ventures where analyst language is strongly positive but models "
        "flag risk, or vice versa. These are the deals that benefit most from deeper human review."
    ),
}

st.html(metric_highlight(mandate_framing.get(mandate, '')))

# KPI Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.html(kpi_card("Analyst Memos", f"{len(sentiment['memos']):,}", "NLP-analyzed"))
with k2:
    vader_dist = sentiment['distributions']['vader']
    pct_pos_vader = vader_dist['Positive'] / sum(vader_dist.values())
    st.html(kpi_card("VADER Positive", f"{pct_pos_vader:.1%}", f"{vader_dist['Positive']:,} memos"))
with k3:
    nb_acc = sentiment['naive_bayes']['cv_accuracy']
    st.html(kpi_card("NB Accuracy", f"{nb_acc:.1%}", f"5-fold CV"))
with k4:
    h2h = sentiment['head_to_head']
    avg_agreement = np.mean([v for k, v in h2h['agreement_matrix']['VADER'].items() if k != 'VADER'])
    st.html(kpi_card("Avg Agreement", f"{avg_agreement:.1%}", "across methods"))

st.markdown("")

# ================================================================
# SECTION B: SENTIMENT DISTRIBUTIONS (3 Methods)
# ================================================================
st.html(section_header("Sentiment Distribution — Three Methods Compared"))

col1, col2, col3 = st.columns(3)

color_map = {'Positive': COLORS['success'], 'Neutral': COLORS['warning'], 'Negative': COLORS['danger']}

for col, (method, dist) in zip([col1, col2, col3],
    [('VADER', sentiment['distributions']['vader']),
     ('TextBlob (Lexicon)', sentiment['distributions']['textblob']),
     ('Naive Bayes', sentiment['distributions']['naive_bayes'])]):
    with col:
        st.markdown(f"**{method}**")
        labels = list(dist.keys())
        values = list(dist.values())
        colors_list = [color_map.get(l, COLORS['accent2']) for l in labels]
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.5,
            marker_colors=colors_list,
            textinfo='label+percent', textfont_size=11
        )])
        fig.update_layout(showlegend=False)
        fig = apply_theme(fig, 280)
        st.plotly_chart(fig, use_container_width=True)
        total = sum(values)
        st.caption(f"Positive: {dist['Positive']/total:.1%} | Neutral: {dist['Neutral']/total:.1%} | Negative: {dist['Negative']/total:.1%}")

st.info("**Interpretation:** VADER and TextBlob use unsupervised lexicon-based scoring (no training needed) — ideal for screening new ventures in real-time. Naive Bayes is a supervised classifier trained on outcome-aligned sentiment labels — better for retrospective analysis. Distribution differences reveal each method's sensitivity to financial language and risk terminology used in SWF contexts.")

# ================================================================
# SECTION C: SENTIMENT SCORE DISTRIBUTIONS
# ================================================================
st.markdown("---")
st.html(section_header("Sentiment Score Distributions"))

vader_scores = sentiment['scores']['vader_compound']
textblob_scores = sentiment['scores']['textblob_polarity']

col1, col2 = st.columns(2)

with col1:
    st.markdown("**VADER Compound Score Distribution**")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vader_scores, nbinsx=50,
        marker_color=COLORS['accent2'], opacity=0.8,
        name='VADER'
    ))
    fig.update_layout(
        xaxis_title='Compound Score (-1 to +1)',
        yaxis_title='Count',
        title='VADER Sentiment Scores'
    )
    fig = apply_theme(fig, 350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Mean: {np.mean(vader_scores):.3f} | Median: {np.median(vader_scores):.3f} | Std: {np.std(vader_scores):.3f}")

with col2:
    st.markdown("**TextBlob Polarity Score Distribution**")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=textblob_scores, nbinsx=50,
        marker_color=COLORS['accent'], opacity=0.8,
        name='TextBlob'
    ))
    fig.update_layout(
        xaxis_title='Polarity Score (-1 to +1)',
        yaxis_title='Count',
        title='TextBlob Polarity Scores'
    )
    fig = apply_theme(fig, 350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Mean: {np.mean(textblob_scores):.3f} | Median: {np.median(textblob_scores):.3f} | Std: {np.std(textblob_scores):.3f}")

# VADER vs TextBlob scatter
st.markdown("**VADER vs TextBlob — Score Correlation**")
actual_labels = sentiment['classifications']['actual']
fig = go.Figure()
for label, color in [('Positive', COLORS['success']), ('Neutral', COLORS['warning']), ('Negative', COLORS['danger'])]:
    mask = [i for i, l in enumerate(actual_labels) if l == label]
    fig.add_trace(go.Scatter(
        x=[vader_scores[i] for i in mask],
        y=[textblob_scores[i] for i in mask],
        mode='markers', name=label,
        marker=dict(color=color, size=4, opacity=0.4),
    ))
fig.update_layout(
    xaxis_title='VADER Compound', yaxis_title='TextBlob Polarity',
    title='VADER vs TextBlob Scores (colored by actual outcome sentiment)'
)
fig = apply_theme(fig, 400)
st.plotly_chart(fig, use_container_width=True)
st.caption("Each point is one venture. Strong diagonal correlation means the methods agree; dispersion reveals where they diverge.")

# ================================================================
# SECTION D: SENTIMENT BY OUTCOME, SECTOR, STAGE
# ================================================================
st.markdown("---")
st.html(section_header("Sentiment Breakdown — By Outcome, Sector & Stage"))

tab1, tab2, tab3 = st.tabs(["By Outcome", "By Sector", "By Funding Stage"])

with tab1:
    outcome_data = sentiment['breakdowns']['by_outcome']
    outcomes = [d['investment_outcome'] for d in outcome_data]
    vader_means = [d['vader_mean'] for d in outcome_data]
    textblob_means = [d['textblob_mean'] for d in outcome_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=outcomes, y=vader_means, name='VADER', marker_color=COLORS['accent2']))
    fig.add_trace(go.Bar(x=outcomes, y=textblob_means, name='TextBlob', marker_color=COLORS['accent']))
    fig.update_layout(barmode='group', title='Mean Sentiment Score by Investment Outcome',
                      yaxis_title='Mean Sentiment Score')
    fig = apply_theme(fig, 400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Successful exits should show higher sentiment scores if the NLP methods are capturing meaningful signal from the analyst memos. The gap between VADER and TextBlob reveals their different sensitivities to financial language.")

with tab2:
    sector_data = sentiment['breakdowns']['by_sector']
    sectors = [d['sector'] for d in sector_data]
    vader_s = [d['vader_mean'] for d in sector_data]
    textblob_s = [d['textblob_mean'] for d in sector_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=sectors, x=vader_s, name='VADER', orientation='h', marker_color=COLORS['accent2']))
    fig.add_trace(go.Bar(y=sectors, x=textblob_s, name='TextBlob', orientation='h', marker_color=COLORS['accent']))
    fig.update_layout(barmode='group', title='Mean Sentiment by Sector',
                      xaxis_title='Mean Sentiment Score')
    fig = apply_theme(fig, 500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Sector-level sentiment variation reflects the different risk and growth profiles across industries. Sectors with stronger growth metrics generate more positive analyst language.")

with tab3:
    stage_data = sentiment['breakdowns']['by_stage']
    stages = [d['funding_stage'] for d in stage_data]
    vader_st = [d['vader_mean'] for d in stage_data]
    textblob_st = [d['textblob_mean'] for d in stage_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=stages, y=vader_st, name='VADER', marker_color=COLORS['accent2']))
    fig.add_trace(go.Bar(x=stages, y=textblob_st, name='TextBlob', marker_color=COLORS['accent']))
    fig.update_layout(barmode='group', title='Mean Sentiment by Funding Stage',
                      yaxis_title='Mean Sentiment Score')
    fig = apply_theme(fig, 400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Later-stage ventures tend to generate more positive analyst sentiment — reflecting stronger financials, clearer exit paths, and more established operations.")

# ================================================================
# SECTION E: NAIVE BAYES DEEP DIVE
# ================================================================
st.markdown("---")
st.html(section_header("Naive Bayes Classifier — Deep Dive"))

nb_data = sentiment['naive_bayes']

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Model Performance**")
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | **CV Accuracy** | {nb_data['cv_accuracy']:.1%} |
    | **CV Std Dev** | {nb_data['cv_std']:.3f} |
    | **Classes** | {', '.join(nb_data['classes'])} |
    | **Features** | TF-IDF (500 terms, 1-2 ngrams) |
    """)

    # NB confusion-style: predicted distribution
    nb_dist = sentiment['distributions']['naive_bayes']
    st.markdown("**Predicted Distribution:**")
    for cls, count in nb_dist.items():
        pct = count / sum(nb_dist.values())
        st.markdown(f"- {cls}: {count:,} ({pct:.1%})")

with col2:
    st.markdown("**Top Predictive Words by Class**")
    class_select = st.selectbox("Select sentiment class:", nb_data['classes'])
    if class_select in nb_data['top_words_per_class']:
        words_data = nb_data['top_words_per_class'][class_select]
        words = [w[0] for w in words_data]
        scores = [w[1] for w in words_data]

        fig = go.Figure(data=[go.Bar(
            y=words[::-1], x=scores[::-1], orientation='h',
            marker_color=color_map.get(class_select, COLORS['accent2'])
        )])
        fig.update_layout(title=f'Top TF-IDF Features — {class_select}',
                          xaxis_title='Log Probability')
        fig = apply_theme(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

st.caption("Naive Bayes learns which words and phrases are most associated with each sentiment class from the training data. Higher log probability = stronger association with that class.")

# ================================================================
# SECTION F: HEAD-TO-HEAD COMPARISONS
# ================================================================
st.markdown("---")
st.html(section_header("Head-to-Head: Method Comparison"))

h2h = sentiment['head_to_head']

# Agreement Matrix Heatmap
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Agreement Matrix**")
    methods = list(h2h['agreement_matrix'].keys())
    z_values = [[h2h['agreement_matrix'][m1][m2] for m2 in methods] for m1 in methods]

    fig = go.Figure(data=go.Heatmap(
        z=z_values, x=methods, y=methods,
        colorscale='RdYlGn', texttemplate='%{z:.1%}',
        zmin=0, zmax=1,
        hovertemplate='%{x} vs %{y}: %{z:.1%} agreement<extra></extra>'
    ))
    fig.update_layout(title='Pairwise Agreement Rate')
    fig = apply_theme(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Cohen's Kappa (Inter-rater Reliability)**")
    kappa_data = h2h['kappa_scores']
    pairs = list(kappa_data.keys())
    kappas = list(kappa_data.values())

    fig = go.Figure(data=[go.Bar(
        x=pairs, y=kappas,
        marker_color=[COLORS['success'] if k > 0.4 else COLORS['warning'] if k > 0.2 else COLORS['danger'] for k in kappas],
        text=[f"{k:.3f}" for k in kappas], textposition='auto'
    )])
    fig.update_layout(title="Cohen's Kappa Scores", yaxis_title='Kappa',
                      yaxis_range=[0, max(kappas) * 1.3 if kappas else 1])
    fig = apply_theme(fig, 350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Kappa > 0.6 = substantial agreement, 0.4-0.6 = moderate, 0.2-0.4 = fair, < 0.2 = slight")

# Performance vs Actual Labels
st.markdown("**Performance vs Actual Outcome-Derived Labels**")

perf = h2h['method_performance']
perf_df = pd.DataFrame({
    'Method': list(perf.keys()),
    'Accuracy': [perf[m]['accuracy'] for m in perf],
    'Macro F1': [perf[m]['macro_f1'] for m in perf],
    'Weighted F1': [perf[m]['weighted_f1'] for m in perf],
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(perf_df.style.format({
        'Accuracy': '{:.1%}', 'Macro F1': '{:.1%}', 'Weighted F1': '{:.1%}'
    }), hide_index=True, use_container_width=True)

with col2:
    fig = go.Figure()
    for metric, color in [('Accuracy', COLORS['accent2']), ('Macro F1', COLORS['accent']), ('Weighted F1', COLORS['success'])]:
        fig.add_trace(go.Bar(x=perf_df['Method'], y=perf_df[metric], name=metric, marker_color=color))
    fig.update_layout(barmode='group', title='Method Performance Comparison',
                      yaxis_title='Score', yaxis_tickformat='.0%')
    fig = apply_theme(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

# ── Prescriptive: Agreement interpretation ──
avg_agree = np.mean([v for k, v in h2h['agreement_matrix']['VADER'].items() if k != 'VADER'])
if avg_agree > 0.7:
    agree_action = "High inter-method agreement — sentiment signal is reliable. Use any single method for screening."
elif avg_agree > 0.4:
    agree_action = "Moderate agreement — use VADER as primary, TextBlob as tiebreaker. Where all 3 disagree, escalate to human review."
else:
    agree_action = "Low agreement — methods capture different signals. Require 2-of-3 consensus before acting on sentiment."
st.html(swf_insight(
    f"<b>Method Reliability:</b> Average cross-method agreement is <b>{avg_agree:.0%}</b>. {agree_action}"
))

st.info("""**Key Finding — SWF Decision Context:** Naive Bayes achieves the highest accuracy because it is trained on outcome-derived labels — it learns which language patterns correlate with success vs failure. However, **for live deal screening, unsupervised methods (VADER, TextBlob) are more practical** because they do not require historical outcome labels to function.

**Recommended SWF Workflow:** Use VADER for real-time memo screening (fast, no training needed), TextBlob for a second opinion on ambiguous cases, and Naive Bayes for retrospective portfolio sentiment analysis where outcomes are known.""")

# ================================================================
# SECTION G: SAMPLE MEMOS EXPLORER
# ================================================================
st.markdown("---")
st.html(section_header("Analyst Memo Explorer"))

# Filter options
col1, col2, col3 = st.columns(3)
with col1:
    filter_outcome = st.selectbox("Filter by outcome:", ['All'] + sorted(df['investment_outcome'].unique().tolist()))
with col2:
    filter_sentiment = st.selectbox("Filter by VADER sentiment:", ['All', 'Positive', 'Neutral', 'Negative'])
with col3:
    n_samples = st.slider("Number of memos:", 1, 20, 5)

# Apply filters
mask = pd.Series([True] * len(df))
if filter_outcome != 'All':
    mask = mask & (df['investment_outcome'] == filter_outcome)
vader_classes = sentiment['classifications']['vader']
if filter_sentiment != 'All':
    sentiment_mask = [vc == filter_sentiment for vc in vader_classes]
    mask = mask & pd.Series(sentiment_mask)

filtered_indices = mask[mask].index.tolist()

if filtered_indices:
    sample_indices = np.random.choice(filtered_indices, min(n_samples, len(filtered_indices)), replace=False)

    for idx in sample_indices:
        row = df.iloc[idx]
        memo = sentiment['memos'][idx]
        v_score = sentiment['scores']['vader_compound'][idx]
        tb_score = sentiment['scores']['textblob_polarity'][idx]
        v_class = vader_classes[idx]
        nb_class = sentiment['classifications']['naive_bayes'][idx]

        # Color-code by VADER sentiment
        badge_color = COLORS['success'] if v_class == 'Positive' else COLORS['danger'] if v_class == 'Negative' else COLORS['warning']

        st.html(f"""
        <div style="background: #1E2A3A; border-left: 4px solid {badge_color}; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0;">
            <div style="color: #95A5A6; font-size: 0.8em; margin-bottom: 8px;">
                <strong>{row['company_name']}</strong> | {row['sector']} | {row['country']} | {row['funding_stage']} | Outcome: {row['investment_outcome']}
            </div>
            <div style="color: #ECF0F1; font-size: 0.9em; line-height: 1.5;">
                {memo}
            </div>
            <div style="margin-top: 10px; font-size: 0.8em;">
                <span style="color: {COLORS['accent2']};">VADER: {v_score:+.3f} ({v_class})</span> &nbsp;|&nbsp;
                <span style="color: {COLORS['accent']};">TextBlob: {tb_score:+.3f}</span> &nbsp;|&nbsp;
                <span style="color: {COLORS['success']};">NB: {nb_class}</span>
            </div>
        </div>
        """)
else:
    st.warning("No memos match the selected filters.")

# ================================================================
# SECTION H: DISAGREEMENT FLAGS — Quant vs Qual
# ================================================================
st.markdown("---")
st.html(section_header("Disagreement Flags — Quantitative Model vs Analyst Sentiment"))

st.markdown(
    "Ventures where the **quantitative model predicts success** (outcome_binary = 1 or high probability) "
    "but **analyst sentiment is negative** represent the highest-value disagreements. These are cases where "
    "the human signal may be catching something the numbers miss — or vice versa."
)

# Build disagreement table
vader_cls_list = sentiment['classifications']['vader']
actual_outcomes = df['outcome_binary'].values

disagree_quant_pos_sent_neg = []  # Model says invest, analyst says negative
disagree_quant_neg_sent_pos = []  # Model says avoid, analyst says positive

for i in range(len(df)):
    outcome = actual_outcomes[i]
    v_cls = vader_cls_list[i]
    v_score = sentiment['scores']['vader_compound'][i]

    if outcome == 1 and v_cls == 'Negative':
        disagree_quant_pos_sent_neg.append(i)
    elif outcome == 0 and v_cls == 'Positive':
        disagree_quant_neg_sent_pos.append(i)

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown(f"##### Model: INVEST / Analyst: NEGATIVE ({len(disagree_quant_pos_sent_neg):,} ventures)")
    st.caption("These ventures succeeded on paper but analysts flagged concerns — possible hidden risks.")
    if disagree_quant_pos_sent_neg:
        sample_neg = disagree_quant_pos_sent_neg[:10]
        neg_df = pd.DataFrame({
            'Company': [df.iloc[i]['company_name'] for i in sample_neg],
            'Sector': [df.iloc[i].get('sector', 'N/A') for i in sample_neg],
            'Outcome': [df.iloc[i].get('investment_outcome', 'N/A') for i in sample_neg],
            'VADER Score': [f"{sentiment['scores']['vader_compound'][i]:+.3f}" for i in sample_neg],
            'VADER Class': [vader_cls_list[i] for i in sample_neg],
        })
        st.dataframe(neg_df, use_container_width=True, hide_index=True)
    else:
        st.info("No disagreements of this type found.")

with col_d2:
    st.markdown(f"##### Model: AVOID / Analyst: POSITIVE ({len(disagree_quant_neg_sent_pos):,} ventures)")
    st.caption("These ventures failed despite positive analyst sentiment — possible over-optimism.")
    if disagree_quant_neg_sent_pos:
        sample_pos = disagree_quant_neg_sent_pos[:10]
        pos_df = pd.DataFrame({
            'Company': [df.iloc[i]['company_name'] for i in sample_pos],
            'Sector': [df.iloc[i].get('sector', 'N/A') for i in sample_pos],
            'Outcome': [df.iloc[i].get('investment_outcome', 'N/A') for i in sample_pos],
            'VADER Score': [f"{sentiment['scores']['vader_compound'][i]:+.3f}" for i in sample_pos],
            'VADER Class': [vader_cls_list[i] for i in sample_pos],
        })
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    else:
        st.info("No disagreements of this type found.")

total_disagree = len(disagree_quant_pos_sent_neg) + len(disagree_quant_neg_sent_pos)
disagree_rate = total_disagree / len(df) if len(df) > 0 else 0

st.html(metric_highlight(
    f"<strong>Disagreement Rate:</strong> {disagree_rate:.1%} of ventures ({total_disagree:,} out of {len(df):,}) "
    f"show a mismatch between quantitative outcome and analyst sentiment. "
    f"For <strong>{mandate_cfg['fund']}</strong>, these warrant priority review."
))

# ── Prescriptive: Disagreement action ──
st.html(swf_insight(
    f"<b>Disagreement = Decision Point:</b> {total_disagree:,} ventures show quant-vs-qual mismatch. "
    f"<b>Action:</b> Pull the top 10 disagreements from each column above and assign to senior analysts for deep review. "
    f"'Model INVEST / Analyst NEGATIVE' cases may have hidden execution risk. "
    f"'Model AVOID / Analyst POSITIVE' cases may be undervalued — the analyst sees qualitative upside the model misses. "
    f"Each disagreement is a potential alpha opportunity or risk mitigation."
))

# ================================================================
# SECTION I: PRESCRIPTIVE — SENTIMENT AS SCREENING SIGNAL
# ================================================================
st.markdown("---")
st.html(section_header("Prescriptive: Sentiment as a Screening Signal"))

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sentiment-Outcome Correlation**")
    # Success rate by VADER sentiment class
    vader_cls = sentiment['classifications']['vader']
    actual_outcomes_arr = df['outcome_binary'].values

    sentiment_success = {}
    for cls in ['Positive', 'Neutral', 'Negative']:
        mask_cls = [i for i, c in enumerate(vader_cls) if c == cls]
        if mask_cls:
            sr = np.mean([actual_outcomes_arr[i] for i in mask_cls])
            sentiment_success[cls] = sr

    if sentiment_success:
        fig = go.Figure(data=[go.Bar(
            x=list(sentiment_success.keys()),
            y=list(sentiment_success.values()),
            marker_color=[color_map[k] for k in sentiment_success.keys()],
            text=[f"{v:.1%}" for v in sentiment_success.values()],
            textposition='auto'
        )])
        fig.update_layout(title='Actual Success Rate by VADER Sentiment',
                          yaxis_title='Success Rate', yaxis_tickformat='.0%')
        fig = apply_theme(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("If sentiment captures real signal, positive-sentiment ventures should have higher actual success rates.")

with col2:
    st.markdown("**VADER Component Breakdown**")
    vader_pos_scores = sentiment['scores']['vader_pos']
    vader_neg_scores = sentiment['scores']['vader_neg']
    vader_neu_scores = sentiment['scores']['vader_neu']

    fig = go.Figure()
    fig.add_trace(go.Box(y=vader_pos_scores, name='Positive', marker_color=COLORS['success']))
    fig.add_trace(go.Box(y=vader_neg_scores, name='Negative', marker_color=COLORS['danger']))
    fig.add_trace(go.Box(y=vader_neu_scores, name='Neutral', marker_color=COLORS['warning']))
    fig.update_layout(title='VADER Component Score Distributions', yaxis_title='Score')
    fig = apply_theme(fig, 350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("VADER decomposes each text into positive, negative, and neutral proportions. The compound score combines these into a single metric.")

# Mandate-tailored prescriptive recommendations
prescriptive_by_mandate = {
    'GIC — Financial Return': """
**Prescriptive Recommendation for GIC (Financial Return Focus):**
- **Screen new deals** using VADER sentiment as a first-pass filter — ventures with compound score < -0.1 warrant closer scrutiny on financial fundamentals
- **Prioritize positive-sentiment + high-margin** combinations — these have the strongest signal for successful exits
- **Flag disagreements** where analyst language is negative but financial metrics look strong — the analyst may be seeing execution risk not captured in the numbers
- **Use Naive Bayes** for quarterly portfolio reviews to track sentiment drift before formal downgrade decisions
- **Discount neutral sentiment** — for a return-focused mandate, ambiguity is a negative signal
""",
    'PIF — Strategic Alignment': """
**Prescriptive Recommendation for PIF (Strategic Alignment Focus):**
- **Weight strategic language** in memos more heavily than pure financial sentiment — look for terms about technology transfer, bilateral alignment, and domestic capability
- **Accept moderate financial sentiment** if strategic conviction is strong — PIF's mandate allows for lower financial returns when strategic value is high
- **Flag ventures** where analyst sentiment is positive but bilateral alignment scores are low — the opportunity may not serve Vision 2030
- **Track sentiment around national strategy keywords** as a leading indicator of policy alignment shifts
- **Use disagreement flags** to identify ventures where strategic value is under-recognized by quantitative models
""",
    'Mubadala — Balanced Portfolio': """
**Prescriptive Recommendation for Mubadala (Balanced Portfolio Focus):**
- **Use sentiment as a cross-check** on all five SWF dimensions — it should broadly align with the composite mandate score
- **Pay special attention to disagreement flags** — for a balanced fund, any systematic bias in analyst sentiment vs quantitative scores reveals portfolio blind spots
- **Diversify sentiment exposure** — avoid concentrating in ventures where all analysts are either bullish or bearish; healthy portfolios contain a mix
- **Use VADER for real-time screening** and Naive Bayes for retrospective analysis of portfolio sentiment evolution
- **Weight method agreement** as a confidence signal — when VADER, TextBlob, and NB all agree, the sentiment signal is stronger
""",
}

st.success(prescriptive_by_mandate.get(mandate, """
**Prescriptive Recommendation for SWF Analysts:**
- **Screen new deals** using VADER sentiment as a first-pass filter — ventures with compound score < -0.1 warrant closer scrutiny
- **Flag disagreements** between VADER and TextBlob — when methods disagree, the venture likely has mixed signals requiring human review
- **Leverage Naive Bayes** for quarterly portfolio reviews to identify ventures where analyst language has shifted from positive to negative
- **Combine with quantitative models** — sentiment provides a complementary signal to the classification and clustering models in other SIIP modules
- **Track sentiment drift** over time as a leading indicator of changing analyst conviction before formal downgrade decisions
"""))

# ================================================================
# ANALYTICS PIPELINE POSITIONING
# ================================================================
st.markdown("---")
st.html(section_header("Analytics Pipeline Position"))

p1, p2, p3, p4 = st.columns(4)
with p1:
    st.success("**DESCRIPTIVE**")
    st.markdown("Memo generation, word frequency analysis, sentiment score distributions")
with p2:
    st.warning("**DIAGNOSTIC**")
    st.markdown("Three NLP methods compared: why do VADER, TextBlob, and Naive Bayes produce different results?")
with p3:
    st.info("**PREDICTIVE**")
    st.markdown("Naive Bayes trained to predict sentiment class from text features (TF-IDF)")
with p4:
    st.error("**PRESCRIPTIVE**")
    st.markdown("Sentiment-based screening rules, method selection guidance, analyst workflow recommendations")
