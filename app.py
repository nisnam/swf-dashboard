"""
SIIP — Sovereign Investment Intelligence Platform
===================================================
Fund Command Center: Mandate-driven executive intelligence
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import (
    load_data, load_frameworks, load_interpretations,
    load_classification_results, load_clustering_results, load_arm_rules,
    MANDATES, compute_mandate_score, compute_dimension_scores,
    get_df_stats, get_mandate_insight, DIMENSION_ICONS,
)
from utils.styles import (
    inject_css, kpi_card, section_header, metric_highlight,
    format_currency, format_pct, COLORS, PLOTLY_COLORS,
    mandate_badge, bridge_sentence, swf_insight,
)
from utils.charts import donut_chart, bar_chart, apply_theme, radar_chart, gauge_chart

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIIP — Fund Command Center",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html(inject_css())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ SIIP")
    st.markdown("**Sovereign Investment Intelligence Platform**")
    st.markdown("---")

    # Mandate selector (radio buttons, persisted in session_state)
    st.markdown("##### Investment Mandate")
    mandate_options = list(MANDATES.keys())
    default_idx = mandate_options.index(
        st.session_state.get("mandate", "GIC — Financial Return")
    )
    selected_mandate = st.radio(
        "Select the active fund mandate",
        mandate_options,
        index=default_idx,
        key="mandate",
        label_visibility="collapsed",
    )
    mandate_cfg = MANDATES[selected_mandate]
    st.html(mandate_badge(selected_mandate, mandate_cfg))

    st.markdown("---")
    st.markdown("##### Navigation")
    st.markdown(
        """
    - 🏠 **Command Center** — This page
    - 📊 Pipeline Intelligence
    - 📈 Risk Engine
    - 🎯 Prediction Engine
    - 🧬 Segmentation Lab
    - 🔗 Pattern Discovery
    - 🛠️ Deal Evaluator
    - 💬 Sentiment Intelligence
    """
    )
    st.markdown("---")
    st.caption("Built for SWF Decision Intelligence")
    st.caption("Data: 6,798 ventures | 12 sectors | 8 regions")

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()
frameworks = load_frameworks()
interp = load_interpretations()
stats = frameworks["overall_stats"]
df_stats = get_df_stats(df)

mandate_name = st.session_state.get("mandate", "GIC — Financial Return")
mandate = MANDATES[mandate_name]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏛️ Fund Command Center")
st.caption(
    f"Real-time venture intelligence for **{mandate['fund']}** — "
    f'_{mandate["lens"]}_'
)
st.html(
    swf_insight(
        f"<strong>Active Mandate:</strong> {mandate['description']}"
    )
)

# ── Row 1: Mandate-framed KPI Cards ──────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

# Compute portfolio-wide mandate score
mandate_scores = df.apply(
    lambda row: compute_mandate_score(row, mandate_name, df_stats), axis=1
)
avg_mandate_score = mandate_scores.mean()

with c1:
    st.html(
        kpi_card(
            "Portfolio Ventures",
            f"{stats['total_ventures']:,}",
            f"{len(stats['sectors'])} sectors | {len(stats['regions'])} regions",
        )
    )
with c2:
    st.html(
        kpi_card(
            "Capital Deployed",
            format_currency(stats["total_capital"]),
            f"Avg {format_currency(stats['total_capital'] / stats['total_ventures'])} / venture",
        )
    )
with c3:
    st.html(
        kpi_card(
            f"Mandate Fit Score",
            f"{avg_mandate_score:.1f}",
            f"{mandate['fund']} weighted average",
        )
    )
with c4:
    st.html(
        kpi_card(
            "Success Rate",
            format_pct(stats["overall_success_rate"]),
            f"{stats['outcome_dist'].get('Successful Exit', 0):,} exits",
        )
    )

st.markdown("")

# ── Bridge ────────────────────────────────────────────────────────────────────
st.html(
    bridge_sentence(
        "The KPIs above reflect portfolio health through the "
        f"{mandate['fund']} lens. Below, explore how individual ventures "
        "rank under this mandate."
    )
)

# ── Row 2: Top 10 Ventures by Mandate Score + Portfolio Radar ─────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.html(
        section_header(f"Top 10 Ventures — {mandate['fund']} Mandate Score")
    )

    # Build top-10 table
    top_df = df.copy()
    top_df["Mandate Score"] = mandate_scores
    top_cols = {
        "company_name": "Company",
        "sector": "Sector",
        "funding_stage": "Stage",
        "Mandate Score": "Mandate Score",
        "outcome_binary": "Success Prob.",
    }
    available_cols = [c for c in top_cols.keys() if c in top_df.columns]
    top_display = (
        top_df[available_cols]
        .rename(columns=top_cols)
        .sort_values("Mandate Score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_display.index = top_display.index + 1  # 1-based ranking
    if "Success Prob." in top_display.columns:
        top_display["Success Prob."] = top_display["Success Prob."].apply(
            lambda v: f"{v:.0%}"
        )

    st.dataframe(top_display, use_container_width=True, height=390)

    # ── Prescriptive: Top 10 interpretation ──
    top3_sectors = top_display["Sector"].value_counts().head(3).index.tolist() if "Sector" in top_display.columns else []
    top3_stages = top_display["Stage"].value_counts().head(2).index.tolist() if "Stage" in top_display.columns else []
    if mandate == "GIC — Financial Return":
        top10_action = (
            f"These 10 ventures score highest on financial return dimensions. "
            f"Dominant sectors: <b>{', '.join(top3_sectors)}</b>. "
            f"<b>Action:</b> Fast-track IC review for the top 3. For ventures below rank 5, request updated financials before committing diligence resources."
        )
    elif mandate == "PIF — Strategic Alignment":
        top10_action = (
            f"These ventures rank highest on strategic alignment and domestic impact. "
            f"<b>Action:</b> Cross-reference top 3 against Vision 2030 priority sectors. "
            f"If tech transfer scores are below portfolio average, escalate for bilateral review."
        )
    else:
        top10_action = (
            f"Balanced scoring across all 5 dimensions. Sectors: <b>{', '.join(top3_sectors)}</b>. "
            f"<b>Action:</b> Check dimension radar for imbalances — any venture scoring below 30 on a single dimension needs a mitigation plan."
        )
    st.html(swf_insight(top10_action))

with col_right:
    st.html(
        section_header("Portfolio Dimension Profile (Average)")
    )

    # Compute average dimension scores across all ventures
    all_dim_scores = df.apply(
        lambda row: compute_dimension_scores(row, df_stats), axis=1
    )
    dim_avg = pd.DataFrame(all_dim_scores.tolist()).mean().to_dict()

    categories = list(dim_avg.keys())
    cat_labels = [f"{DIMENSION_ICONS.get(c, '')} {c}" for c in categories]
    avg_values = [dim_avg[c] for c in categories]

    # Also show mandate weight profile for comparison
    weight_values = [mandate["weights"][c] * 100 for c in categories]

    fig = radar_chart(
        cat_labels,
        {
            "Portfolio Avg (0-100)": avg_values,
            f"{mandate['fund']} Weight (%)": weight_values,
        },
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Blue = average venture score per dimension. "
        "Orange = how heavily this mandate weights each dimension. "
        "Gaps between the two reveal where portfolio strengths diverge from mandate priorities."
    )

# ── Bridge ────────────────────────────────────────────────────────────────────
st.html(
    bridge_sentence(
        "The radar chart reveals portfolio strengths versus mandate priorities. "
        "Next, examine how outcomes and deal flow break down across the portfolio."
    )
)

# ── Row 3: Outcome Distribution + Sector Deal Flow ───────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.html(
        section_header("Portfolio Outcome Distribution")
    )
    outcome_data = stats["outcome_dist"]
    fig = donut_chart(
        labels=list(outcome_data.keys()),
        values=list(outcome_data.values()),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(interp["home"]["kpi_summary"])

    # ── Prescriptive: Outcome interpretation ──
    active_pct = outcome_data.get("Still Active", 0) / sum(outcome_data.values()) * 100
    success_pct = outcome_data.get("Successful Exit", 0) / sum(outcome_data.values()) * 100
    writeoff_pct = outcome_data.get("Write-off", 0) / sum(outcome_data.values()) * 100
    st.html(swf_insight(
        f"<b>Portfolio Health Check:</b> {success_pct:.0f}% of ventures achieved successful exits, while {writeoff_pct:.0f}% were written off. "
        f"{active_pct:.0f}% are still active. <b>Action:</b> Focus portfolio management on converting active ventures to exits — "
        f"review the 'Still Active' cohort for stalled deals that should be escalated or written down."
    ))

with col2:
    st.html(section_header("Deal Flow by Sector"))
    sector_counts = df["sector"].value_counts().sort_values()
    fig = bar_chart(
        x=sector_counts.index.tolist(),
        y=sector_counts.values.tolist(),
        orientation="h",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
    top3 = sector_counts.index[-3:][::-1]
    top3_share = sector_counts.values[-3:].sum() / len(df)
    st.caption(
        f"Top sectors: {', '.join(top3)}. Together they represent "
        f"{top3_share:.0%} of the pipeline."
    )

    # ── Prescriptive: Sector concentration risk ──
    if top3_share > 0.5:
        st.html(swf_insight(
            f"<b>Concentration Risk:</b> Top 3 sectors account for {top3_share:.0%} of the pipeline — "
            f"above the 50% diversification threshold. <b>Action:</b> Increase sourcing in underrepresented sectors "
            f"to reduce single-sector exposure. Consider sector caps in the deal screening criteria."
        ))
    else:
        st.html(swf_insight(
            f"<b>Diversification Check:</b> Top 3 sectors represent {top3_share:.0%} — well-diversified. "
            f"<b>Action:</b> Maintain current sourcing balance. Monitor for emerging sector opportunities."
        ))

# ── Row 4: Success Rate by Stage + Region Distribution ───────────────────────
col1, col2 = st.columns(2)

with col1:
    st.html(
        section_header("Success Rate by Funding Stage")
    )
    stage_order = ["Seed", "Series A", "Series B", "Series C", "Series D", "Pre-IPO"]
    stage_success = df.groupby("funding_stage")["outcome_binary"].mean().reindex(stage_order)
    colors = [
        COLORS["danger"] if v < 0.15
        else COLORS["warning"] if v < 0.20
        else COLORS["success"]
        for v in stage_success.values
    ]
    fig = bar_chart(
        x=stage_order,
        y=stage_success.values.tolist(),
        color=colors,
        height=320,
        text=[f"{v:.1%}" for v in stage_success.values],
    )
    fig.update_layout(yaxis_title="Success Rate", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    best_stage = stage_success.idxmax()
    worst_stage = stage_success.idxmin()
    st.caption(
        f"{best_stage} ventures show the highest success rate at "
        f"{stage_success.max():.1%}. Later stages correlate with higher exit probability."
    )

    # ── Prescriptive: Stage allocation guidance ──
    if mandate == "GIC — Financial Return":
        st.html(swf_insight(
            f"<b>Stage Strategy (GIC):</b> {best_stage} shows {stage_success.max():.1%} success rate — "
            f"highest in the portfolio. <b>Action:</b> Overweight later-stage deals for return certainty. "
            f"Cap {worst_stage} exposure at 15% of new commitments to limit downside."
        ))
    elif mandate == "PIF — Strategic Alignment":
        st.html(swf_insight(
            f"<b>Stage Strategy (PIF):</b> Early-stage ventures offer higher strategic value through tech transfer, "
            f"even at lower success rates. <b>Action:</b> Accept {worst_stage} risk where strategic alignment scores exceed 70. "
            f"Use {best_stage} deals as portfolio anchors."
        ))
    else:
        st.html(swf_insight(
            f"<b>Stage Strategy (Mubadala):</b> Spread exposure across stages proportional to risk-adjusted returns. "
            f"<b>Action:</b> Target 40% later-stage ({best_stage}/Pre-IPO), 35% mid-stage (Series B/C), 25% early-stage for upside."
        ))

with col2:
    st.html(section_header("Ventures by Region"))
    region_data = (
        df.groupby("region")
        .agg(count=("venture_id", "count"), success_rate=("outcome_binary", "mean"))
        .sort_values("count", ascending=True)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=region_data.index,
            x=region_data["count"],
            orientation="h",
            marker_color=COLORS["accent2"],
            name="Deal Count",
            hovertemplate="%{y}: %{x} ventures<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Number of Ventures")
    fig = apply_theme(fig, 320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(interp["home"]["portfolio_health"])

# ── Bridge ────────────────────────────────────────────────────────────────────
st.html(
    bridge_sentence(
        "Portfolio diagnostics are complete. The intelligence modules below "
        "translate these patterns into actionable strategies for "
        f"{mandate['fund']}."
    )
)

# ── Row 5: Intelligence Module Cards (business-framed) ───────────────────────
st.markdown("---")
st.html(
    section_header("Intelligence Modules — Strategic Findings")
)

# Mandate-specific module insight
st.html(
    swf_insight(get_mandate_insight("pipeline", mandate_name))
)

# Load module summaries
try:
    clf_results = load_classification_results()
    best_clf = max(clf_results, key=lambda k: clf_results[k]["auc_roc"])
    best_auc = clf_results[best_clf]["auc_roc"]
except Exception:
    best_clf, best_auc = "N/A", 0

try:
    clust = load_clustering_results()
    best_k = clust["best_k"]
except Exception:
    best_k = "N/A"

try:
    arm = load_arm_rules()
    n_rules = arm["n_rules"]
except Exception:
    n_rules = 0

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.info("**📈 Risk Engine**")
    st.markdown(
        "Identifies which **financial and macro factors** drive venture "
        "outcomes. Reveals whether returns are explained by fundamentals "
        "or structural market forces."
    )
    st.caption("Answers: What drives outcomes?")

with m2:
    st.info("**🎯 Prediction Engine**")
    st.markdown(
        f"Best model (**{best_clf}**, AUC {best_auc:.3f}) predicts exit "
        "success. Enables **pre-screening** of new deals before "
        "committing diligence resources."
    )
    st.caption("Answers: Will this venture succeed?")

with m3:
    st.info("**🧬 Segmentation Lab**")
    st.markdown(
        f"**{best_k} venture archetypes** discovered. Each archetype has "
        "a distinct risk-return profile — tailor allocation strategy and "
        "monitoring cadence per segment."
    )
    st.caption("Answers: What type of venture is this?")

with m4:
    st.info("**🔗 Pattern Discovery**")
    st.markdown(
        f"**{n_rules:,} association rules** reveal hidden feature "
        "combinations that signal success or failure. Use these as "
        "deal screening checklists."
    )
    st.caption("Answers: What patterns predict exits?")

with m5:
    st.info("**💬 Sentiment Intelligence**")
    st.markdown(
        "NLP analysis of analyst memos surfaces **qualitative signals** "
        "invisible to financial models. Cross-reference sentiment with "
        "quantitative scores."
    )
    st.caption("Answers: What does the street think?")

# ── Analytics Pipeline ────────────────────────────────────────────────────────
st.markdown("---")
st.html(
    section_header(
        "Analytics Pipeline: Descriptive → Diagnostic → Predictive → Prescriptive"
    )
)

p1, p2, p3, p4 = st.columns(4)
with p1:
    st.success("**DESCRIPTIVE**")
    st.markdown(
        "Portfolio distributions, sector mix, regional exposure, outcome breakdowns"
    )
with p2:
    st.warning("**DIAGNOSTIC**")
    st.markdown(
        "Regression analysis: which features explain returns and why some ventures fail"
    )
with p3:
    st.info("**PREDICTIVE**")
    st.markdown(
        "ML classifiers, clustering archetypes, time-series forecasts, anomaly detection"
    )
with p4:
    st.error("**PRESCRIPTIVE**")
    st.markdown(
        "Deal scoring, mandate-weighted rankings, screening rules, strategy recommendations"
    )

st.html(
    bridge_sentence(
        "Use the sidebar to navigate to any intelligence module. "
        "Your selected mandate will persist across all pages."
    )
)
