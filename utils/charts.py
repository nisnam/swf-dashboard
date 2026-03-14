"""
SIIP Dashboard — Reusable Chart Builders
==========================================
Consistent Plotly charts with SIIP theme
"""

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
from utils.styles import PLOTLY_COLORS, COLORS


def apply_theme(fig, height=400):
    """Apply consistent SIIP theme to any Plotly figure"""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ECF0F1', family='Inter, sans-serif', size=12),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.15)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.15)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    )
    return fig


def donut_chart(labels, values, title="", height=350):
    """Create a donut chart"""
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker_colors=PLOTLY_COLORS[:len(labels)],
        textinfo='label+percent', textfont_size=11,
        hovertemplate='%{label}: %{value:,}<extra></extra>'
    )])
    fig.update_layout(title=title, showlegend=False)
    return apply_theme(fig, height)


def bar_chart(x, y, title="", color=None, orientation='v', height=400, text=None):
    """Create a bar chart"""
    if orientation == 'h':
        fig = go.Figure(data=[go.Bar(
            y=x, x=y, orientation='h',
            marker_color=color or COLORS['accent2'],
            text=text, textposition='auto',
            hovertemplate='%{y}: %{x}<extra></extra>'
        )])
    else:
        fig = go.Figure(data=[go.Bar(
            x=x, y=y,
            marker_color=color or COLORS['accent2'],
            text=text, textposition='auto',
            hovertemplate='%{x}: %{y}<extra></extra>'
        )])
    fig.update_layout(title=title)
    return apply_theme(fig, height)


def grouped_bar(df, x, y, color, title="", height=400):
    """Create a grouped/stacked bar chart"""
    fig = px.bar(df, x=x, y=y, color=color, barmode='group',
                 color_discrete_sequence=PLOTLY_COLORS)
    fig.update_layout(title=title)
    return apply_theme(fig, height)


def heatmap_chart(z, x_labels, y_labels, title="", height=400, colorscale='RdYlGn'):
    """Create a heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=colorscale, texttemplate='%{z:.2f}',
        hovertemplate='%{x} × %{y}: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(title=title)
    return apply_theme(fig, height)


def scatter_chart(x, y, color=None, size=None, hover_data=None, labels=None,
                  title="", height=400, color_discrete_map=None):
    """Create a scatter plot"""
    df = pd.DataFrame({'x': x, 'y': y})
    if color is not None:
        df['color'] = color
    if size is not None:
        df['size'] = size

    fig = px.scatter(
        df, x='x', y='y',
        color='color' if color is not None else None,
        size='size' if size is not None else None,
        color_discrete_sequence=PLOTLY_COLORS,
        color_discrete_map=color_discrete_map,
        labels=labels or {},
        title=title
    )
    fig.update_traces(marker=dict(opacity=0.7))
    return apply_theme(fig, height)


def line_chart(x, y, title="", height=400, name=None):
    """Create a line chart"""
    fig = go.Figure()
    if isinstance(y, dict):
        for label, values in y.items():
            fig.add_trace(go.Scatter(x=x, y=values, mode='lines+markers', name=label))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name or '',
                                 line=dict(color=COLORS['accent2'])))
    fig.update_layout(title=title)
    return apply_theme(fig, height)


def roc_curves(clf_results, title="ROC Curves — All Models"):
    """Create overlaid ROC curves for all classifiers"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color='rgba(255,255,255,0.3)'),
        name='Random (AUC=0.5)', showlegend=True
    ))
    for i, (name, result) in enumerate(clf_results.items()):
        roc = result['roc_curve']
        fig.add_trace(go.Scatter(
            x=roc['fpr'], y=roc['tpr'], mode='lines',
            name=f"{name} (AUC={result['auc_roc']:.3f})",
            line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=2)
        ))
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.55, y=0.05)
    )
    return apply_theme(fig, 450)


def confusion_matrix_chart(cm, title="Confusion Matrix"):
    """Create a confusion matrix heatmap"""
    labels = ['Failure (0)', 'Success (1)']
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale='Blues',
        texttemplate='%{z:,}',
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z:,}<extra></extra>'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    return apply_theme(fig, 350)


def feature_importance_chart(importances, feature_names, title="Feature Importances", top_n=15):
    """Horizontal bar chart of feature importances"""
    sorted_idx = np.argsort(list(importances.values()))[::-1][:top_n]
    names = [feature_names[i] for i in sorted_idx]
    vals = [list(importances.values())[i] for i in sorted_idx]

    colors = [COLORS['accent'] if v > np.mean(vals) else COLORS['accent2'] for v in vals]

    fig = go.Figure(data=[go.Bar(
        y=names[::-1], x=vals[::-1], orientation='h',
        marker_color=colors[::-1]
    )])
    fig.update_layout(title=title)
    return apply_theme(fig, max(300, top_n * 25))


def radar_chart(categories, values_dict, title="", height=400):
    """Create a radar/spider chart comparing multiple profiles"""
    fig = go.Figure()
    for name, values in values_dict.items():
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # close the polygon
            theta=categories + [categories[0]],
            fill='toself', name=name, opacity=0.6
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        ),
        title=title, showlegend=True
    )
    return apply_theme(fig, height)


def histogram_chart(values, title="", nbins=30, height=350):
    """Create a histogram"""
    fig = go.Figure(data=[go.Histogram(
        x=values, nbinsx=nbins,
        marker_color=COLORS['accent2'],
        opacity=0.8
    )])
    fig.update_layout(title=title, yaxis_title='Count')
    return apply_theme(fig, height)


def gauge_chart(value, title="", max_val=100, height=250):
    """Create a gauge/dial chart for probability display"""
    color = COLORS['success'] if value > 60 else COLORS['warning'] if value > 30 else COLORS['danger']
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#ECF0F1'}},
        number={'suffix': '%', 'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#ECF0F1'},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0)',
            'bordercolor': 'rgba(255,255,255,0.2)',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(231,76,60,0.2)'},
                {'range': [30, 60], 'color': 'rgba(243,156,18,0.2)'},
                {'range': [60, 100], 'color': 'rgba(46,204,113,0.2)'}
            ]
        }
    ))
    return apply_theme(fig, height)
