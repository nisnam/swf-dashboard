"""
SIIP Dashboard — Styles & Theme
================================
Color palette, inline-styled HTML helpers.
All component helpers use inline styles for reliable rendering via st.html().
"""

# Color palette — Palantir/Bloomberg-inspired
COLORS = {
    'primary': '#1B2838',
    'secondary': '#2C3E50',
    'accent': '#D4AF37',
    'accent2': '#3498DB',
    'success': '#2ECC71',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'text': '#ECF0F1',
    'muted': '#95A5A6',
    'bg_card': '#1E2A3A',
    'bg_dark': '#0D1117',
}

PLOTLY_COLORS = [
    '#3498DB', '#E74C3C', '#2ECC71', '#F39C12',
    '#9B59B6', '#1ABC9C', '#E67E22', '#D4AF37',
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'
]

PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#ECF0F1', 'family': 'Inter, sans-serif'},
        'colorway': PLOTLY_COLORS,
        'xaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'zerolinecolor': 'rgba(255,255,255,0.2)'},
        'yaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'zerolinecolor': 'rgba(255,255,255,0.2)'},
    }
}


def inject_css():
    """CSS for native Streamlit elements only."""
    return """
    <style>
        .stApp { background-color: #0D1117; }
        .stApp, .stApp p, .stApp li, .stApp span, .stApp label {
            color: #E8EAED !important; font-size: 15px;
        }
        .stApp h1 { color: #FFFFFF !important; font-size: 2em !important; }
        .stApp h2 { color: #FFFFFF !important; font-size: 1.6em !important; }
        .stApp h3 { color: #ECF0F1 !important; font-size: 1.3em !important; }
        .stApp .stCaption, .stApp caption { color: #B0B8C4 !important; font-size: 13px !important; }
        .stDataFrame, [data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; overflow: hidden;
        }
        .stAlert > div { color: #FFFFFF !important; font-size: 14px !important; line-height: 1.6 !important; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
        .css-1d391kg, [data-testid="stSidebar"] { background-color: #1B2838; }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] li {
            color: #B0B8C4 !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background-color: #1E2A3A; border-radius: 8px; color: #B0B8C4; padding: 8px 16px; }
        .stTabs [aria-selected="true"] { background-color: #2C3E50; color: #D4AF37; }
        .streamlit-expanderHeader { color: #E8EAED !important; font-size: 15px !important; }
    </style>
    """


def swf_insight(text):
    """SWF insight callout — fully inline styled."""
    return (
        '<div style="background:rgba(52,152,219,0.1); border:1px solid rgba(52,152,219,0.3); '
        'border-radius:8px; padding:15px 18px; margin:15px 0; color:#E8EAED; '
        f'font-size:14px; line-height:1.7;">{text}</div>'
    )


def kpi_card(label, value, delta=None):
    """KPI card — fully inline styled."""
    delta_html = f'<div style="font-size:0.85em; color:#2ECC71;">{delta}</div>' if delta else ''
    return (
        '<div style="background:linear-gradient(135deg,#1E2A3A 0%,#2C3E50 100%); '
        'border:1px solid rgba(212,175,55,0.3); border-radius:12px; '
        'padding:20px; text-align:center; margin:5px 0;">'
        f'<div style="font-size:0.85em; color:#B0B8C4; text-transform:uppercase; letter-spacing:1px;">{label}</div>'
        f'<div style="font-size:2.2em; font-weight:700; color:#D4AF37; margin:5px 0;">{value}</div>'
        f'{delta_html}'
        '</div>'
    )


def evidence_panel(label, content=""):
    """Evidence panel — fully inline styled."""
    return (
        '<div style="background:#1E2A3A; border:1px solid rgba(52,152,219,0.3); '
        'border-radius:8px; padding:15px; margin:10px 0;">'
        f'<div style="font-size:0.8em; color:#5DADE2; text-transform:uppercase; '
        f'letter-spacing:1.5px; margin-bottom:10px; font-weight:600;">{label}</div>'
        f'{content}'
        '</div>'
    )


def section_header(title):
    """Section header — fully inline styled."""
    return (
        '<div style="border-bottom:2px solid #D4AF37; padding-bottom:8px; margin-bottom:15px; '
        f'font-size:1.2em; color:#FFFFFF; font-weight:600;">{title}</div>'
    )


def metric_highlight(text):
    """Metric highlight — fully inline styled."""
    return (
        '<div style="background:rgba(212,175,55,0.12); border-left:4px solid #D4AF37; '
        'padding:12px 18px; margin:12px 0; border-radius:0 8px 8px 0; color:#E8EAED; '
        f'font-size:14px; line-height:1.6;">{text}</div>'
    )


def verdict_badge(verdict):
    """Verdict badge — fully inline styled."""
    cmap = {'INVEST': '#2ECC71', 'MONITOR': '#F39C12', 'AVOID': '#E74C3C'}
    c = cmap.get(verdict.upper(), '#F39C12')
    return (
        f'<div style="background:rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.2); '
        f'border:2px solid {c}; color:{c}; padding:15px; border-radius:12px; '
        f'text-align:center; font-size:1.5em; font-weight:700;">{verdict.upper()}</div>'
    )


def format_currency(value):
    if abs(value) >= 1e12: return f"${value/1e12:.1f}T"
    elif abs(value) >= 1e9: return f"${value/1e9:.1f}B"
    elif abs(value) >= 1e6: return f"${value/1e6:.1f}M"
    elif abs(value) >= 1e3: return f"${value/1e3:.0f}K"
    else: return f"${value:.0f}"


def format_pct(value):
    return f"{value:.1%}"


def mandate_badge(mandate_name, mandate_config):
    """Mandate badge — fully inline styled."""
    color = mandate_config['color']
    icon = mandate_config['icon']
    fund = mandate_config['fund']
    lens = mandate_config['lens']
    return (
        f'<div style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15); '
        f'border:1px solid {color}; border-radius:10px; padding:12px 16px; margin:8px 0;">'
        f'<span style="font-size:1.3em;">{icon}</span>'
        f'<span style="color:{color}; font-weight:700; font-size:1.1em; margin-left:8px;">{fund}</span>'
        f'<br><span style="color:#B0B8C4; font-size:0.85em; font-style:italic;">"{lens}"</span>'
        '</div>'
    )


def verdict_card(verdict, explanation):
    """Traffic-light verdict card — fully inline styled."""
    configs = {
        'INVEST': ('#2ECC71', 'rgba(46,204,113,0.12)'),
        'MONITOR': ('#F39C12', 'rgba(243,156,18,0.12)'),
        'AVOID': ('#E74C3C', 'rgba(231,76,60,0.12)'),
    }
    color, bg = configs.get(verdict, configs['MONITOR'])
    return (
        f'<div style="background:{bg}; border:2px solid {color}; '
        f'border-radius:12px; padding:20px; text-align:center; margin:10px 0;">'
        f'<div style="color:{color}; font-size:1.8em; font-weight:700; margin:8px 0;">{verdict}</div>'
        f'<div style="color:#E8EAED; font-size:0.9em; line-height:1.5;">{explanation}</div>'
        '</div>'
    )


def bridge_sentence(text, next_page=""):
    """Bridge sentence — fully inline styled."""
    return (
        '<div style="background:rgba(212,175,55,0.08); border:1px solid rgba(212,175,55,0.25); '
        'border-radius:8px; padding:12px 18px; margin:20px 0; text-align:center;">'
        f'<span style="color:#D4AF37; font-size:0.95em;">&rarr; {text}</span>'
        '</div>'
    )
