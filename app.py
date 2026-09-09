"""
INFRAguard AI — Explainable Predictive Intelligence Layer for Infrastructure Project Monitoring
Aligned with MoSPI PAIMANA (April 2026 report context).

Single-file Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# DEFENSIVE IMPORTS — CatBoost & SHAP with graceful fallbacks
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    from catboost import CatBoostClassifier as _CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import shap as _shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="INFRAguard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# SESSION STATE SAFETY — initialize before any UI
# ---------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True
if "user_role" not in st.session_state:
    st.session_state["user_role"] = "National / MoSPI Officer"
if "selected_project_code" not in st.session_state:
    st.session_state["selected_project_code"] = None
if "sim_inputs" not in st.session_state:
    st.session_state["sim_inputs"] = {}


# ---------------------------------------------------------------------------
# CUSTOM CSS — Glassmorphism + Pastel Gradient
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    .stApp {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        background-attachment: fixed;
    }
    /* ---- Main container ---- */
    .stMain, .stMain > section {
        background: transparent;
    }
    /* ---- Glassmorphism card helper ---- */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.6);
        margin-bottom: 18px;
    }
    /* ---- KPI card ---- */
    .kpi-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(31, 38, 135, 0.12);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        text-align: center;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #2d2d44;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 13px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 2px;
    }
    /* ---- Title ---- */
    .app-title {
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }
    .app-subtitle {
        font-size: 14px;
        color: #4b5563;
        margin-top: 0;
    }
    /* ---- Risk badges ---- */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .risk-high { background: #ffd6d6; color: #b91c1c; }
    .risk-medium { background: #fff3d6; color: #b45309; }
    .risk-low { background: #d6f5d6; color: #15803d; }

    /* ---- Sidebar styling ---- */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-right: 1px solid rgba(255, 255, 255, 0.5);
    }
    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.6);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 600;
        font-size: 14px;
    }
    /* ---- Dataframe ---- */
    .stDataFrame, .stTable {
        border-radius: 12px;
        overflow: hidden;
    }
    /* ---- Headers inside tabs ---- */
    .tab-header {
        font-size: 20px;
        font-weight: 700;
        color: #2d2d44;
        margin-bottom: 12px;
    }
    /* ---- Comparison cards ---- */
    .compare-card {
        border-radius: 14px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 18px rgba(31,38,135,0.12);
    }
    .compare-baseline {
        background: rgba(224, 195, 252, 0.55);
        border: 1px solid rgba(106, 17, 203, 0.25);
    }
    .compare-simulated {
        background: rgba(142, 197, 252, 0.55);
        border: 1px solid rgba(37, 117, 252, 0.25);
    }
    /* ---- Metric helper ---- */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(106,17,203,0.3), transparent);
        margin: 16px 0;
    }
    /* ---- Download button override ---- */
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# PASTEL COLOR PALETTE
# ---------------------------------------------------------------------------
PASTEL_COLORS = [
    "#A78BFA", "#60A5FA", "#34D399", "#FBBF24", "#F472B6",
    "#22D3EE", "#C084FC", "#FB923C", "#4ADE80", "#FCD34D",
    "#94A3B8", "#F87171",
]
RISK_COLORS = {"High": "#F87171", "Medium": "#FBBF24", "Low": "#34D399"}


# ---------------------------------------------------------------------------
# SYNTHETIC DATASET — 50 Central Sector Projects (₹150 Cr+)
# ---------------------------------------------------------------------------
@st.cache_data
def generate_dataset():
    rng = np.random.default_rng(42)
    n = 50

    ministries = [
        "Road Transport & Highways", "Railways", "Power", "Urban Affairs",
        "Civil Aviation", "Shipping", "Telecommunications", "Coal",
        "New & Renewable Energy", "Steel", "Petroleum & Natural Gas",
    ]
    sectors = {
        "Road Transport & Highways": "Highways",
        "Railways": "Rail Infrastructure",
        "Power": "Power Generation",
        "Urban Affairs": "Urban Transit",
        "Civil Aviation": "Aviation",
        "Shipping": "Ports",
        "Telecommunications": "Telecom",
        "Coal": "Mining",
        "New & Renewable Energy": "Renewable Energy",
        "Steel": "Heavy Industry",
        "Petroleum & Natural Gas": "Oil & Gas",
    }
    states = [
        "Maharashtra", "Uttar Pradesh", "Karnataka", "Tamil Nadu", "Gujarat",
        "Rajasthan", "Madhya Pradesh", "West Bengal", "Bihar", "Andhra Pradesh",
        "Telangana", "Odisha", "Kerala", "Punjab", "Haryana", "Assam",
        "Chhattisgarh", "Jharkhand", "Delhi", "Himachal Pradesh",
    ]
    delay_reasons = [
        "Land Acquisition", "Forest Clearances", "Utility Shifting",
        "Contracting Issue", "None",
    ]

    project_names_by_sector = {
        "Highways": [
            "NH-48 Six-Laning", "Eastern Peripheral Expressway Phase II",
            "Delhi-Mumbai Expressway spur", "Bharatmala Corridor Section",
            "Coastal Road Connector", "Ganga Expressway Link",
        ],
        "Rail Infrastructure": [
            "HSR Mumbai-Ahmedabad Segment", "Dedicated Freight Corridor E-W",
            "Station Redevelopment Hub", "RRTS Delhi-Meerut",
            "Vande Bharat Maintenance Depot",
        ],
        "Power Generation": [
            "Super Thermal Plant Stage III", "Interstate Transmission System",
            "Hydroelectric Project Upper Basin", "Smart Grid Integration",
        ],
        "Urban Transit": [
            "Metro Phase IV Corridor", "BRTS City Expansion",
            "Smart City Mission Cluster",
        ],
        "Aviation": [
            "Greenfield Airport Terminal", "Regional Connectivity Route",
        ],
        "Ports": [
            "Container Terminal Expansion", "Sagarmala Port Modernization",
        ],
        "Telecom": [
            "BharatNet Fiber Backbone", "5G Core Network Rollout",
        ],
        "Mining": [
            "Coal Block Development", "Washery Capacity Augmentation",
        ],
        "Renewable Energy": [
            "Solar Park 2000MW", "Wind-Solar Hybrid Farm",
        ],
        "Heavy Industry": [
            "Steel Plant Modernization", "Pellet Plant Unit",
        ],
        "Oil & Gas": [
            "Pipeline Grid Extension", "CGD Network City Unit",
        ],
    }

    records = []
    for i in range(n):
        ministry = rng.choice(ministries)
        sector = sectors[ministry]
        name_pool = project_names_by_sector[sector]
        project_name = f"{rng.choice(name_pool)} {rng.integers(1, 9)}"
        state_name = rng.choice(states)

        original_cost = round(rng.uniform(150, 3500), 2)
        cost_overrun_factor = rng.choice(
            [1.0, rng.uniform(1.05, 1.15), rng.uniform(1.15, 1.45), rng.uniform(1.45, 1.8)],
            p=[0.2, 0.35, 0.3, 0.15],
        )
        revised_cost = round(original_cost * cost_overrun_factor, 2)

        # Dates
        start_date = datetime(2018, 1, 1) + timedelta(days=int(rng.integers(0, 1200)))
        original_duration = int(rng.integers(24, 72))
        original_completion = start_date + timedelta(months=original_duration) if False else start_date + timedelta(days=original_duration * 30)
        delay_m = int(rng.choice([0, 0, 0, rng.integers(3, 12), rng.integers(12, 36), rng.integers(36, 60)], p=[0.25, 0.2, 0.15, 0.2, 0.12, 0.08]))
        revised_completion = original_completion + timedelta(days=delay_m * 30)

        # Progress
        total_milestones = int(rng.integers(8, 25))
        elapsed_fraction = min(1.0, max(0.05, (datetime(2026, 4, 1) - start_date).days / max(1, (revised_completion - start_date).days)))
        expected_progress = round(elapsed_fraction * 100, 1)
        slippage = rng.uniform(0, 0.35) if delay_m > 0 else rng.uniform(0, 0.12)
        physical_progress = round(max(0, min(100, expected_progress * (1 - slippage) + rng.normal(0, 3))), 1)
        milestones_completed = int(round(total_milestones * (physical_progress / 100)))

        # Expenditure
        spend_ratio = (physical_progress / 100) + rng.normal(0, 0.08)
        cumulative_expenditure = round(max(0, revised_cost * spend_ratio), 2)

        # Delay reason
        if delay_m == 0:
            primary_delay_reason = "None"
        else:
            primary_delay_reason = rng.choice(delay_reasons[:-1])

        project_code = str(rng.integers(100000, 999999))

        records.append({
            "project_code": project_code,
            "project_name": project_name,
            "ministry": ministry,
            "sector": sector,
            "state": state_name,
            "original_cost_cr": original_cost,
            "revised_cost_cr": revised_cost,
            "cumulative_expenditure_cr": cumulative_expenditure,
            "physical_progress_pct": physical_progress,
            "milestones_completed": milestones_completed,
            "total_milestones": total_milestones,
            "original_completion_date": original_completion.strftime("%Y-%m-%d"),
            "revised_completion_date": revised_completion.strftime("%Y-%m-%d"),
            "delay_months": delay_m,
            "primary_delay_reason": primary_delay_reason,
        })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    df["cost_overrun_pct"] = ((df["revised_cost_cr"] - df["original_cost_cr"]) / df["original_cost_cr"]) * 100

    # expected progress based on time elapsed
    ref_date = datetime(2026, 4, 1)
    expected_list = []
    for _, row in df.iterrows():
        start = datetime.strptime(row["original_completion_date"], "%Y-%m-%d") - timedelta(days=24 * 30)
        end = datetime.strptime(row["revised_completion_date"], "%Y-%m-%d")
        total_days = max(1, (end - start).days)
        elapsed_days = max(0, (ref_date - start).days)
        expected_list.append(round(min(100, max(0, elapsed_days / total_days * 100)), 1))
    df["expected_progress_pct"] = expected_list

    df["progress_variance"] = df["expected_progress_pct"] - df["physical_progress_pct"]
    df["spend_deviation"] = (df["cumulative_expenditure_cr"] / df["revised_cost_cr"]) - (df["physical_progress_pct"] / 100)
    df["milestone_slippage"] = 1 - (df["milestones_completed"] / df["total_milestones"])

    # Risk scores
    df["cost_overrun_risk"] = np.clip(df["cost_overrun_pct"] * 0.8, 0, 100)
    df["schedule_delay_risk"] = np.clip(
        (df["delay_months"] / 60 * 50) + (df["progress_variance"].clip(lower=0) * 0.5),
        0, 100,
    )
    df["composite_risk_score"] = np.clip(
        0.4 * df["cost_overrun_risk"] + 0.4 * df["schedule_delay_risk"] + 0.2 * (df["milestone_slippage"] * 100),
        0, 100,
    )

    def _risk_band(score):
        if score >= 60:
            return "High"
        elif score >= 30:
            return "Medium"
        else:
            return "Low"

    df["risk_band"] = df["composite_risk_score"].apply(_risk_band)
    return df


# ---------------------------------------------------------------------------
# ML PIPELINE
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "cost_overrun_pct", "progress_variance", "spend_deviation",
    "milestone_slippage", "delay_months", "physical_progress_pct",
]


@st.cache_resource
def train_model(df):
    """Train classifier on risk_band and return model + feature importances / SHAP."""
    X = df[FEATURE_COLS].values
    y = df["risk_band"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if CATBOOST_AVAILABLE:
        model = _CatBoostClassifier(
            iterations=200, depth=4, learning_rate=0.1,
            verbose=0, random_seed=42,
        )
        model.fit(X_train_s, y_train)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    # Explainability
    shap_values = None
    feature_importances = None

    if SHAP_AVAILABLE:
        try:
            if CATBOOST_AVAILABLE:
                explainer = _shap.TreeExplainer(model)
            else:
                explainer = _shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test_s)
            # For multi-class, shap returns list or array
            if isinstance(sv, list):
                # average across classes
                shap_arr = np.mean(np.abs(np.array(sv)), axis=0)
            else:
                shap_arr = np.mean(np.abs(sv), axis=0)
            feature_importances = shap_arr
        except Exception:
            feature_importances = None

    if feature_importances is None:
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
        else:
            feature_importances = np.ones(len(FEATURE_COLS))

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "feature_importances": feature_importances,
        "feature_cols": FEATURE_COLS,
    }


def predict_risk(model_bundle, input_features):
    """Predict risk band + per-class probabilities for a single sample."""
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    X = np.array(input_features).reshape(1, -1)
    X_s = scaler.transform(X)
    pred = model.predict(X_s)
    pred_label = pred[0] if isinstance(pred, np.ndarray) else pred
    try:
        proba = model.predict_proba(X_s)[0]
        classes = model.classes_
        proba_dict = {c: float(p) for c, p in zip(classes, proba)}
    except Exception:
        proba_dict = {pred_label: 1.0}
    return pred_label, proba_dict


# ---------------------------------------------------------------------------
# HELPER: glass card wrapper
# ---------------------------------------------------------------------------
def glass_card(body_html):
    st.markdown(f'<div class="glass-card">{body_html}</div>', unsafe_allow_html=True)


def kpi_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(band):
    cls = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(band, "risk-low")
    return f'<span class="risk-badge {cls}">{band}</span>'


# ---------------------------------------------------------------------------
# DATA LOAD
# ---------------------------------------------------------------------------
raw_df = generate_dataset()
df = engineer_features(raw_df)
model_bundle = train_model(df)

# ---------------------------------------------------------------------------
# ROLE-BASED FILTERING
# ---------------------------------------------------------------------------
ROLE_OPTIONS = [
    "National / MoSPI Officer",
    "Ministry Admin",
    "Public / Observer",
]

MINISTRY_OPTIONS = sorted(df["ministry"].unique().tolist())

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ INFRAguard AI")
    st.markdown("**Predictive Intelligence Layer**")
    st.markdown("*Aligned with MoSPI PAIMANA (Apr 2026)*")
    st.markdown("---")

    st.session_state["user_role"] = st.selectbox(
        "Select Role", ROLE_OPTIONS,
        index=ROLE_OPTIONS.index(st.session_state["user_role"]),
    )

    scoped_ministry = None
    if st.session_state["user_role"] == "Ministry Admin":
        scoped_ministry = st.selectbox(
            "Assigned Ministry", MINISTRY_OPTIONS,
            index=0,
        )
    st.markdown("---")

    # Filter data by role
    if st.session_state["user_role"] == "Ministry Admin" and scoped_ministry:
        view_df = df[df["ministry"] == scoped_ministry].copy()
    else:
        view_df = df.copy()

    st.markdown(f"**Visible Projects:** {len(view_df)}")
    st.markdown(f"**Model Accuracy:** {model_bundle['accuracy']:.1%}")

    if CATBOOST_AVAILABLE:
        st.markdown("✅ CatBoost engine active")
    else:
        st.markdown("⚠️ Using RandomForest fallback")

    if SHAP_AVAILABLE:
        st.markdown("✅ SHAP explainability active")
    else:
        st.markdown("⚠️ Using feature_importances_ fallback")

    st.markdown("---")
    st.markdown("###### Quick Stats")
    if len(view_df) > 0:
        st.metric("High-Risk Projects", int((view_df["risk_band"] == "High").sum()))
        st.metric("Total Revised Cost (₹Cr)", f"{view_df['revised_cost_cr'].sum():,.0f}")
        st.metric("Avg Composite Risk", f"{view_df['composite_risk_score'].mean():.1f}")


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    '<p class="app-title">🛡️ INFRAguard AI</p>'
    '<p class="app-subtitle">Explainable Predictive Intelligence for Infrastructure Project Monitoring — MoSPI PAIMANA</p>',
    unsafe_allow_html=True,
)
st.markdown(f"**Active Role:** {st.session_state['user_role']}" + (f" — {scoped_ministry}" if scoped_ministry else ""))
st.markdown("---")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio Triage",
    "🔮 Prediction & Explainability",
    "🎛️ What-If Simulator",
    "📈 Breakdown & Export",
])

# ===========================================================================
# TAB 1 — PORTFOLIO TRIAGE & EARLY WARNINGS
# ===========================================================================
with tab1:
    st.markdown('<div class="tab-header">Portfolio Triage & Early Warnings</div>', unsafe_allow_html=True)

    # KPI Cards
    total_projects = len(view_df)
    high_risk_count = int((view_df["risk_band"] == "High").sum())
    total_sanctioned = view_df["original_cost_cr"].sum()
    total_revised = view_df["revised_cost_cr"].sum()
    total_spend = view_df["cumulative_expenditure_cr"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Monitored Projects", f"{total_projects}", "Central Sector ₹150Cr+")
    with c2:
        kpi_card("High-Risk Projects", f"{high_risk_count}", f"{high_risk_count/total_projects*100:.0f}% of portfolio" if total_projects else "")
    with c3:
        kpi_card("Sanctioned Cost (₹Cr)", f"{total_sanctioned:,.0f}", "Original")
    with c4:
        kpi_card("Revised Cost (₹Cr)", f"{total_revised:,.0f}", f"+{(total_revised-total_sanctioned)/total_sanctioned*100:.1f}% overrun")
    with c5:
        kpi_card("Cumulative Spend (₹Cr)", f"{total_spend:,.0f}", f"{total_spend/total_revised*100:.1f}% of revised")

    st.markdown("")

    # Scatter: Physical Progress vs Expenditure % colored by Composite Risk
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Physical Progress vs Expenditure — Colored by Composite Risk")
    plot_df = view_df.copy()
    plot_df["expenditure_pct"] = (plot_df["cumulative_expenditure_cr"] / plot_df["revised_cost_cr"]) * 100
    plot_df["expenditure_pct"] = plot_df["expenditure_pct"].clip(0, 150)

    fig_scatter = px.scatter(
        plot_df,
        x="physical_progress_pct",
        y="expenditure_pct",
        size="revised_cost_cr",
        color="composite_risk_score",
        color_continuous_scale=["#34D399", "#FBBF24", "#F87171"],
        hover_data=["project_code", "project_name", "ministry", "risk_band"],
        labels={
            "physical_progress_pct": "Physical Progress (%)",
            "expenditure_pct": "Expenditure (%)",
            "composite_risk_score": "Risk Score",
        },
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2d2d44", size=12),
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig_scatter.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="white")))
    # Reference line y=x
    fig_scatter.add_trace(
        go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                   line=dict(dash="dash", color="#94A3B8", width=1.5),
                   showlegend=False, hoverinfo="skip")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Priority Early-Warning Table
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🚨 Priority Early-Warning Queue")
    st.markdown("Projects sorted by highest composite risk score.")

    priority_df = view_df.sort_values("composite_risk_score", ascending=False).head(15)
    display_cols = [
        "project_code", "project_name", "ministry", "state",
        "original_cost_cr", "revised_cost_cr", "physical_progress_pct",
        "delay_months", "primary_delay_reason", "composite_risk_score", "risk_band",
    ]
    table_df = priority_df[display_cols].copy()
    table_df["composite_risk_score"] = table_df["composite_risk_score"].round(1)
    table_df.columns = [
        "Code", "Project", "Ministry", "State",
        "Orig. Cost (₹Cr)", "Rev. Cost (₹Cr)", "Progress (%)",
        "Delay (mo)", "Delay Reason", "Risk Score", "Risk Band",
    ]

    # Render as HTML with badges
    rows_html = ""
    for _, row in table_df.iterrows():
        badge = risk_badge_html(row["Risk Band"])
        rows_html += f"""
        <tr>
            <td style="text-align:center;font-weight:600">{row['Code']}</td>
            <td>{row['Project']}</td>
            <td>{row['Ministry']}</td>
            <td>{row['State']}</td>
            <td style="text-align:right">{row['Orig. Cost (₹Cr)']:,.1f}</td>
            <td style="text-align:right">{row['Rev. Cost (₹Cr)']:,.1f}</td>
            <td style="text-align:center">{row['Progress (%)']:.1f}</td>
            <td style="text-align:center">{row['Delay (mo)']}</td>
            <td>{row['Delay Reason']}</td>
            <td style="text-align:center;font-weight:700">{row['Risk Score']:.1f}</td>
            <td style="text-align:center">{badge}</td>
        </tr>
        """

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;color:#2d2d44;">
        <thead>
            <tr style="background:rgba(106,17,203,0.12);color:#4b5563;">
                <th style="padding:8px 6px;text-align:center">Code</th>
                <th style="padding:8px 6px;text-align:left">Project</th>
                <th style="padding:8px 6px;text-align:left">Ministry</th>
                <th style="padding:8px 6px;text-align:left">State</th>
                <th style="padding:8px 6px;text-align:right">Orig. Cost</th>
                <th style="padding:8px 6px;text-align:right">Rev. Cost</th>
                <th style="padding:8px 6px;text-align:center">Progress</th>
                <th style="padding:8px 6px;text-align:center">Delay</th>
                <th style="padding:8px 6px;text-align:left">Delay Reason</th>
                <th style="padding:8px 6px;text-align:center">Risk</th>
                <th style="padding:8px 6px;text-align:center">Band</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 2 — PREDICTION & EXPLAINABILITY
# ===========================================================================
with tab2:
    st.markdown('<div class="tab-header">Prediction & Explainability</div>', unsafe_allow_html=True)

    # Project selector filtered by role
    project_options = view_df[["project_code", "project_name"]].copy()
    project_options["label"] = project_options["project_code"] + " — " + project_options["project_name"]

    if len(project_options) == 0:
        st.warning("No projects visible for this role.")
    else:
        default_idx = 0
        if st.session_state["selected_project_code"] and st.session_state["selected_project_code"] in project_options["project_code"].values:
            default_idx = int(project_options["project_code"].tolist().index(st.session_state["selected_project_code"]))

        selected_label = st.selectbox(
            "Select Project for Deep-Dive Prediction",
            project_options["label"].tolist(),
            index=default_idx,
        )
        selected_code = selected_label.split(" — ")[0]
        st.session_state["selected_project_code"] = selected_code

        sel_row = view_df[view_df["project_code"] == selected_code].iloc[0]

        # Project info card
        info_html = f"""
        <div class="glass-card">
            <h4 style="margin:0 0 8px 0;color:#2d2d44">{sel_row['project_name']}</h4>
            <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:13px;color:#4b5563;">
                <div><b>Code:</b> {sel_row['project_code']}</div>
                <div><b>Ministry:</b> {sel_row['ministry']}</div>
                <div><b>Sector:</b> {sel_row['sector']}</div>
                <div><b>State:</b> {sel_row['state']}</div>
                <div><b>Original Cost:</b> ₹{sel_row['original_cost_cr']:,.1f} Cr</div>
                <div><b>Revised Cost:</b> ₹{sel_row['revised_cost_cr']:,.1f} Cr</div>
                <div><b>Progress:</b> {sel_row['physical_progress_pct']:.1f}%</div>
                <div><b>Delay:</b> {sel_row['delay_months']} months</div>
                <div><b>Reason:</b> {sel_row['primary_delay_reason']}</div>
            </div>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)

        # Compute prediction
        input_features = [
            float(sel_row["cost_overrun_pct"]),
            float(sel_row["progress_variance"]),
            float(sel_row["spend_deviation"]),
            float(sel_row["milestone_slippage"]),
            float(sel_row["delay_months"]),
            float(sel_row["physical_progress_pct"]),
        ]
        pred_label, proba_dict = predict_risk(model_bundle, input_features)

        # Gauge charts
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Risk Gauges")
        g1, g2, g3 = st.columns(3)

        def make_gauge(title, value, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": title, "font": {"size": 14, "color": "#2d2d44"}},
                number={"font": {"size": 36, "color": "#2d2d44"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                    "bar": {"color": color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(52,211,153,0.25)"},
                        {"range": [30, 60], "color": "rgba(251,191,36,0.25)"},
                        {"range": [60, 100], "color": "rgba(248,113,113,0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#2d2d44", "width": 2},
                        "thickness": 0.85,
                        "value": value,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=220,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            return fig

        with g1:
            st.plotly_chart(make_gauge("Cost Overrun Risk", sel_row["cost_overrun_risk"], "#A78BFA"), use_container_width=True)
        with g2:
            st.plotly_chart(make_gauge("Schedule Delay Risk", sel_row["schedule_delay_risk"], "#60A5FA"), use_container_width=True)
        with g3:
            st.plotly_chart(make_gauge("Composite Risk Score", sel_row["composite_risk_score"], "#F472B6"), use_container_width=True)

        st.markdown(f"**Model Prediction:** {risk_badge_html(pred_label)} &nbsp; Confidence: {max(proba_dict.values()):.1%}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("")

        # Feature importance / SHAP bar chart
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Top Risk Drivers — Feature Attribution")

        fi = model_bundle["feature_importances"]
        # Normalize
        fi_norm = fi / fi.sum() * 100 if fi.sum() > 0 else fi
        fi_labels = {
            "cost_overrun_pct": "Cost Overrun %",
            "progress_variance": "Progress Variance",
            "spend_deviation": "Spend Deviation",
            "milestone_slippage": "Milestone Slippage",
            "delay_months": "Delay (months)",
            "physical_progress_pct": "Physical Progress %",
        }
        fi_names = [fi_labels.get(c, c) for c in model_bundle["feature_cols"]]

        fi_df = pd.DataFrame({"feature": fi_names, "importance": fi_norm}).sort_values("importance", ascending=True)

        fig_bar = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker=dict(
                color=fi_df["importance"],
                colorscale=[[0, "#A78BFA"], [0.5, "#60A5FA"], [1, "#F472B6"]],
            ),
            text=[f"{v:.1f}%" for v in fi_df["importance"]],
            textposition="outside",
            textfont=dict(color="#4b5563", size=12),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#2d2d44", size=12),
            xaxis=dict(title="Relative Attribution (%)", gridcolor="rgba(148,163,184,0.2)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=320,
            margin=dict(l=10, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if SHAP_AVAILABLE:
            st.caption("Attribution computed via SHAP TreeExplainer.")
        else:
            st.caption("SHAP unavailable — attribution from tree feature_importances_.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Engineered features breakdown
        st.markdown("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Engineered Feature Breakdown")
        ef1, ef2, ef3, ef4 = st.columns(4)
        with ef1:
            st.metric("Cost Overrun %", f"{sel_row['cost_overrun_pct']:.1f}%")
        with ef2:
            st.metric("Progress Variance", f"{sel_row['progress_variance']:.1f}")
        with ef3:
            st.metric("Spend Deviation", f"{sel_row['spend_deviation']:.3f}")
        with ef4:
            st.metric("Milestone Slippage", f"{sel_row['milestone_slippage']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 3 — WHAT-IF SCENARIO SIMULATOR
# ===========================================================================
with tab3:
    st.markdown('<div class="tab-header">What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown("Adjust project parameters to simulate intervention impact on risk scores.")

    if len(view_df) == 0:
        st.warning("No projects visible for this role.")
    else:
        sim_project_labels = view_df[["project_code", "project_name"]].copy()
        sim_project_labels["label"] = sim_project_labels["project_code"] + " — " + sim_project_labels["project_name"]

        sim_sel_label = st.selectbox(
            "Select Project to Simulate",
            sim_project_labels["label"].tolist(),
            key="sim_project_select",
        )
        sim_code = sim_sel_label.split(" — ")[0]
        sim_row = view_df[view_df["project_code"] == sim_code].iloc[0]

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"**Project:** {sim_row['project_name']} ({sim_row['project_code']})")
        st.markdown(f"**Ministry:** {sim_row['ministry']} | **State:** {sim_row['state']}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Baseline values
        base_progress = float(sim_row["physical_progress_pct"])
        base_milestones_completed = int(sim_row["milestones_completed"])
        base_total_milestones = int(sim_row["total_milestones"])
        base_delay = int(sim_row["delay_months"])
        base_delay_reason = str(sim_row["primary_delay_reason"])
        base_cost_overrun_pct = float(sim_row["cost_overrun_pct"])
        base_spend = float(sim_row["cumulative_expenditure_cr"])
        base_revised_cost = float(sim_row["revised_cost_cr"])

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Intervention Parameters")
            sim_progress = st.slider(
                "Physical Progress (%)",
                min_value=0.0, max_value=100.0,
                value=base_progress, step=1.0,
                help="Simulate improved on-ground execution",
            )
            sim_milestones = st.slider(
                "Milestones Completed",
                min_value=0, max_value=base_total_milestones,
                value=base_milestones_completed, step=1,
                help="Recover delayed milestones",
            )
            sim_delay = st.slider(
                "Delay Reduction (months)",
                min_value=0, max_value=max(base_delay, 1),
                value=0, step=1,
                help="Months of delay recovered through intervention",
            )
            sim_reason = st.selectbox(
                "Bottleneck Cleared",
                ["No change", "Land Acquisition", "Forest Clearances",
                 "Utility Shifting", "Contracting Issue"],
                index=0,
                help="Clearing a primary bottleneck",
            )
            sim_cost_reduction = st.slider(
                "Cost Overrun Reduction (%)",
                min_value=0.0, max_value=max(base_cost_overrun_pct, 1.0),
                value=0.0, step=1.0,
                help="Reduce cost overrun through better cost control",
            )

        # Compute simulated features
        sim_delay_months = max(0, base_delay - sim_delay)
        sim_cost_overrun_pct = max(0, base_cost_overrun_pct - sim_cost_reduction)
        sim_progress_variance = float(sim_row["expected_progress_pct"]) - sim_progress
        sim_milestone_slippage = 1 - (sim_milestones / base_total_milestones)
        # Adjusted expenditure: if progress improves, spend adjusts proportionally
        sim_spend = base_spend * (sim_progress / max(base_progress, 1.0)) if base_progress > 0 else base_spend
        sim_spend_deviation = (sim_spend / base_revised_cost) - (sim_progress / 100)

        # Simulated risk scores
        sim_cost_risk = np.clip(sim_cost_overrun_pct * 0.8, 0, 100)
        sim_schedule_risk = np.clip(
            (sim_delay_months / 60 * 50) + (max(0, sim_progress_variance) * 0.5),
            0, 100,
        )
        sim_composite = np.clip(
            0.4 * sim_cost_risk + 0.4 * sim_schedule_risk + 0.2 * (sim_milestone_slippage * 100),
            0, 100,
        )

        # Predict with model
        sim_features = [
            sim_cost_overrun_pct, sim_progress_variance, sim_spend_deviation,
            sim_milestone_slippage, sim_delay_months, sim_progress,
        ]
        sim_pred, sim_proba = predict_risk(model_bundle, sim_features)

        # Baseline prediction
        base_features = [
            base_cost_overrun_pct, float(sim_row["progress_variance"]),
            float(sim_row["spend_deviation"]), float(sim_row["milestone_slippage"]),
            base_delay, base_progress,
        ]
        base_pred, base_proba = predict_risk(model_bundle, base_features)

        with col_right:
            st.markdown("#### Side-by-Side Comparison")

            bc, sc = st.columns(2)
            with bc:
                st.markdown(
                    f"""
                    <div class="compare-card compare-baseline">
                        <div style="font-size:13px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;">Baseline</div>
                        <div style="font-size:36px;font-weight:800;color:#6a11cb;margin:6px 0;">{sim_row['composite_risk_score']:.1f}</div>
                        <div style="font-size:13px;color:#4b5563;">{risk_badge_html(base_pred)}</div>
                        <div style="margin-top:10px;font-size:12px;color:#6b7280;">
                            Cost Risk: {sim_row['cost_overrun_risk']:.1f}<br>
                            Schedule Risk: {sim_row['schedule_delay_risk']:.1f}<br>
                            Delay: {base_delay} months
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sc:
                delta = sim_composite - sim_row["composite_risk_score"]
                delta_color = "#15803d" if delta < 0 else "#b91c1c"
                st.markdown(
                    f"""
                    <div class="compare-card compare-simulated">
                        <div style="font-size:13px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;">Simulated</div>
                        <div style="font-size:36px;font-weight:800;color:#2575fc;margin:6px 0;">{sim_composite:.1f}</div>
                        <div style="font-size:13px;color:#4b5563;">{risk_badge_html(sim_pred)}</div>
                        <div style="margin-top:10px;font-size:12px;color:#6b7280;">
                            Cost Risk: {sim_cost_risk:.1f}<br>
                            Schedule Risk: {sim_schedule_risk:.1f}<br>
                            Delay: {sim_delay_months} months
                        </div>
                        <div style="margin-top:8px;font-size:14px;font-weight:700;color:{delta_color};">
                            {'▼' if delta < 0 else '▲'} {abs(delta):.1f} pts
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Delta chart
        st.markdown("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Risk Score Delta — Baseline vs Simulated")

        delta_fig = go.Figure()
        categories = ["Cost Overrun Risk", "Schedule Delay Risk", "Composite Risk"]
        baseline_vals = [sim_row["cost_overrun_risk"], sim_row["schedule_delay_risk"], sim_row["composite_risk_score"]]
        sim_vals = [sim_cost_risk, sim_schedule_risk, sim_composite]

        delta_fig.add_trace(go.Bar(
            x=categories, y=baseline_vals, name="Baseline",
            marker_color="#A78BFA", opacity=0.8,
            text=[f"{v:.1f}" for v in baseline_vals], textposition="outside",
        ))
        delta_fig.add_trace(go.Bar(
            x=categories, y=sim_vals, name="Simulated",
            marker_color="#60A5FA", opacity=0.8,
            text=[f"{v:.1f}" for v in sim_vals], textposition="outside",
        ))
        delta_fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#2d2d44", size=12),
            yaxis=dict(title="Risk Score", range=[0, 100], gridcolor="rgba(148,163,184,0.2)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(delta_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Intervention summary
        st.markdown("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Intervention Summary")
        interventions = []
        if sim_progress > base_progress:
            interventions.append(f"Physical progress improved from {base_progress:.1f}% to {sim_progress:.1f}% (+{sim_progress-base_progress:.1f}pp)")
        if sim_milestones > base_milestones_completed:
            interventions.append(f"Milestones recovered: {base_milestones_completed} → {sim_milestones} (+{sim_milestones-base_milestones_completed})")
        if sim_delay > 0:
            interventions.append(f"Delay reduced by {sim_delay} months ({base_delay} → {sim_delay_months})")
        if sim_cost_reduction > 0:
            interventions.append(f"Cost overrun reduced by {sim_cost_reduction:.1f}pp ({base_cost_overrun_pct:.1f}% → {sim_cost_overrun_pct:.1f}%)")
        if sim_reason != "No change":
            interventions.append(f"Bottleneck cleared: {sim_reason}")

        if interventions:
            for item in interventions:
                st.markdown(f"✅ {item}")
            st.markdown(f"**Net Risk Reduction:** {abs(delta):.1f} points ({'improvement' if delta < 0 else 'worsening'})")
        else:
            st.info("Adjust sliders on the left to simulate interventions and see risk impact.")
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 4 — PORTFOLIO BREAKDOWN & EXPORT
# ===========================================================================
with tab4:
    st.markdown('<div class="tab-header">Portfolio Breakdown & Export</div>', unsafe_allow_html=True)

    # Ministry-wise distribution
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Ministry-wise Distribution — Composite Risk")
    ministry_agg = view_df.groupby("ministry").agg(
        project_count=("project_code", "count"),
        avg_risk=("composite_risk_score", "mean"),
        total_revised=("revised_cost_cr", "sum"),
        high_risk_count=("risk_band", lambda x: (x == "High").sum()),
    ).reset_index().sort_values("avg_risk", ascending=True)

    fig_min = go.Figure()
    fig_min.add_trace(go.Bar(
        y=ministry_agg["ministry"],
        x=ministry_agg["avg_risk"],
        orientation="h",
        marker=dict(
            color=ministry_agg["avg_risk"],
            colorscale=[[0, "#34D399"], [0.5, "#FBBF24"], [1, "#F87171"]],
        ),
        text=[f"{v:.1f}" for v in ministry_agg["avg_risk"]],
        textposition="outside",
        textfont=dict(color="#4b5563", size=11),
        customdata=ministry_agg[["project_count", "high_risk_count"]],
        hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.1f}<br>Projects: %{customdata[0]}<br>High-Risk: %{customdata[1]}<extra></extra>",
    ))
    fig_min.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2d2d44", size=12),
        xaxis=dict(title="Average Composite Risk Score", range=[0, 100], gridcolor="rgba(148,163,184,0.2)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=max(350, len(ministry_agg) * 35),
        margin=dict(l=10, r=60, t=20, b=20),
    )
    st.plotly_chart(fig_min, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Sector-wise distribution pie
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Sector-wise Project Distribution")
    sector_counts = view_df["sector"].value_counts().reset_index()
    sector_counts.columns = ["sector", "count"]

    fig_pie = go.Figure(go.Pie(
        labels=sector_counts["sector"],
        values=sector_counts["count"],
        hole=0.45,
        marker=dict(colors=PASTEL_COLORS[:len(sector_counts)]),
        textinfo="label+percent",
        textfont=dict(color="#2d2d44", size=12),
    ))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2d2d44", size=12),
        height=380,
        margin=dict(l=10, r=10, t=20, b=20),
        showlegend=True,
        legend=dict(font=dict(color="#4b5563", size=11)),
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Delay reason distribution
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Primary Delay Reasons — Distribution")
    reason_counts = view_df["primary_delay_reason"].value_counts().reset_index()
    reason_counts.columns = ["reason", "count"]

    fig_reason = go.Figure(go.Bar(
        x=reason_counts["reason"],
        y=reason_counts["count"],
        marker=dict(color=PASTEL_COLORS[:len(reason_counts)]),
        text=reason_counts["count"],
        textposition="outside",
        textfont=dict(color="#4b5563", size=12),
    ))
    fig_reason.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2d2d44", size=12),
        xaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Project Count", gridcolor="rgba(148,163,184,0.2)"),
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_reason, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Summary table + CSV export
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Full Portfolio Summary")

    export_df = view_df[[
        "project_code", "project_name", "ministry", "sector", "state",
        "original_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr",
        "physical_progress_pct", "milestones_completed", "total_milestones",
        "delay_months", "primary_delay_reason",
        "cost_overrun_pct", "progress_variance", "spend_deviation", "milestone_slippage",
        "cost_overrun_risk", "schedule_delay_risk", "composite_risk_score", "risk_band",
    ]].copy()

    export_df.columns = [
        "Project Code", "Project Name", "Ministry", "Sector", "State",
        "Original Cost (Cr)", "Revised Cost (Cr)", "Cumulative Expenditure (Cr)",
        "Physical Progress (%)", "Milestones Completed", "Total Milestones",
        "Delay (Months)", "Primary Delay Reason",
        "Cost Overrun (%)", "Progress Variance", "Spend Deviation", "Milestone Slippage",
        "Cost Overrun Risk", "Schedule Delay Risk", "Composite Risk Score", "Risk Band",
    ]

    # Round numeric columns
    numeric_cols = export_df.select_dtypes(include=[np.number]).columns
    export_df[numeric_cols] = export_df[numeric_cols].round(2)

    st.dataframe(export_df, use_container_width=True, height=450)

    # CSV export
    csv_data = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Portfolio Report (CSV)",
        data=csv_data.encode("utf-8"),
        file_name=f"infraguard_ai_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.markdown(
        f"*Report generated for role: **{st.session_state['user_role']}**"
        + (f" — {scoped_ministry}" if scoped_ministry else "")
        + f" | {len(export_df)} projects*"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#6b7280;font-size:12px;">
        INFRAguard AI — Explainable Predictive Intelligence Layer for Infrastructure Monitoring<br>
        Aligned with MoSPI PAIMANA (April 2026) | Synthetic data for prototype demonstration
    </div>
    """,
    unsafe_allow_html=True,
)
