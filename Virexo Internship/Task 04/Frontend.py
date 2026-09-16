"""
Student Performance Lab
A Streamlit interface for the Student Performance Analysis & Prediction
System: data cleaning, EDA, a single interpretable Linear Regression model,
and a live, explainable prediction tool.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Design tokens & global style
# ----------------------------------------------------------------------
INK = "#1B2A41"
MUTED = "#5B6472"
BG = "#F8F6F0"
PANEL = "#EFEBDD"
LINE = "#DDD7C6"
NAVY = "#2B3A67"
NAVY_DEEP = "#1B2747"
NAVY_WASH = "#E7EAF2"
GOLD = "#B8862B"
GOLD_DEEP = "#8C6320"
GOLD_WASH = "#F3E9D2"
CLAY = "#B5562F"
GREEN = "#3F7D5C"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": LINE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
})

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {INK};
}}
.stApp {{ background-color: {BG}; }}

section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}

h1, h2, h3, h4 {{
    font-family: 'Source Serif 4', serif;
    color: {INK};
    letter-spacing: -0.01em;
}}

/* ---- Report header ---- */
.lab-header {{
    padding-bottom: 1.1rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid {LINE};
}}
.lab-kicker {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: {GOLD_DEEP};
    letter-spacing: 0.02em;
    margin-bottom: 0.3rem;
}}
.lab-title {{
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 2.3rem;
    line-height: 1.15;
    margin: 0 0 0.35rem 0;
    color: {INK};
}}
.lab-subtitle {{
    font-size: 0.98rem;
    color: {MUTED};
    max-width: 64ch;
    line-height: 1.5;
    margin-bottom: 0.9rem;
}}

.stat-row {{ display: flex; flex-wrap: wrap; gap: 0; margin-top: 0.6rem; }}
.stat-block {{ padding: 0 1.4rem; border-left: 1px solid {LINE}; }}
.stat-block:first-child {{ padding-left: 0; border-left: none; }}
.stat-figure {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 500;
    color: {NAVY_DEEP};
    line-height: 1.1;
}}
.stat-label {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.15rem; }}

.section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: {GOLD_DEEP};
    margin-bottom: 0.15rem;
}}
.section-title {{
    font-family: 'Source Serif 4', serif;
    font-size: 1.45rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
}}
.section-note {{
    color: {MUTED};
    font-size: 0.92rem;
    line-height: 1.55;
    max-width: 82ch;
    margin-bottom: 1rem;
}}

.verdict {{
    border: 1px solid {LINE};
    border-left: 4px solid {NAVY};
    background: {NAVY_WASH};
    padding: 1rem 1.3rem;
    margin: 0.6rem 0 1.2rem 0;
}}
.verdict b {{ color: {NAVY_DEEP}; }}

.note-box {{
    border: 1px solid {LINE};
    border-left: 4px solid {GOLD};
    background: {GOLD_WASH};
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0 1.1rem 0;
    font-size: 0.92rem;
    color: {INK};
    line-height: 1.5;
}}

button[data-baseweb="tab"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.95rem;
    color: {MUTED};
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {NAVY_DEEP};
    font-weight: 600;
}}
div[data-baseweb="tab-highlight"] {{ background-color: {NAVY} !important; }}
div[data-baseweb="tab-border"] {{ background-color: {LINE} !important; }}

/* ---- Result card for the score prediction ---- */
.score-card {{
    border: 1px solid {LINE};
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}}
.score-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: {MUTED};
    margin-bottom: 0.3rem;
}}
.score-figure {{
    font-family: 'Source Serif 4', serif;
    font-size: 3.2rem;
    font-weight: 600;
    color: {NAVY_DEEP};
    line-height: 1;
}}
.score-band {{
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.2rem 0.7rem;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
}}
.score-band.strong {{ background: #E7F1EA; color: {GREEN}; }}
.score-band.ontrack {{ background: {GOLD_WASH}; color: {GOLD_DEEP}; }}
.score-band.support {{ background: #F7E9E2; color: {CLAY}; }}

.contrib-row {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.35rem 0;
    font-size: 0.86rem;
}}
.contrib-label {{ width: 190px; color: {INK}; flex-shrink: 0; }}
.contrib-track {{
    flex-grow: 1;
    height: 18px;
    background: {LINE};
    position: relative;
}}
.contrib-fill {{ height: 18px; position: absolute; top: 0; }}
.contrib-fill.pos {{ background: {NAVY}; left: 50%; }}
.contrib-fill.neg {{ background: {CLAY}; right: 50%; }}
.contrib-center {{
    position: absolute; left: 50%; top: 0; bottom: 0;
    width: 1px; background: {INK}; opacity: 0.35;
}}
.contrib-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: {MUTED};
    width: 48px;
    text-align: right;
}}

.stButton button {{
    background-color: white;
    color: {NAVY_DEEP};
    border: 1px solid {NAVY};
    border-radius: 2px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
}}
.stButton button:hover {{
    background-color: {NAVY_WASH};
    border-color: {NAVY};
}}

[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; }}
footer {{ visibility: hidden; }}

div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {{ color: {MUTED}; }}
div[data-testid="stThumbValue"] {{ color: {NAVY_DEEP} !important; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Data & model pipeline (cached)
# ----------------------------------------------------------------------
FEATURE_COLS = [
    "study_hours_per_week", "attendance_rate", "previous_grade", "sleep_hours",
    "extracurricular_count", "part_time_job", "internet_access",
    "parental_education_encoded", "study_group",
]
FEATURE_LABELS = {
    "study_hours_per_week": "Study hours / week",
    "attendance_rate": "Attendance rate",
    "previous_grade": "Previous grade",
    "sleep_hours": "Sleep hours",
    "extracurricular_count": "Extracurriculars",
    "part_time_job": "Part-time job",
    "internet_access": "Internet access",
    "parental_education_encoded": "Parental education",
    "study_group": "Study group",
}
PARENT_ED_ORDER = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
PARENT_ED_LEVELS = list(PARENT_ED_ORDER.keys())


@st.cache_data
def load_raw():
    return pd.read_csv("student_performance_dataset.csv")


@st.cache_data
def clean_data(df_raw: pd.DataFrame):
    n_raw = len(df_raw)
    df = df_raw.drop_duplicates().copy()
    n_after_dedup = len(df)
    missing_before = df.isnull().sum()
    missing_before = missing_before[missing_before > 0]

    for col in ["attendance_rate", "previous_grade", "sleep_hours"]:
        df[col] = df[col].fillna(df[col].median())
    df["parental_education"] = df["parental_education"].fillna(df["parental_education"].mode()[0])

    df["parental_education_encoded"] = df["parental_education"].map(PARENT_ED_ORDER)

    return df, {
        "n_raw": n_raw, "n_after_dedup": n_after_dedup,
        "n_duplicates_removed": n_raw - n_after_dedup,
        "missing_before": missing_before,
    }


@st.cache_resource(show_spinner="Training model…")
def train_model(_df: pd.DataFrame, test_size: float, random_state: int):
    X = _df[FEATURE_COLS]
    y = _df["final_exam_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }
    coefs = pd.Series(model.coef_, index=FEATURE_COLS)

    return {
        "model": model, "scaler": scaler,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "metrics": metrics, "coefs": coefs,
        "feature_means": X.mean(), "feature_stds": X.std(),
    }


# ----------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuration")
    st.caption("Adjust the pipeline and the model retrains automatically.")

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05,
                           help="Fraction of students held out for evaluation.")
    random_state = st.number_input("Random seed", value=42, min_value=0, step=1,
                                    help="Controls the train/test split.")

    st.markdown("---")
    st.markdown("### Dataset")
    st.caption(
        "812 synthetic student records built to resemble a realistic academic "
        "survey export — study habits, attendance, prior grades, lifestyle, "
        "and family background — including missing values and duplicate rows."
    )

df_raw = load_raw()
df, clean_info = clean_data(df_raw)
bundle = train_model(df, test_size, int(random_state))
metrics = bundle["metrics"]

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="lab-header">
    <div class="lab-kicker">AI / ML — student performance analysis &amp; prediction system</div>
    <div class="lab-title">Student Performance Lab</div>
    <div class="lab-subtitle">
        A single, fully explainable Linear Regression model that predicts a student's
        final exam score from study habits, attendance, and background factors —
        cleaned, evaluated, and ready to test on a new student.
    </div>
    <div class="stat-row">
        <div class="stat-block">
            <div class="stat-figure">{clean_info['n_after_dedup']}</div>
            <div class="stat-label">students (cleaned)</div>
        </div>
        <div class="stat-block">
            <div class="stat-figure">{len(FEATURE_COLS)}</div>
            <div class="stat-label">features</div>
        </div>
        <div class="stat-block">
            <div class="stat-figure">{metrics['mae']:.2f}</div>
            <div class="stat-label">MAE (points)</div>
        </div>
        <div class="stat-block">
            <div class="stat-figure">{metrics['r2']:.3f}</div>
            <div class="stat-label">R\u00b2</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_overview, tab_cleaning, tab_eda, tab_model, tab_explain, tab_predict = st.tabs(
    ["Overview", "Data cleaning", "Exploratory analysis", "Model & evaluation",
     "Feature influence", "Predict a score"]
)

# ---- Overview ----------------------------------------------------------
with tab_overview:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">01</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset overview</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Each row is one student, described by weekly study hours, '
            'attendance rate, previous grade, sleep habits, extracurricular involvement, part-time '
            'employment, internet access, parental education, and study-group participation. The '
            'target is <b>final_exam_score</b> (0-100) — a regression problem.</div>',
            unsafe_allow_html=True
        )
        st.dataframe(df_raw.head(8), use_container_width=True, height=270)
    with right:
        st.markdown('<div class="section-label">Score distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.8, 3.7))
        sns.histplot(df["final_exam_score"], bins=25, kde=True, color=NAVY, ax=ax)
        ax.set_xlabel("Final exam score")
        ax.set_ylabel("")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        st.caption(
            f"Mean {df['final_exam_score'].mean():.1f} · "
            f"Median {df['final_exam_score'].median():.1f} · "
            f"Std {df['final_exam_score'].std():.1f}"
        )

# ---- Data cleaning -------------------------------------------------------
with tab_cleaning:
    st.markdown('<div class="section-label">02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data cleaning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Real survey exports arrive messy. This dataset was built the '
        'same way, with duplicate rows and scattered missing values that get detected and cleaned '
        'below before any modeling happens.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
        <div class="verdict">
            <b>{clean_info['n_duplicates_removed']} duplicate rows</b> removed
            ({clean_info['n_raw']} \u2192 {clean_info['n_after_dedup']} rows).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Missing values found (before cleaning)**")
        miss_df = clean_info["missing_before"].rename("Missing values").to_frame()
        st.dataframe(miss_df, use_container_width=True)
        st.caption("Numeric columns imputed with the median; the categorical column with the mode.")
    with c2:
        st.markdown("**Cleaned data preview**")
        st.dataframe(df[FEATURE_COLS + ["final_exam_score"]].head(10), use_container_width=True, height=330)

# ---- EDA ------------------------------------------------------------------
with tab_eda:
    st.markdown('<div class="section-label">03</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Exploratory data analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Where the signal lives, before any model is trained.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.markdown("**Feature correlation with final score**")
        corr = df[FEATURE_COLS + ["final_exam_score"]].corr()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        cmap = sns.diverging_palette(15, 220, s=60, l=50, as_cmap=True)
        sns.heatmap(corr, cmap=cmap, center=0, annot=True, fmt=".2f",
                    annot_kws={"size": 7.5}, cbar_kws={"shrink": 0.75}, ax=ax,
                    xticklabels=[FEATURE_LABELS.get(c, c) for c in corr.columns],
                    yticklabels=[FEATURE_LABELS.get(c, c) for c in corr.columns])
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.markdown("**Study hours vs. final score**")
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        sns.scatterplot(data=df, x="study_hours_per_week", y="final_exam_score",
                         alpha=0.35, color=NAVY, ax=ax, s=22)
        sns.regplot(data=df, x="study_hours_per_week", y="final_exam_score",
                     scatter=False, color=GOLD_DEEP, ax=ax)
        ax.set_xlabel("Study hours / week")
        ax.set_ylabel("Final exam score")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    st.markdown(
        '<div class="note-box">Previous grade and weekly study hours show the strongest positive '
        'correlation with final score — both relationships look roughly linear, which supports using '
        'a linear model rather than a more complex one.</div>',
        unsafe_allow_html=True
    )

# ---- Model & evaluation ----------------------------------------------------
with tab_model:
    st.markdown('<div class="section-label">04</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model &amp; evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">A single <b>Linear Regression</b> model is trained on standardized '
        'features. It is a suitable choice here: the EDA shows an approximately linear relationship '
        'between the key predictors and the target, and a linear model keeps every prediction traceable '
        'to a readable set of coefficients — see the Feature influence tab.</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    for col, label, val, suffix in [
        (m1, "Mean Absolute Error", metrics["mae"], " pts"),
        (m2, "Root Mean Squared Error", metrics["rmse"], " pts"),
        (m3, "R\u00b2", metrics["r2"], ""),
    ]:
        col.markdown(f"""
        <div class="score-card" style="padding:1rem 1.2rem;">
            <div class="score-eyebrow">{label}</div>
            <div class="score-figure" style="font-size:2rem;">{val:.3f}{suffix}</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Actual vs. predicted**")
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(bundle["y_test"], bundle["y_pred"], alpha=0.5, color=NAVY, s=22)
        ax.plot([0, 100], [0, 100], "--", color=CLAY, linewidth=1.3)
        ax.set_xlabel("Actual score")
        ax.set_ylabel("Predicted score")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.markdown("**Residuals**")
        residuals = bundle["y_test"] - bundle["y_pred"]
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(bundle["y_pred"], residuals, alpha=0.5, color=NAVY, s=22)
        ax.axhline(0, color=CLAY, linestyle="--", linewidth=1.3)
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Residual (actual \u2212 predicted)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    st.caption(
        "Residuals scattered evenly around zero, with no obvious funnel or curve, "
        "indicate the model's linear assumptions are reasonably well met."
    )

# ---- Feature influence ------------------------------------------------------
with tab_explain:
    st.markdown('<div class="section-label">05</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature influence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Because features were standardized before fitting, each '
        'coefficient is directly comparable — this is the model\'s full explanation for its own '
        'behavior, not an approximation.</div>',
        unsafe_allow_html=True
    )

    coefs = bundle["coefs"].rename(index=FEATURE_LABELS).sort_values(key=abs)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [NAVY if v > 0 else CLAY for v in coefs.values]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color=INK, linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Standardized coefficient")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    top_pos = coefs.idxmax()
    top_neg = coefs.idxmin()
    st.markdown(f"""
    <div class="verdict">
        <b>{top_pos}</b> has the strongest positive influence on predicted score;
        <b>{top_neg}</b> has the strongest negative influence. Background and lifestyle
        factors contribute smaller, secondary effects.
    </div>
    """, unsafe_allow_html=True)

# ---- Predict a score --------------------------------------------------------
with tab_predict:
    st.markdown('<div class="section-label">06</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predict a student\'s score</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Set a student\'s profile below. The model predicts a final exam '
        'score and shows exactly which factors pushed it up or down.</div>',
        unsafe_allow_html=True
    )

    if "seed_row" not in st.session_state:
        st.session_state.seed_row = df[FEATURE_COLS].iloc[0]

    b1, b2, b3 = st.columns(3)
    if b1.button("Load a random student"):
        st.session_state.seed_row = df[FEATURE_COLS].sample(1, random_state=np.random.randint(0, 10000)).iloc[0]
    if b2.button("Load a top performer"):
        idx = df.sort_values("final_exam_score", ascending=False).index[:40]
        st.session_state.seed_row = df.loc[idx, FEATURE_COLS].sample(1).iloc[0]
    if b3.button("Load a struggling student"):
        idx = df.sort_values("final_exam_score", ascending=True).index[:40]
        st.session_state.seed_row = df.loc[idx, FEATURE_COLS].sample(1).iloc[0]

    seed = st.session_state.seed_row
    input_col, result_col = st.columns([1.1, 1], gap="large")

    with input_col:
        st.markdown("**Student profile**")
        study_hours = st.slider("Study hours / week", 0.0, 35.0, float(seed["study_hours_per_week"]), 0.5)
        attendance = st.slider("Attendance rate (%)", 40.0, 100.0, float(seed["attendance_rate"]), 0.5)
        prev_grade = st.slider("Previous grade", 30.0, 100.0, float(seed["previous_grade"]), 0.5)
        sleep = st.slider("Sleep hours", 3.0, 10.0, float(seed["sleep_hours"]), 0.1)
        extracurricular = st.slider("Extracurricular activities", 0, 6, int(seed["extracurricular_count"]))

        cc1, cc2 = st.columns(2)
        part_time = cc1.toggle("Part-time job", value=bool(seed["part_time_job"]))
        internet = cc2.toggle("Internet access", value=bool(seed["internet_access"]))
        study_group = st.toggle("Study group", value=bool(seed["study_group"]))
        parent_ed = st.selectbox(
            "Parental education", PARENT_ED_LEVELS,
            index=int(seed["parental_education_encoded"])
        )

    input_row = pd.Series({
        "study_hours_per_week": study_hours,
        "attendance_rate": attendance,
        "previous_grade": prev_grade,
        "sleep_hours": sleep,
        "extracurricular_count": extracurricular,
        "part_time_job": int(part_time),
        "internet_access": int(internet),
        "parental_education_encoded": PARENT_ED_ORDER[parent_ed],
        "study_group": int(study_group),
    })[FEATURE_COLS]

    input_scaled = bundle["scaler"].transform(pd.DataFrame([input_row]))
    predicted = float(bundle["model"].predict(input_scaled)[0])
    predicted_clamped = max(0.0, min(100.0, predicted))

    if predicted_clamped >= 75:
        band_class, band_label = "strong", "Strong outcome"
    elif predicted_clamped >= 55:
        band_class, band_label = "ontrack", "On track"
    else:
        band_class, band_label = "support", "May benefit from support"

    # Per-feature contribution: coef * standardized(value)
    z = (input_row - bundle["feature_means"]) / bundle["feature_stds"]
    contributions = (bundle["coefs"] * z).rename(index=FEATURE_LABELS)
    max_abs = max(contributions.abs().max(), 0.01)

    with result_col:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-eyebrow">Predicted final exam score</div>
            <div class="score-figure">{predicted_clamped:.1f}</div>
            <div class="score-band {band_class}">{band_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**What's driving this prediction**")
        for feat, val in contributions.sort_values(key=abs, ascending=False).items():
            pct = abs(val) / max_abs * 50
            cls = "pos" if val >= 0 else "neg"
            style = f"width:{pct:.1f}%;"
            st.markdown(f"""
            <div class="contrib-row">
                <div class="contrib-label">{feat}</div>
                <div class="contrib-track">
                    <div class="contrib-center"></div>
                    <div class="contrib-fill {cls}" style="{style}"></div>
                </div>
                <div class="contrib-val">{val:+.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "Bars show each factor's push on the predicted score, in points, relative to an "
            "average student. This is an educational estimate, not an official assessment."
        )