import io

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

MODEL_PATH = "lead_conversion_model"
DATA_PATH = "customer_conversion_traing_dataset.csv"


def apply_style():
    st.set_page_config(page_title="Lead Conversion Predictor", page_icon="🛒", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f0c1d 0%, #17132b 100%);
            color: #f2f2f7;
        }
        h1, h2, h3 { font-family: 'Trebuchet MS', sans-serif; }
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
            margin-bottom: 1rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6C63FF, #00D9C0);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            transition: transform 0.15s ease;
        }
        .stButton>button:hover { transform: scale(1.03); color: white; }
        section[data-testid="stSidebar"] { background: #14101f; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle):
    st.markdown(
        f"""
        <div style="text-align:center; padding: 1rem 0 2rem 0;">
            <h1 style="font-size:2.4rem; margin-bottom:0.2rem;">{title}</h1>
            <p style="font-size:1.05rem; color:#b8b3d1;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(html):
    st.markdown(f"<div class='glass-card'>{html}</div>", unsafe_allow_html=True)


FEATURE_IMPORTANCE = [
    ("📄", "Pages Viewed", 48.6),
    ("🌍", "Location", 34.4),
    ("🎂", "Age", 9.1),
    ("⏱️", "Time Spent", 4.6),
]


def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 0.4rem 0 1rem 0;">
            <div style="font-size:2rem;">🛒</div>
            <h2 style="margin:0.2rem 0 0 0; font-size:1.35rem;">LeadScope AI</h2>
            <p style="color:#8a84ad; font-size:0.8rem; margin-top:2px;">
                Conversion Intelligence Suite
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "🔮 Predict", "📊 Explore Dataset"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    st.sidebar.markdown(
        "<p style='font-weight:700; font-size:0.95rem; margin-bottom:0.6rem;'>⚙️ Model Metadata</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<p style='font-size:0.8rem; color:#8a84ad; margin:0 0 0.6rem 0;'>Project Baseline Context:</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div style="background:rgba(69,120,180,0.18); border:1px solid rgba(108,99,255,0.35);
                    border-radius:12px; padding:0.85rem 1rem; margin-bottom:1.1rem;">
            <p style="margin:0 0 0.7rem 0; font-size:0.82rem; line-height:1.4;">
                🎯 <b style="color:#8ec9ff;">Target Domain:</b>
                <span style="color:#c7d6f0;">Lead Conversion Analysis</span>
            </p>
            <p style="margin:0; font-size:0.82rem; line-height:1.4;">
                🔍 <b style="color:#8ec9ff;">Evaluation Focus:</b>
                <span style="color:#c7d6f0;">Imbalance Optimization via F1-Score Metric Tracking</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "<p style='font-weight:700; font-size:0.95rem; margin-bottom:0.6rem;'>📈 Top Feature Power Splits</p>",
        unsafe_allow_html=True,
    )
    for icon, name, pct in FEATURE_IMPORTANCE:
        st.sidebar.markdown(
            f"""
            <div style="margin-bottom:0.65rem;">
                <p style="margin:0 0 4px 0; font-size:0.82rem;">{icon} {name} ({pct}%)</p>
                <div style="background:#241f38; border-radius:6px; height:8px; width:100%;">
                    <div style="background:linear-gradient(90deg,#FF6B6B,#FF9472);
                                width:{pct}%; height:8px; border-radius:6px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.sidebar.caption("Built with Streamlit · Decision Tree Classifier")

    return page


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


def page_home():
    hero(
        "🛒 Customer Lead Conversion Predictor",
        "A machine learning app that predicts whether a website visitor "
        "will convert into a paying customer for Stuffmart.com.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Decision Tree", "Best of 3 models")
    c2.metric("Accuracy", "1.00")
    c3.metric("F1 Score", "1.00")
    c4.metric("Task", "Binary Classification")

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        card(
            """
            <h3>📌 About this project</h3>
            <p>This project predicts <b>Lead Conversion</b> — whether a customer
            who visits Stuffmart.com will eventually make a purchase
            (<b>Hot Lead</b>) or not (<b>Cold Lead</b>).</p>
            <p>The pipeline follows the standard ML workflow:</p>
            <ul>
                <li>Data cleaning — duplicates, missing values, outliers (IQR + Z-score)</li>
                <li>Exploratory Data Analysis (EDA) — distributions, correlations</li>
                <li>Feature selection — noisy / low-impact columns removed</li>
                <li>Preprocessing — imputation + ordinal encoding via <code>ColumnTransformer</code></li>
                <li>Model selection — <code>RandomizedSearchCV</code> over Decision Tree,
                    Random Forest and KNN</li>
                <li>Evaluation — accuracy, precision, recall, F1, confusion matrix</li>
            </ul>
            """
        )

    with right:
        card(
            """
            <h3>🧠 Winning Model</h3>
            <p><b>Decision Tree Classifier</b> (class_weight="balanced")</p>
            <ul>
                <li>criterion: <code>gini</code></li>
                <li>max_depth: <code>15</code></li>
                <li>min_samples_split: <code>5</code></li>
            </ul>
            <p>Selected out of Decision Tree, Random Forest and
            K-Nearest Neighbors using 3-fold cross validated
            <code>RandomizedSearchCV</code> optimizing F1-score.</p>
            """
        )

    card(
        """
        <h3>🧬 Features Used by the Model</h3>
        <p><b>Numeric:</b> Age, TimeSpent (minutes), PagesViewed, EmailSent, FollowUpEmails</p>
        <p><b>Categorical:</b> Location, LeadStatus</p>
        <p>Several columns (LeadID, Gender, LeadSource, DeviceType, ReferralSource,
        FormSubmissions, Downloads, CTR_ProductPage, ResponseTime, SocialMediaEngagement,
        PaymentHistory) were dropped after they showed little to no impact on the model's
        feature importance.</p>
        """
    )

    st.info(
        "👈 Use the sidebar to open **Predict** and try the model on your own "
        "input, or **Explore Dataset** to inspect the raw data (df.head(), "
        "df.describe(), df.info(), correlations, and more)."
    )


def page_predict():
    hero("🔮 Predict Lead Conversion", "Fill in the visitor details below and click Predict.")

    model = None
    try:
        model = load_model(MODEL_PATH)
        st.success(f"Model loaded from `{MODEL_PATH}`", icon="✅")
    except FileNotFoundError:
        st.warning(
            f"Couldn't find `{MODEL_PATH}` next to the app. "
            "Upload the trained model file below to continue.",
            icon="⚠️",
        )
        uploaded_model = st.file_uploader("Upload model file (joblib)", type=None, key="model_upload")
        if uploaded_model is not None:
            with open(MODEL_PATH, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.cache_resource.clear()
            model = load_model(MODEL_PATH)
            st.success("Model loaded successfully!", icon="✅")

    st.write("")

    with st.form("prediction_form"):
        st.markdown("#### Visitor Details")
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", min_value=15, max_value=90, value=32, step=1)
            location = st.selectbox(
                "Location",
                ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad",
                 "Multan", "Peshawar", "Quetta", "Other"],
            )
            if location == "Other":
                location = st.text_input("Enter city name", value="Lahore")
            lead_status = st.selectbox("Lead Status", ["Hot", "Warm", "Cold"])

        with c2:
            time_spent = st.number_input(
                "Time Spent on Site (minutes)", min_value=0.0, max_value=300.0,
                value=60.00, step=0.5,
            )
            pages_viewed = st.number_input("Pages Viewed", min_value=0, max_value=100, value=15, step=1)
            email_sent = st.number_input("Emails Sent", min_value=0, max_value=50, value=4, step=1)
            follow_up_emails = st.number_input("Follow-up Emails", min_value=0, max_value=50, value=4, step=1)

        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        if model is None:
            st.error("Please load a model first (see the upload box above).")
        else:
            user_data = {
                "Age": age,
                "Location": location,
                "TimeSpent (minutes)": time_spent,
                "PagesViewed": pages_viewed,
                "LeadStatus": lead_status,
                "EmailSent": email_sent,
                "FollowUpEmails": follow_up_emails,
            }
            input_df = pd.DataFrame([user_data])

            try:
                prediction = model.predict(input_df)[0]

                if prediction == 1:
                    card(
                        "<h2 style='color:#00D9C0;'>🔥 CONVERT — Hot Lead</h2>"
                        "<p>This visitor is likely to become a paying customer.</p>"
                    )
                    st.balloons()
                else:
                    card(
                        "<h2 style='color:#FF6B6B;'>❄️ NOT CONVERT — Cold Lead</h2>"
                        "<p>This visitor is unlikely to convert based on current behaviour.</p>"
                    )

                with st.expander("See the exact input sent to the model"):
                    st.dataframe(input_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


def page_explore():
    hero("📊 Explore the Dataset", "Everything the model was trained and evaluated on.")

    df = None
    try:
        df = load_csv(DATA_PATH)
        st.success(f"Loaded `{DATA_PATH}`", icon="✅")
    except FileNotFoundError:
        st.warning(
            f"Couldn't find `{DATA_PATH}` next to the app. Upload a CSV to explore it.",
            icon="⚠️",
        )
        uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"], key="data_upload")
        if uploaded is not None:
            df = load_csv(uploaded)

    if df is None:
        st.stop()

    st.write("")

    option = st.selectbox(
        "Choose what you'd like to see:",
        [
            "Shape & Preview (head / tail)",
            "Column Info (df.info())",
            "Summary Statistics (df.describe())",
            "Missing Values",
            "Duplicate Rows",
            "Categorical Value Counts",
            "Correlation Heatmap",
            "Target Distribution",
        ],
    )

    st.write("")

    if option == "Shape & Preview (head / tail)":
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        n = st.slider("Rows to show", 5, 50, 5)
        t1, t2 = st.tabs(["df.head()", "df.tail()"])
        with t1:
            st.dataframe(df.head(n), use_container_width=True)
        with t2:
            st.dataframe(df.tail(n), use_container_width=True)

    elif option == "Column Info (df.info())":
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.code(buffer.getvalue(), language="text")

    elif option == "Summary Statistics (df.describe())":
        t1, t2 = st.tabs(["Numeric columns", "Categorical / text columns"])
        with t1:
            st.dataframe(df.describe(), use_container_width=True)
        with t2:
            obj_df = df.select_dtypes(include=["object", "string"])
            if obj_df.empty:
                st.info("No categorical / text columns found.")
            else:
                st.dataframe(obj_df.describe(), use_container_width=True)

    elif option == "Missing Values":
        miss = pd.DataFrame(
            {
                "Missing Count": df.isnull().sum(),
                "Missing %": (df.isnull().mean() * 100).round(2),
            }
        ).sort_values("Missing Count", ascending=False)
        st.dataframe(miss, use_container_width=True)

    elif option == "Duplicate Rows":
        st.metric("Total Duplicate Rows", int(df.duplicated().sum()))

    elif option == "Categorical Value Counts":
        cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            col = st.selectbox("Choose a categorical column", cat_cols)
            counts = df[col].value_counts()
            c1, c2 = st.columns([1, 1.4])
            with c1:
                st.dataframe(counts, use_container_width=True)
            with c2:
                fig, ax = plt.subplots()
                sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis",
                            hue=counts.index, legend=False)
                ax.set_xlabel("Count")
                st.pyplot(fig)

    elif option == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            st.info("Not enough numeric columns for a correlation heatmap.")
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="viridis", ax=ax)
            st.pyplot(fig)

    elif option == "Target Distribution":
        target_candidates = [c for c in df.columns if "Conversion" in c or "Target" in c]
        target_col = st.selectbox(
            "Target column", target_candidates if target_candidates else df.columns.tolist()
        )
        counts = df[target_col].value_counts()
        c1, c2 = st.columns(2)
        c1.metric("Class 0 (No Convert)", int(counts.get(0, 0)))
        c2.metric("Class 1 (Convert)", int(counts.get(1, 0)))
        fig, ax = plt.subplots()
        sns.countplot(x=df[target_col], ax=ax, palette="viridis", hue=df[target_col], legend=False)
        st.pyplot(fig)


def main():
    apply_style()

    page = render_sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "🔮 Predict":
        page_predict()
    else:
        page_explore()


if __name__ == "__main__":
    main()
