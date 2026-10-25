
import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "crop_production_starter_dataset.csv"
)
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AgriPredict | Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 3. THEME: SAGE GREEN, CONCRETE GREY, BURNT AMBER
# ==================================================

SAGE = "#A3B18A"
SAGE_LIGHT = "#E7EBDD"
SAGE_DARK = "#485640"
GREY = "#E3E3DF"
GREY_DARK = "#6F736D"
AMBER = "#C56A32"
AMBER_DARK = "#A94F20"
PAPER = "#F7F7F3"
INK = "#303A32"

st.markdown(f"""
<style>
.stApp {{
    background-color: {PAPER};
    color: {INK};
}}

.block-container {{
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}

h1, h2, h3 {{
    color: {SAGE_DARK} !important;
    font-weight: 750 !important;
}}

section[data-testid="stSidebar"] {{
    background-color: {SAGE_LIGHT};
    border-right: 1px solid {GREY};
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {SAGE_DARK} !important;
}}

div[data-testid="stMetric"] {{
    background: white;
    border: 1px solid {GREY};
    border-left: 4px solid {SAGE};
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(48,58,50,0.05);
}}

div[data-testid="stMetricLabel"] {{
    color: {GREY_DARK};
}}

div[data-testid="stMetricValue"] {{
    color: {SAGE_DARK};
    font-weight: 750;
}}

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {{
    background-color: {AMBER};
    color: white;
    border: 1px solid {AMBER};
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 700;
}}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {{
    background-color: {AMBER_DARK};
    color: white;
    border-color: {AMBER_DARK};
}}

.hero {{
    background:
        linear-gradient(110deg,
        rgba(38,49,39,0.94),
        rgba(72,86,64,0.78)),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85");
    background-size: cover;
    background-position: center;
    border-radius: 24px;
    padding: 60px 45px;
    min-height: 340px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-bottom: 24px;
    border: 1px solid {SAGE};
    box-shadow: 0 12px 28px rgba(48,58,50,0.12);
}}

.hero .eyebrow {{
    color: #E9B38F;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}}

.hero h1 {{
    color: white !important;
    font-size: clamp(36px, 5vw, 58px) !important;
    line-height: 1.12;
    margin: 14px 0;
}}

.hero p {{
    color: #F4F3EE !important;
    font-size: 17px;
    line-height: 1.7;
    max-width: 750px;
}}

.feature {{
    background: white;
    border: 1px solid {GREY};
    border-top: 4px solid {SAGE};
    border-radius: 15px;
    padding: 22px;
    min-height: 185px;
    box-shadow: 0 4px 14px rgba(48,58,50,0.04);
}}

.feature h3 {{
    font-size: 20px;
    margin-top: 8px;
}}

.feature p {{
    color: {GREY_DARK};
    line-height: 1.6;
}}

.kicker {{
    color: {AMBER_DARK};
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}}

div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"] {{
    background: white;
    border: 1px solid {GREY};
    border-radius: 12px;
    padding: 8px;
}}

hr {{
    border-color: {GREY};
}}

@media(max-width: 768px) {{
    .hero {{
        padding: 30px 22px;
        min-height: 280px;
    }}
}}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 4. LOAD DATA AND MODEL
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()
except Exception as error:
    st.error(f"Could not load the dataset or model: {error}")
    st.info(
        "Check that the dataset is inside the data folder "
        "and crop_model.joblib is inside the models folder."
    )
    st.stop()

df.columns = df.columns.str.strip()

required_columns = [
    "State", "District", "Crop_Year",
    "Season", "Crop", "Area", "Production"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Dataset is missing these columns: {missing}")
    st.stop()

for column in ["Crop_Year", "Area", "Production"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required_columns).copy()

if df.empty:
    st.error("The dataset contains no valid records.")
    st.stop()

# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    st.markdown("## 🌿 AgriPredict")
    st.caption("Smart Agriculture Platform")
    st.divider()

    if "navigation" not in st.session_state:
        st.session_state["navigation"] = "Welcome"

    page = st.radio(
        "NAVIGATION",
        [
            "Welcome",
            "Home",
            "Predict Production",
            "Data Analysis",
            "Model Performance",
            "About"
        ],
        key="navigation"
    )

    st.divider()
    st.caption("Model: Random Forest Regressor")
    st.caption("Academic project demonstration")

# ==================================================
# 6. COMMON HEADER AND DATA NOTICE
# ==================================================

if page != "Welcome":
    st.title("🌾 AgriPredict")
    st.markdown("### Agriculture Crop Production Prediction")
    st.caption("Machine learning • Crop analytics • Interactive insights")

st.warning(
    "DEMO DATA NOTICE: This version uses synthetic sample data. "
    "Predictions, charts and metrics are illustrative and do not "
    "represent verified real-world agricultural results."
)

# ==================================================
# 7. WELCOME / LANDING PAGE
# ==================================================

if page == "Welcome":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Smart Agriculture • Data-Driven Insights</div>
        <h1>Predict Smarter.<br>Grow Better.</h1>
        <p>
            Welcome to AgriPredict — explore crop production data,
            generate machine-learning estimates and discover patterns
            through interactive agricultural analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="kicker">DISCOVER AGRIPREDICT</div>',
        unsafe_allow_html=True
    )
    st.header("Agricultural insights, all in one place")
    st.write(
        "Explore crop records, experiment with production predictions "
        "and understand the model through clear visualizations."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature">
            <div style="font-size:32px">🌱</div>
            <h3>Crop Prediction</h3>
            <p>Enter crop, location, season, year and area details
            to generate an illustrative production estimate.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature">
            <div style="font-size:32px">📊</div>
            <h3>Interactive Analytics</h3>
            <p>Explore crop-wise, season-wise and year-wise patterns
            using interactive charts and filters.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature">
            <div style="font-size:32px">🤖</div>
            <h3>Machine Learning</h3>
            <p>Learn how a Random Forest regression model is used
            to estimate crop production from input features.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("How it works")

        st.markdown("""
        **01 — Explore the dataset**  
        Review the available crop production records.

        **02 — Enter crop details**  
        Select the location, crop, season, year and cultivated area.

        **03 — Generate insights**  
        View a model estimate, interactive charts and evaluation metrics.
        """)

    with right:
        st.subheader("Project snapshot")

        a, b = st.columns(2)
        a.metric("Dataset records", f"{len(df):,}")
        b.metric("Crop types", f"{df['Crop'].nunique():,}")

        a, b = st.columns(2)
        a.metric("States", f"{df['State'].nunique():,}")
        b.metric("ML model", "Random Forest")

    st.info(
        "Academic demonstration: the current dataset is synthetic. "
        "Use verified historical data before drawing real-world "
        "agricultural conclusions."
    )

    st.subheader("Ready to explore?")

    b1, b2 = st.columns(2)

    with b1:
        st.button(
            "🌾 Open Dashboard",
            use_container_width=True,
            on_click=lambda: st.session_state.update(
                {"navigation": "Home"}
            )
        )

    with b2:
        st.button(
            "🔮 Predict Production",
            use_container_width=True,
            on_click=lambda: st.session_state.update(
                {"navigation": "Predict Production"}
            )
        )

    st.caption("AgriPredict | Agriculture analytics project")

# ==================================================
# 8. HOME DASHBOARD
# ==================================================

elif page == "Home":

    st.subheader("Project Overview")
    st.write(
        "A quick overview of the crop records available in the dataset."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{len(df):,}")
    c2.metric("Crop Types", f"{df['Crop'].nunique():,}")
    c3.metric("States", f"{df['State'].nunique():,}")
    c4.metric("Total Production", f"{df['Production'].sum():,.0f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("🌾 Production by Crop")

        crop_summary = (
            df.groupby("Crop", as_index=False)["Production"]
            .sum()
            .sort_values("Production", ascending=False)
            .head(8)
            .sort_values("Production")
        )

        fig = px.bar(
            crop_summary,
            x="Production",
            y="Crop",
            orientation="h",
            color="Production",
            color_continuous_scale=[
                [0, "#DDE3D3"],
                [0.55, SAGE],
                [1, AMBER]
            ],
            template="plotly_white",
            title="Top Crops in the Dataset"
        )
        fig.update_layout(
            height=400,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("📈 Production by Year")

        yearly = (
            df.groupby("Crop_Year", as_index=False)["Production"]
            .sum()
            .sort_values("Crop_Year")
        )

        fig = px.line(
            yearly,
            x="Crop_Year",
            y="Production",
            markers=True,
            template="plotly_white",
            title="Year-wise Production Trend"
        )
        fig.update_traces(
            line=dict(color=AMBER, width=3),
            marker=dict(size=8)
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Complete Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="crop_production_dataset.csv",
        mime="text/csv"
    )

# ==================================================
# 9. CROP PRODUCTION PREDICTION
# ==================================================

elif page == "Predict Production":

    st.header("🔮 Predict Crop Production")
    st.write("Enter the crop and cultivation details below.")

    states = sorted(df["State"].astype(str).unique())
    crops = sorted(df["Crop"].astype(str).unique())
    seasons = sorted(df["Season"].astype(str).unique())

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            state = st.selectbox("Select State", states)

            districts = sorted(
                df.loc[
                    df["State"].astype(str) == state, "District"
                ].astype(str).unique()
            )
            district = st.selectbox("Select District", districts)
            crop = st.selectbox("Select Crop", crops)

        with col2:
            min_year = int(df["Crop_Year"].min())
            max_year = int(df["Crop_Year"].max())

            year = st.number_input(
                "Crop Year",
                min_value=1900,
                max_value=2100,
                value=max(min_year, min(max_year, 2024)),
                step=1
            )

            season = st.selectbox("Select Season", seasons)
            area = st.number_input(
                "Cultivated Area (dataset units)",
                min_value=0.1,
                value=100.0,
                step=10.0
            )

        submitted = st.form_submit_button(
            "🌱 Predict Production",
            use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame([{
            "State": state,
            "District": district,
            "Crop_Year": int(year),
            "Season": season,
            "Crop": crop,
            "Area": float(area)
        }])

        try:
            prediction = float(model.predict(input_data)[0])

            st.divider()
            st.subheader("Prediction Result")
            st.metric(
                "Predicted Production",
                f"{prediction:,.2f} dataset units"
            )

            st.caption(
                f"Crop: {crop} | State: {state} | "
                f"District: {district} | Year: {int(year)}"
            )

            result = input_data.copy()
            result["Predicted_Production"] = prediction

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Prediction Result",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="crop_prediction_result.csv",
                mime="text/csv"
            )

            st.info(
                "This estimate is based on synthetic training data. "
                "It is for demonstration only, not actual farming "
                "or financial decisions."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# ==================================================
# 10. DATA ANALYSIS
# ==================================================

elif page == "Data Analysis":

    st.header("📊 Crop Production Analytics")
    st.write("Filter the dataset to explore different patterns.")

    st.info(
        "The current records are synthetic. Patterns shown here "
        "do not represent verified agricultural trends."
    )

    all_states = sorted(df["State"].astype(str).unique())
    all_crops = sorted(df["Crop"].astype(str).unique())

    c1, c2 = st.columns(2)

    with c1:
        selected_states = st.multiselect(
            "Select States",
            all_states,
            default=all_states
        )

    with c2:
        selected_crops = st.multiselect(
            "Select Crops",
            all_crops,
            default=all_crops
        )

    min_year = int(df["Crop_Year"].min())
    max_year = int(df["Crop_Year"].max())

    if min_year < max_year:
        year_range = st.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = (min_year, max_year)
        st.caption(f"Available year: {min_year}")

    filtered = df[
        df["State"].astype(str).isin(selected_states)
        & df["Crop"].astype(str).isin(selected_crops)
        & df["Crop_Year"].between(year_range[0], year_range[1])
    ].copy()

    if filtered.empty:
        st.warning("No records match these filters. Change your selections.")

    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Production", f"{filtered['Production'].sum():,.0f}")
        c2.metric("Cultivated Area", f"{filtered['Area'].sum():,.0f}")
        c3.metric("Crop Types", filtered["Crop"].nunique())
        c4.metric("Records", len(filtered))

        st.divider()
        st.subheader("📈 Year-wise Production Trend")

        yearly = (
            filtered.groupby("Crop_Year", as_index=False)["Production"]
            .sum()
            .sort_values("Crop_Year")
        )

        fig_year = px.line(
            yearly,
            x="Crop_Year",
            y="Production",
            markers=True,
            template="plotly_white",
            title="Annual Crop Production",
            labels={
                "Crop_Year": "Crop Year",
                "Production": "Production (dataset units)"
            }
        )
        fig_year.update_traces(
            line=dict(color=AMBER, width=3),
            marker=dict(size=9)
        )
        fig_year.update_layout(height=420)
        st.plotly_chart(fig_year, use_container_width=True)

        st.subheader("🌾 Crop-wise Production Comparison")

        crop_totals = (
            filtered.groupby("Crop", as_index=False)["Production"]
            .sum()
            .sort_values("Production")
        )

        fig_crop = px.bar(
            crop_totals,
            x="Production",
            y="Crop",
            orientation="h",
            color="Production",
            color_continuous_scale=[
                [0, "#DDE3D3"],
                [0.55, SAGE],
                [1, AMBER]
            ],
            template="plotly_white",
            title="Production Ranking by Crop"
        )
        fig_crop.update_layout(
            height=max(400, len(crop_totals) * 35),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_crop, use_container_width=True)

        st.subheader("🌦️ Season-wise Production")

        seasonal = (
            filtered.groupby("Season", as_index=False)["Production"]
            .sum()
            .sort_values("Production", ascending=False)
        )

        fig_season = px.bar(
            seasonal,
            x="Season",
            y="Production",
            color="Season",
            template="plotly_white",
            title="Production Across Growing Seasons",
            color_discrete_sequence=[
                SAGE, AMBER, GREY_DARK, SAGE_DARK, "#D8B08C"
            ]
        )
        fig_season.update_layout(
            height=420,
            showlegend=False
        )
        st.plotly_chart(fig_season, use_container_width=True)

        st.subheader("🌱 Cultivated Area vs Production")

        fig_area = px.scatter(
            filtered,
            x="Area",
            y="Production",
            color="Crop",
            size="Production",
            size_max=35,
            hover_data=["State", "District", "Crop_Year", "Season"],
            template="plotly_white",
            title="Area and Production Relationship"
        )
        fig_area.update_layout(height=480)
        st.plotly_chart(fig_area, use_container_width=True)

        st.divider()
        st.subheader("📋 Filtered Dataset")

        st.dataframe(
            filtered.sort_values("Crop_Year"),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Filtered Data",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_crop_production.csv",
            mime="text/csv"
        )

# ==================================================
# 11. MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.header("🎯 Model Performance")
    st.write("Evaluation metrics for the Random Forest regression model.")

    if os.path.exists(METRICS_PATH):

        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        st.info(
            "These metrics come from the synthetic starter dataset. "
            "They should not be interpreted as real-world accuracy."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("MAE", f"{metrics.get('MAE', 0):,.2f}")
        c2.metric("RMSE", f"{metrics.get('RMSE', 0):,.2f}")
        c3.metric("R² Score", f"{metrics.get('R2', 0):.3f}")

        st.subheader("Understanding the Metrics")

        st.markdown("""
        - **MAE:** Average absolute difference between actual and
          predicted production values.
        - **RMSE:** Measures prediction errors and penalizes larger
          errors more heavily.
        - **R² Score:** Indicates how well the model explains
          variation in the test data. It can be negative.
        """)

        st.subheader("Training Details")
        st.json(metrics)

    else:
        st.warning(
            "Metrics file not found. Run `py train_model.py` "
            "to create it."
        )

# ==================================================
# 12. ABOUT THE PROJECT
# ==================================================

elif page == "About":

    st.header("ℹ️ About AgriPredict")

    st.markdown("""
    ### Agriculture Crop Production Prediction

    This project demonstrates how machine learning and data
    visualization can be used to explore crop production records.

    **Project objectives**
    - Prepare and analyze crop production data.
    - Train a Random Forest regression model.
    - Estimate production from selected crop details.
    - Visualize crop-wise, year-wise and season-wise patterns.

    **Technology stack**
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Plotly
    - Joblib

    **Machine learning workflow**
    1. Load and clean the dataset.
    2. Encode categorical features.
    3. Split records into training and testing sets.
    4. Train a Random Forest Regressor.
    5. Evaluate and save the model.
    6. Load the trained model in the website.
    """)

    st.warning(
        "The current starter dataset is synthetic. Replace it with "
        "verified historical agricultural data before making claims "
        "about real-world production or model accuracy."
    )

    st.caption("AgriPredict | Academic Project Demonstration")
