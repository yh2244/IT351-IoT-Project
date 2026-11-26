import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Smart Home Energy Dashboard", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp {background:#F6F8FB;color:#172B4D}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E7ECF2}
.main-title {color:#173B57;font-size:2rem;font-weight:750}
.subtitle {color:#66788A;margin-bottom:1.2rem}
div[data-testid="stMetric"] {background:#FFFFFF;border:1px solid #E7ECF2;padding:16px;border-radius:12px}
div[data-testid="stMetricLabel"] {color:#66788A}
div[data-testid="stMetricValue"] {color:#173B57;font-size:1.45rem;white-space:nowrap}
</style>
""", unsafe_allow_html=True)

DATA_FILE = Path(__file__).parent / "energy_data.csv"
REQUIRED = ["timestamp", "household_id", "energy_consumption_kWh", "future_consumption_kWh"]

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for col in REQUIRED[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=REQUIRED).sort_values("timestamp").reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid records remain after cleaning.")
    return data

@st.cache_resource
def train_model(data):
    X = data[["energy_consumption_kWh", "household_id"]]
    y = data["future_consumption_kWh"]
    if len(data) < 5:
        raise ValueError("At least 5 valid records are needed.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, r2_score(y_test, predictions), mean_squared_error(y_test, predictions), y_test, predictions

def kwh(value):
    return f"{value:.4f} kWh"

def make_time_chart(data):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(data["timestamp"], data["energy_consumption_kWh"],
            color="#268A91", linewidth=1.7, label="Energy consumption")
    ax.set_title("Smart Home Energy Consumption Over Time", color="#173B57")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Energy Consumption (kWh)")
    ax.grid(True, linestyle="--", alpha=.22)
    ax.tick_params(axis="x", rotation=25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

st.sidebar.markdown("## ⚡ Energy Insights")
st.sidebar.caption("IoT Smart Home Monitoring")
st.sidebar.markdown("---")
page = st.sidebar.radio("NAVIGATION", [
    "Dashboard Overview", "Energy Analysis", "Prediction Engine", "Model Performance"
])
st.sidebar.markdown("---")
st.sidebar.caption("IT351 – Data Science for the Internet of Things")

if not DATA_FILE.exists():
    st.error("Dataset not found. Place energy_data.csv in the same folder as app.py.")
    st.code("SmartHomeEnergyDashboard/\\n├── app.py\\n├── energy_data.csv\\n└── requirements.txt")
    st.stop()

try:
    df = load_data(DATA_FILE)
    model, model_r2, model_mse, y_test, y_pred = train_model(df)
except Exception as exc:
    st.error(f"Could not prepare the dashboard: {exc}")
    st.stop()

st.markdown('<div class="main-title">Smart Home Energy Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Monitor electricity consumption, explore historical trends, and predict future energy usage.</div>', unsafe_allow_html=True)

households = sorted(df["household_id"].unique().tolist())
selected = st.sidebar.selectbox("Select household", ["All households"] + households)
view = df if selected == "All households" else df[df["household_id"] == selected]
if view.empty:
    st.warning("No records for this household.")
    st.stop()

if page == "Dashboard Overview":
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(view):,}")
    c2.metric("Average Consumption", kwh(view["energy_consumption_kWh"].mean()))
    c3.metric("Highest Consumption", kwh(view["energy_consumption_kWh"].max()))
    c4.metric("Average Future Consumption", kwh(view["future_consumption_kWh"].mean()))
    st.subheader("Historical Consumption")
    time_fig = make_time_chart(view)
    st.pyplot(time_fig, width="stretch")
    plt.close(time_fig)
    left, right = st.columns(2)
    with left:
        st.markdown("**Average actual vs. future consumption**")
        st.bar_chart(view[["energy_consumption_kWh", "future_consumption_kWh"]].mean(), width="stretch")
    with right:
        st.markdown("**Dataset information**")
        st.write(f"Date range: {view.timestamp.min():%Y-%m-%d} to {view.timestamp.max():%Y-%m-%d}")
        st.write(f"Households: {view.household_id.nunique()}")
        st.write(f"Missing values after cleaning: {int(view[REQUIRED].isna().sum().sum())}")
        st.write("Sampling interval stated in report: 5 minutes")
    with st.expander("Preview dataset"):
        st.dataframe(view.head(20), width="stretch", hide_index=True)

elif page == "Energy Analysis":
    st.subheader("Historical Energy Analysis")
    min_date, max_date = view.timestamp.min().date(), view.timestamp.max().date()
    date_range = st.date_input("Choose date range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    analysis = view
    if isinstance(date_range, tuple) and len(date_range) == 2:
        analysis = view[view.timestamp.dt.date.between(date_range[0], date_range[1])]
    if analysis.empty:
        st.info("No records in this date range.")
    else:
        a, b, c = st.columns(3)
        a.metric("Average Usage", kwh(analysis.energy_consumption_kWh.mean()))
        b.metric("Minimum Usage", kwh(analysis.energy_consumption_kWh.min()))
        c.metric("Maximum Usage", kwh(analysis.energy_consumption_kWh.max()))
        time_fig = make_time_chart(analysis)
        st.pyplot(time_fig, width="stretch")
        plt.close(time_fig)
        st.markdown("**Consumption distribution**")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.hist(analysis.energy_consumption_kWh, bins=25, color="#67AEB0", edgecolor="white")
        ax.set_xlabel("Energy Consumption (kWh)")
        ax.set_ylabel("Number of readings")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

elif page == "Prediction Engine":
    st.subheader("Future Consumption Prediction")
    st.info("Enter a current reading and household ID. The trained Linear Regression model estimates future consumption.")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        current = col1.number_input("Current energy consumption (kWh)", min_value=0.0,
                                    value=float(df.energy_consumption_kWh.median()),
                                    step=0.01, format="%.4f")
        household = col2.selectbox("Household ID", households)
        submitted = st.form_submit_button("Predict Future Consumption", type="primary",
                                          width="stretch")
    if submitted:
        sample = pd.DataFrame({"energy_consumption_kWh": [current], "household_id": [household]})
        prediction = float(model.predict(sample)[0])
        a, b = st.columns(2)
        a.metric("Predicted Future Consumption", kwh(prediction))
        b.metric("Current Consumption", kwh(current))
        if prediction >= current:
            st.info("The model predicts higher future consumption than the current reading.")
        else:
            st.success("The model predicts lower future consumption than the current reading.")
        st.caption("This is a model-based estimate from entered values, not a live sensor reading.")
    st.markdown("**Model:** Linear Regression | **Features:** current energy consumption and household ID | **Target:** future energy consumption")

elif page == "Model Performance":
    st.subheader("Machine Learning Model Performance")
    a, b = st.columns(2)
    a.metric("R² Score", f"{model_r2:.4f}")
    b.metric("Mean Squared Error", f"{model_mse:.6f}")
    st.info("Metrics are recalculated from the loaded dataset using the report's 80/20 split and random_state=42.")
    st.markdown("**Actual vs. predicted values (test set)**")
    results = pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": np.asarray(y_pred)}).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(results.index, results.Actual, label="Actual", color="#268A91", linewidth=1.7)
    ax.plot(results.index, results.Predicted, label="Predicted", color="#E39B55", linewidth=1.7)
    ax.set_xlabel("Test record")
    ax.set_ylabel("Future Consumption (kWh)")
    ax.grid(True, alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown("**Model coefficients**")
    st.dataframe(pd.DataFrame({
        "Feature": ["energy_consumption_kWh", "household_id"],
        "Coefficient": model.coef_
    }), width="stretch", hide_index=True)
    st.write(f"Intercept: {model.intercept_:.6f}")
    with st.expander("View test-set predictions"):
        st.dataframe(results, width="stretch", hide_index=True)

st.markdown("---")
st.caption("IT351 | IoT Smart Home Energy Consumption Prediction Dashboard")
