import streamlit as st
import pandas as pd
from forecast_engine import run_forecast
import pickle
from io import BytesIO

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting",
    layout="wide"
)

st.title("📈 Sales Forecasting")
st.caption("Scenario-based forward sales forecasting using trained ML models")

# ---------------------------------------------------
# Load metadata for filters (safe, lightweight)
# ---------------------------------------------------
@st.cache_resource
def load_filter_metadata():
    with open("artifacts/df_h_ext.pkl", "rb") as f:
        df_h = pickle.load(f)
    with open("artifacts/df_l_ext.pkl", "rb") as f:
        df_l = pickle.load(f)

    df_all = pd.concat([df_h, df_l])

    return (
        sorted(df_all["Model"].dropna().unique()),
        sorted(df_all["FuelType"].dropna().unique()),
        sorted(df_all["Color"].dropna().unique())
    )

models, fuels, colors = load_filter_metadata()

# ---------------------------------------------------
# Sidebar – Controls
# ---------------------------------------------------
st.sidebar.header("Forecast Controls")

start_month = st.sidebar.date_input(
    "Forecast start month",
    value=pd.to_datetime("2026-01-01")
)

n_months = st.sidebar.slider(
    "Forecast horizon (months)",
    min_value=1,
    max_value=18,
    value=6
)

correction_factor = st.sidebar.number_input(
    "Correction factor",
    min_value=0.70,
    max_value=1.30,
    value=0.96,
    step=0.01
)

st.sidebar.markdown("---")
st.sidebar.caption("Optional Filters")

selected_models = st.sidebar.multiselect(
    "Model",
    options=models
)

selected_fuels = st.sidebar.multiselect(
    "Fuel Type",
    options=fuels
)

selected_colors = st.sidebar.multiselect(
    "Color",
    options=colors
)

# ---------------------------------------------------
# Run Forecast
# ---------------------------------------------------
if st.button("🚀 Generate Forecast", type="primary"):

    with st.spinner("Generating forecast..."):
        result = run_forecast(
            start_month=start_month,
            n_months=n_months,
            correction_factor=correction_factor,
            selected_models=selected_models or None,
            selected_fuels=selected_fuels or None,
            selected_colors=selected_colors or None
        )

    if result.empty:
        st.warning("No data available for the selected inputs.")
        st.stop()

    # ---------------------------------------------------
    # Summary KPIs
    # ---------------------------------------------------
    st.subheader("📊 Forecast Summary")

    kpi1, kpi2 = st.columns(2)

    total_units = int(result["Predicted"].sum())
    avg_monthly = int(result.groupby("ForecastMonth")["Predicted"].sum().mean())

    monthly_totals = (
    result
    .groupby("ForecastMonth")["Predicted"]
    .sum()
    .reset_index()
    .rename(columns={"Predicted": "Total Units"})
    )

    st.subheader("🗓 Month-wise Forecast Totals")
    st.dataframe(
    monthly_totals,
    use_container_width=True
    )



    kpi1.metric("Total Forecast Units", f"{total_units:,}")
    kpi2.metric("Avg Monthly Volume", f"{avg_monthly:,}")

    # ---------------------------------------------------
    # Detailed Table
    # ---------------------------------------------------
    st.subheader("📋 Detailed Forecast")

    st.dataframe(
        result.sort_values(
            ["ForecastMonth", "Predicted"],
            ascending=[True, False]
        ),
        use_container_width=True
    )

    # ---------------------------------------------------
    # Monthly Trend Chart
    # ---------------------------------------------------
    st.subheader("📈 Monthly Trend")

    monthly = (
        result
        .groupby("ForecastMonth")["Predicted"]
        .sum()
        .reset_index()
    )

    st.bar_chart(
        monthly.set_index("ForecastMonth")
    )

    # ---------------------------------------------------
    # Excel Export
    # ---------------------------------------------------
    st.subheader("⬇ Export")

    buffer = BytesIO()
    result.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    start_dt = pd.to_datetime(start_month)
    end_dt = start_dt + pd.DateOffset(months=n_months - 1)

    file_name = (
        f"Sales_Forecast_"
        f"{start_dt.strftime('%b%Y')}_to_{end_dt.strftime('%b%Y')}.xlsx"
   )

    st.download_button(
    label="Download Forecast (Excel)",
    data=buffer,
    file_name=file_name,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
   )

