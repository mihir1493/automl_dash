import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from holidays import country_holidays
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import plotly.graph_objects as go

# Helper Functions
def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics["MAPE"] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["sMAPE"] = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    metrics["Accuracy"] = (1 - metrics["MAPE"] / 100) * 100
    return metrics

# Streamlit App
st.title("Time Series Forecasting with Prophet")

# Sidebar for Selections
with st.sidebar:
    st.header("Configuration")

    # Step 1: File Upload
    uploaded_file = st.file_uploader("Upload Excel or CSV File", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Detect file type
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        # Subset Selection (Optional)
        subset_col = st.selectbox("Select Column to Subset (Optional)", ["None"] + list(df.columns))
        if subset_col != "None":
            unique_values = df[subset_col].unique()
            selected_value = st.selectbox("Select Category (Optional)", ["None"] + list(unique_values))
            if selected_value != "None":
                df = df[df[subset_col] == selected_value]

        # Display the filtered data
        st.write("Filtered Data Preview", df.head())

        # Step 2: Target and Date Column Selection
        date_col = st.selectbox("Select Date Column", df.columns)
        target_col = st.selectbox("Select Target Column", df.columns)
        st.write("Converting Target Column to Numeric...")
        try:
            df[target_col] = df[target_col].replace({r"[^\d\.\-]": ""}, regex=True  # Remove non-numeric characters
    )
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")  # Convert to numeric
            st.success("Target column successfully converted to numeric.")
        except Exception as e:
            st.error(f"Error converting target column to numeric: {e}")
            st.stop()

        # Handle Missing Values
        if df[target_col].isna().sum() > 0:
            st.warning(f"Found {df[target_col].isna().sum()} rows with missing target values after conversion. Dropping them.")
            df.dropna(subset=[target_col], inplace=True)

        df[date_col] = pd.to_datetime(df[date_col])
        df.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)

        # Step 3: Prior Scale Settings
        changepoint_prior = st.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05)
        seasonality_prior = st.slider("Seasonality Prior Scale", 1.0, 20.0, 10.0)
        holidays_prior = st.slider("Holidays Prior Scale", 1.0, 20.0, 10.0)

        # Step 4: Seasonality Options
        seasonality_options = st.multiselect("Select Seasonality", ["monthly", "weekly", "yearly"], default=["yearly"])

        # Step 5: Holiday Selection
        country = st.selectbox("Select Country for Holidays", ["US", "Canada"])
        years = st.multiselect("Select Years for Holidays", range(2015, 2031), default=[2023])
        holiday_data = []
        for date_, name in country_holidays(country, years=years).items():
            holiday_data.append({"ds": date_, "holiday": f"{country} Holidays", "lower_window": -2, "upper_window": 1})
        holidays_df = pd.DataFrame(holiday_data)

        # Step 6: Training and Validation Dates
        train_start = st.date_input("Training Start Date", value=df["ds"].min())
        train_end = st.date_input("Training End Date", value=df["ds"].max())
        forecast_periods = st.number_input("Forecast Periods (days)", min_value=1, max_value=365, value=30)

# Prophet Model
if uploaded_file is not None:
    model = Prophet(
        changepoint_prior_scale=changepoint_prior,
        seasonality_prior_scale=seasonality_prior,
        holidays_prior_scale=holidays_prior,
        holidays=holidays_df
    )
    if "monthly" in seasonality_options:
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    if "weekly" in seasonality_options:
        model.add_seasonality(name="weekly", period=7, fourier_order=3)
    if "yearly" in seasonality_options:
        model.add_seasonality(name="yearly", period=365.25, fourier_order=10)

    # Train Model
    train_df = df[(df["ds"] >= str(train_start)) & (df["ds"] <= str(train_end))]
    model.fit(train_df)

    # Future Dataframe
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)

    # Visualization
    st.subheader("Forecast Visualization")
    validation_start = st.date_input("Validation Start Date", value=df["ds"].max() - pd.Timedelta(days=forecast_periods))
    validation_end = st.date_input("Validation End Date", value=df["ds"].max())
    validation_df = df[(df["ds"] >= str(validation_start)) & (df["ds"] <= str(validation_end))]

    # Plot Forecast with Validation
    fig = plot_plotly(model, forecast)
    if not validation_df.empty:
        fig.add_trace(
            go.Scatter(
                x=validation_df["ds"],
                y=validation_df["y"],
                mode="lines",
                name="Validation Data",
                line=dict(color="red", dash="dot")
            )
        )
    st.plotly_chart(fig)

    # Component Decomposition
    st.subheader("Component Decomposition")
    component_plot = plot_components_plotly(model, forecast)
    st.plotly_chart(component_plot)

    # Evaluation Metrics
    if not validation_df.empty:
        y_true = validation_df["y"]
        y_pred = forecast.loc[forecast["ds"].isin(validation_df["ds"]), "yhat"]
        metrics = calculate_metrics(y_true, y_pred)
        st.subheader("Model Evaluation Metrics")
        for metric, value in metrics.items():
            st.write(f"{metric}: {value:.2f}")

    # Export Forecast
    export_forecast = st.checkbox("Export Forecast as CSV")
    if export_forecast:
        csv_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="forecast.csv")
