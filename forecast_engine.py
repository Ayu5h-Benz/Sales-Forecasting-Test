import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor


# ---------------------------------------------------
# Load all artifacts ONCE
# ---------------------------------------------------
def load_artifacts(artifact_path="artifacts"):

    model_h = CatBoostRegressor()
    model_h.load_model(f"{artifact_path}/model_h.cbm")

    model_l = CatBoostRegressor()
    model_l.load_model(f"{artifact_path}/model_l.cbm")

    with open(f"{artifact_path}/features.pkl", "rb") as f:
        features = pickle.load(f)

    with open(f"{artifact_path}/discontinued.pkl", "rb") as f:
        discontinued_models = pickle.load(f)

    with open(f"{artifact_path}/df_h_ext.pkl", "rb") as f:
        df_h_ext = pickle.load(f)

    with open(f"{artifact_path}/df_l_ext.pkl", "rb") as f:
        df_l_ext = pickle.load(f)

    return (
        model_h,
        model_l,
        features,
        discontinued_models,
        df_h_ext,
        df_l_ext
    )


# ---------------------------------------------------
# Core Forecast Function
# ---------------------------------------------------
def run_forecast(
    start_month,
    n_months,
    correction_factor=0.96,
    selected_models=None,
    selected_fuels=None,
    selected_colors=None,
    artifact_path="artifacts"
):
    """
    Returns a detailed forecast dataframe
    """

    (
        model_h,
        model_l,
        features,
        discontinued_models,
        df_h_ext,
        df_l_ext
    ) = load_artifacts(artifact_path)

    start_month = pd.to_datetime(start_month)
    forecast_months = pd.date_range(
        start_month,
        periods=n_months,
        freq="MS"
    )

    all_forecasts = []

    for month in forecast_months:

        # -------------------------------
        # Slice future rows
        # -------------------------------
        fh = df_h_ext[df_h_ext["YearMonth"] == month].copy()
        fl = df_l_ext[df_l_ext["YearMonth"] == month].copy()

        if fh.empty and fl.empty:
            continue

        # -------------------------------
        # Predict
        # -------------------------------
        if not fh.empty:
            fh["Predicted"] = np.floor(
                np.maximum(model_h.predict(fh[features]), 0)
                * correction_factor
            ).astype(int)

        if not fl.empty:
            fl["Predicted"] = np.floor(
                np.maximum(model_l.predict(fl[features]), 0)
                * correction_factor
            ).astype(int)

        forecast = pd.concat([fh, fl], ignore_index=True)

        # -------------------------------
        # Apply discontinued logic
        # -------------------------------
        forecast.loc[
            forecast["Model"].isin(discontinued_models),
            "Predicted"
        ] = 0

        # -------------------------------
        # Apply filters (optional)
        # -------------------------------
        if selected_models:
            forecast = forecast[forecast["Model"].isin(selected_models)]

        if selected_fuels:
            forecast = forecast[forecast["FuelType"].isin(selected_fuels)]

        if selected_colors:
            forecast = forecast[forecast["Color"].isin(selected_colors)]

        forecast["ForecastMonth"] = month.strftime("%b %Y")

        all_forecasts.append(
            forecast[
                [
                    "ForecastMonth",
                    "Model",
                    "FuelType",
                    "Color",
                    "Predicted"
                ]
            ]
        )

    if not all_forecasts:
        return pd.DataFrame()

    final_df = pd.concat(all_forecasts, ignore_index=True)

    return final_df
