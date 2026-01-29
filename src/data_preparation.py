# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.feature_engineering import add_arima_residuals, add_garch_volatility
from src import config
import logging # اضافه شد

def load_and_prepare_data():
    # # داده‌ها را از فایل CSV بارگذاری و آماده می‌کند.
    # # Loads and prepares data from the CSV file.
    logging.info(f"Loading data from {config.DATA_FILE_PATH}...") # جایگزین print
    df = pd.read_csv(config.DATA_FILE_PATH, index_col=config.DATE_COLUMN, parse_dates=True)

    # --- اصلاحیه: فیلتر کردن ۶ ماه پایانی داده‌ها ---
    # --- Modification: Filtering for the last 6 months of data ---
    logging.info("Filtering data for the last 6 months...") # جایگزین print
    max_date = df.index.max()
    six_months_ago = max_date - pd.DateOffset(months=6)
    df = df[df.index >= six_months_ago]
    logging.info(f"Data filtered from {six_months_ago.date()} to {max_date.date()}.") # جایگزین print
    # --- پایان اصلاحیه ---

    # # افزودن ویژگی‌های مدل-محور
    # # Adding model-based features
    feature_df = add_arima_residuals(df.copy())
    feature_df = add_garch_volatility(feature_df)
    feature_df.dropna(inplace=True)
    logging.info("Feature engineering complete.") # جایگزین print

    # # نرمال‌سازی داده‌ها
    # # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_df)
    logging.info("Data scaling complete.") # جایگزین print

    return scaled_data, scaler, feature_df