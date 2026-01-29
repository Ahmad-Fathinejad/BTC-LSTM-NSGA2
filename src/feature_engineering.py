# -*- coding: utf-8 -*-
import pandas as pd
import statsmodels.api as sm
import arch
import logging # اضافه شد

def add_arima_residuals(dataframe: pd.DataFrame) -> pd.DataFrame:
    # # یک مدل ARIMA(5,1,0) را روی قیمت close فیت کرده و باقی‌مانده‌های آن را اضافه می‌کند.
    # # Fits an ARIMA(5,1,0) model on the close price and adds its residuals as a feature.
    logging.info("Calculating ARIMA Residuals feature...") # جایگزین print
    model_arima = sm.tsa.arima.ARIMA(dataframe['close'], order=(5, 1, 0))
    results_arima = model_arima.fit()
    dataframe['arima_residuals'] = results_arima.resid.fillna(0)
    return dataframe

def add_garch_volatility(dataframe: pd.DataFrame) -> pd.DataFrame:
    # # یک مدل GARCH(1,1) را روی بازده قیمت فیت کرده و نوسان شرطی را اضافه می‌کند.
    # # Fits a GARCH(1,1) model on price returns and adds the conditional volatility.
    logging.info("Calculating GARCH Volatility feature...") # جایگزین print
    returns = dataframe['close'].pct_change().dropna() * 100
    model_garch = arch.arch_model(returns, vol='Garch', p=1, q=1)
    results_garch = model_garch.fit(disp='off')
    dataframe['garch_volatility'] = results_garch.conditional_volatility.fillna(method='bfill')
    return dataframe