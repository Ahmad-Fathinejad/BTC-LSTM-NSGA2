# BTC-LSTM-NSGA2: Probabilistic Bitcoin Price Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project leverages a sophisticated meta-model to perform probabilistic forecasting of Bitcoin (BTC) price trends. It utilizes a Long Short-Term Memory (LSTM) network whose hyperparameters are optimized by the NSGA-II multi-objective genetic algorithm.

The optimization process finds a Pareto front of optimal models by balancing three key objectives:
1.  **Minimizing Prediction Error** (Validation Loss)
2.  **Minimizing Model Complexity** (Number of Parameters)
3.  **Minimizing Training Time**

The final output is not a single price point but a **probabilistic range (quantile regression)**, providing a richer, more risk-aware forecast.

## Key Features ✨

* **Probabilistic Forecasting**: Employs a Quantile Regression LSTM to predict a range of future prices (low, mid, high) instead of a single value.
* **Multi-Objective Hyperparameter Optimization**: Uses the NSGA-II algorithm to automatically find the best trade-offs between model accuracy, complexity, and speed.
* **Advanced Feature Engineering**: Enriches the input data with features derived from ARIMA, GARCH, Awesome Oscillator (as an Elliott Wave proxy), and Fibonacci Proximity levels.
* **Ensemble Predictions**: Includes a complete pipeline to train the top models from the Pareto front and run them in parallel for a robust ensemble forecast.
* **Automated Pipeline**: A fully automated `run_pipeline.sh` script handles the entire workflow, from optimization to prediction and reporting.

## Project Development Lifecycle  metodología

This project was developed with close adherence to a standard, rigorous machine learning project lifecycle, implementing several stages in an advanced, automated fashion.

* **✅ Step 1: Problem Framing**
    The project objective was clearly defined: to perform probabilistic (range-based) forecasting for the next hour of Bitcoin's price. Success criteria were established as the multi-objective optimization problem detailed above.

* **✅ Step 2: Data Acquisition**
    A well-defined historical dataset (`btc_usd_1h`) containing hourly OHLCV data was used as the primary data source.

* **✅ Step 3: Data Exploration**
    An implicit exploration of the data was conducted by selecting appropriate models for feature engineering. The choice of ARIMA (for time-series analysis), GARCH (for volatility), and Awesome Oscillator (for momentum) reflects a deep understanding of the characteristics of financial data.

* **✅ Step 4: Data Preparation**
    A complete and robust data preparation pipeline was implemented. This includes data loading, advanced feature engineering with multiple technical and statistical models, and data scaling.

* **✅ Step 5: Shortlisting Promising Models**
    Instead of manually selecting different model types, this project uses a more advanced approach. The NSGA-II algorithm automatically generates a "shortlist" of the best-performing LSTM architectures, presented as the Pareto front.

* **✅ Step 6: System Fine-Tuning**
    This step is the core of the meta-model. The entire NSGA-II process is a sophisticated system for fine-tuning hyperparameters. Furthermore, the `predict_ensemble.py` script explicitly combines the best models to create a robust ensemble system.

* **✅ Step 7: Presenting the System**
    Results are delivered in clean, documented, and machine-readable output files (`.json` reports for results and timing, `.txt` report for predictions). The code is organized in a clean, modular structure.

* **✅ Step 8: Launch & Maintenance**
    The foundations for launching and maintaining the system have been established. This includes the fully automated `run_pipeline.sh` script, the ability to run on different environments (like HPCs), and a clear methodology for continuous retraining with new data.
