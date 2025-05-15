# Multi-Building Energy Load Forecasting with Transfer Learning and Fine-Tuning

This project implements a pipeline for forecasting hourly energy load for a target building by leveraging historical data from other buildings and fine-tuning a general model with limited data from the target building. It uses LightGBM as the core modeling algorithm and Optuna for hyperparameter optimization.

## Project Goal

The primary goal is to accurately forecast the hourly energy consumption for a specific "target" building for a future period (e.g., 10 months), given only a short period (e.g., 2 months) of its own historical load data. The model learns general load patterns and weather/time dependencies from a set of "training" buildings and then adapts this knowledge to the target building.

## Features

*   **Data Ingestion & Preparation:** Loads building energy consumption data and hourly weather data, merges them, and handles timestamp alignment.
*   **Feature Engineering:**
    *   Utilizes a comprehensive set of weather and time-based features (e.g., temperature, humidity, cyclical time encodings, holidays).
    *   Generates load-specific features:
        *   Lagged load values (e.g., load 1 hour ago, 24 hours ago, 1 week ago).
        *   Rolling window statistics (e.g., mean and standard deviation of load over past X hours/days).
*   **General Model Training:**
    *   Trains a LightGBM model on historical data from multiple "training" buildings.
    *   Uses **Optuna with TimeSeriesSplit cross-validation** for robust hyperparameter optimization of this general model.
*   **Fine-Tuning for Target Building:**
    *   Adapts the trained general model to the specific "target" building using its limited known historical data (e.g., 2 months).
    *   Uses **Optuna with a dedicated validation split** from the target building's known data to optimize fine-tuning hyperparameters.
*   **Recursive Forecasting:**
    *   Performs multi-step ahead hourly forecasts for the target building for the desired future period (e.g., 10 months).
*   **Evaluation:**
    *   Calculates Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) to assess model performance.


## Files

*   `main_script.py`: The main script implementing the forecasting pipeline.
*   `df_el.csv`: CSV file containing hourly energy load data for multiple buildings.
*   `df_weather_hourly.csv`: CSV file containing hourly weather and time related data.

## Requirements

*   Python 3.x
*   pandas
*   numpy
*   scikit-learn
*   lightgbm
*   optuna


## Team

This project was developed by:

*   Alireza Shooshtari
*   Tolga Yalçin
*   Antonio Pepiciello


