# CS3244 Project: Stock Market Forecasting

## Group Information
**Group Number:** 26  
**Project Title:** Hybrid LSTM-XGBoost Models for Stock Market Forecasting

### Team Members
- SIAH JIN THAU
- XIAO XINKAI
- XIA TANGZIHAN
- GUPTA KARTIK
- NGUYEN MINH GIAP

## Project Overview
This project investigates machine learning approaches for forecasting stock market movements using historical price and volume data. We aim to compare baseline and advanced models for financial time series prediction, with a focus on combining sequential learning and tabular feature modeling.

Our main proposed approach is a **hybrid LSTM-XGBoost framework**:
- **LSTM** is used to capture temporal patterns in stock price sequences.
- **XGBoost** is used to model nonlinear relationships in engineered features and latent representations extracted from the LSTM.

We will compare this hybrid approach against simpler baseline models to evaluate whether the combination improves predictive performance.

## Dataset
We use the **Huge Stock Market Dataset** provided for the CS3244 project.

### Dataset Description
The dataset contains historical daily stock prices and trading volumes for U.S. stocks and ETFs listed on:
- NYSE
- NASDAQ
- NYSE MKT

### Features
Typical columns include:
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `OpenInt`

## Project Goals
The goals of this project are:
1. Preprocess and explore financial time series data.
2. Engineer useful technical indicators and lag-based features (possibly).
3. Build and tune different machine learning models for stock prediction such as XGBoost, LSTM+XGBoost, etc.
4. Evaluate and compare model performance on stock price prediction

## Planned Models (In discussion)
### Baseline Models
- Naive / simple baseline
- Linear Regression

### Advanced Models
- XGBoost
- Hybrid LSTM-XGBoost

## Evaluation Metrics (In discussion)
We plan to evaluate model performance using:
- **MAPE** (Mean Absolute Percentage Error)


## Project Structure
```text
cs3244-stock-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│
├── results/
│   ├── plots/
│   └── metrics/
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train_xgboost.py
│   ├── train_lstm_xgboost.py
│   ├── hybrid_model.py
│   ├── evaluate.py (To Be Considered)
│
├── .gitignore
├── README.md
└── requirements.txt
