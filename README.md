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
2. Engineer useful technical indicators and lag-based features.
3. Build and compare different machine learning models for stock prediction.
4. Analyze model performance under different market conditions such as high- and low-volatility periods.
5. Evaluate whether a hybrid neural-ensemble model outperforms standalone methods.

## Planned Models
### Baseline Models
- Naive / simple baseline
- Linear Regression
- XGBoost

### Advanced Models
- LSTM
- Hybrid LSTM-XGBoost

## Evaluation Metrics
We plan to evaluate model performance using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **Directional Accuracy** (optional, for up/down movement prediction)

## Project Structure
```text
cs3244-stock-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── hybrid_model.py
│   ├── evaluate.py
│
├── results/
│   ├── plots/
│   └── metrics/
│
├── models/
├── requirements.txt
├── .gitignore
└── README.md
