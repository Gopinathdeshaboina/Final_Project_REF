# Life Expectancy Prediction from Socioeconomic Indicators (2000–2022)

This project investigates how socioeconomic indicators relate to global life expectancy and evaluates machine learning models for predicting life expectancy using World Bank data.

## Overview
Using World Bank country-year data from **2000–2022**, the project:
- Performs exploratory data analysis (EDA) to understand relationships between predictors and life expectancy.
- Trains and compares three regression models:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor
- Tunes Random Forest and XGBoost hyperparameters using cross-validated search.
- Interprets model behaviour using permutation feature importance.

## Dataset
Source: **World Bank Open Data** (CSV exports)

Each observation represents a **country-year** record. The target and predictors used:

- **Target**
  - `Life_Expectancy` — Life expectancy at birth (years)

- **Predictors**
  - `GDP_per_Capita` — GDP per capita (current US$)
  - `PPP_GDP_per_Capita` — GDP per capita, PPP (current international $) *(World Bank indicator: NY.GDP.PCAP.PP.CD)*
  - `School_Enrollment` — School enrolment (%)
  - `Health_Expenditure` — Health expenditure (current US$ / per capita depending on indicator used)
  - `Inflation` — Inflation rate (%)
  - `Unemployment` — Unemployment rate (%)

> Note: The modelling pipeline uses log-transformed versions of GDP, PPP, and Health Expenditure to reduce skewness.

## Methods

### Preprocessing & Feature Engineering
- Merged multiple World Bank indicator files into a single country-year dataset.
- Created log features:
  - `log_GDP_per_Capita = log10(GDP_per_Capita)`
  - `log_PPP = log10(PPP_GDP_per_Capita)`
  - `log_Health_Expenditure = log10(Health_Expenditure)`
- Missing values were handled **inside scikit-learn pipelines** using **median imputation** (to avoid leakage and preserve data).

### Models
- **Linear Regression** as an interpretable baseline.
- **Random Forest Regressor** to capture nonlinearities and interactions.
- **XGBoost Regressor** (gradient-boosted trees) for strong tabular performance.

### Evaluation Metrics
Models are evaluated on a held-out test set using:
- **R²** (explained variance)
- **MAE** (mean absolute error, years)
- **RMSE** (root mean squared error, years)

### Hyperparameter Tuning
Random Forest and XGBoost are tuned using cross-validated randomized search.

### Feature Importance
Permutation feature importance is computed on the test set for tuned models to identify influential predictors.

## Results Summary (Typical Findings)
- Tree-based ensembles (RF, XGB) substantially outperform Linear Regression.
- Most influential predictors are consistently:
  - log(GDP per capita)
  - log(PPP GDP per capita)
  - log(Health expenditure)
  - School enrollment  
- Inflation and unemployment contribute relatively less predictive power in this dataset.

## Project Structure
