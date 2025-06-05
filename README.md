# Airbnb NYC Price Prediction

This project aims to predict Airbnb rental prices in New York City using machine learning techniques. It covers data cleaning, exploratory data analysis (EDA), feature engineering, model training with hyperparameter tuning, evaluation, and interpretation.

---

## About 

The project is currently **a work in progress**.

---

## Dataset

- Raw data: Airbnb NYC Open Data (`AB_NYC_2019.csv`)
- Cleaned data: `cleaned_airbnb_nyc.csv` (generated after preprocessing and feature engineering)

---

## Project Structure

### 1. Data Cleaning and Feature Engineering (`data_cleaning.py`)

- Load raw data and check basic info.
- Handle missing values (e.g., fill `reviews_per_month` nulls with 0).
- Convert date fields and remove duplicates.
- Filter listings with invalid prices.
- Normalize categorical text fields.
- Engineer new features such as:
  - Days since last review
  - Keyword flags in listing names (e.g., luxury, cozy)
  - Log-transformed host listings count
  - Host average reviews
  - Distance to Manhattan center (using geopy)
  - Binary flags such as `no_reviews`
- Perform geographic clustering (KMeans) and one-hot encode clusters.
- Save cleaned dataset as `cleaned_airbnb_nyc.csv`.

### 2. Exploratory Data Analysis (`eda.py`)

- Visualize price distribution (with cap at 99th percentile).
- Bar charts for listing counts per borough and average price by room type.
- Scatter plot of price vs. number of reviews colored by room type.
- Interactive map of listings using Plotly.
- Correlation heatmap for numerical features.

### 3. Model Training and Evaluation (`model_training.py`)

- Load cleaned data and select features and target (`price`).
- Split data into train and test sets.
- Build preprocessing pipeline for numeric scaling and categorical encoding.
- Train XGBoost regression model within pipeline.
- Tune hyperparameters with `RandomizedSearchCV`.
- Evaluate model with MAE, RMSE, and R².
- Export predicted vs actual prices and feature importances to CSV.
- Display evaluation metrics and top important features.

---

## Results Summary

Example model performance on test data:

- **MAE:** $46.35  
- **RMSE:** $81.69  
- **R²:** 0.48  

Top important features include room type, neighborhood average price, and geographic clusters.

---

## Future Enhancements

- Incorporate external datasets such as demographics, transit accessibility, and crime statistics.  
- Experiment with other machine learning models (Random Forest, LightGBM, neural networks).  
- Develop interactive dashboards and visualizations.  
- Automate data update and model retraining pipelines.  
- Add PowerBI visualizations to present insights interactively.

  ## Dependencies:

```bash
pip install pandas numpy scikit-learn xgboost geopy matplotlib seaborn plotly

