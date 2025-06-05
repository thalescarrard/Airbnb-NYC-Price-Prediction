import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("cleaned_airbnb_nyc.csv")

target = 'price'
feature_cols = [
    'minimum_nights','number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365','days_since_last_review', 
    'name_length', 'log_host_listings', 'host_avg_reviews', 'dist_to_manhattan',
    'neighborhood_avg_price', 'no_reviews',
    'room_type', 'neighbourhood_group', 'name_has_luxury', 'is_multi_host', 'is_top_host', 
    'name_has_modern', 'name_has_cozy', 'name_has_spacious', 'neighbourhood'
] + [col for col in df.columns if col.startswith('zone_')]

X = df[feature_cols]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_cols = ['neighbourhood_group', 'room_type', 'neighbourhood']
num_cols = [col for col in X.columns if col not in cat_cols]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# Pipeline with XGBRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=42, eval_metric='rmse'))
])

# Parameter grid for RandomizedSearchCV (note the 'model__' prefix)
param_dist = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='r2',
    verbose=1,
    n_jobs=-1,
    random_state=42,
    error_score='raise'
)

# Fit model with hyperparameter tuning
random_search.fit(X_train, y_train)

# Best model pipeline
best_pipeline = random_search.best_estimator_

# Predict and evaluate
y_pred = best_pipeline.predict(X_test)

# Compare predictions with actual values
results = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})
results['Error'] = results['Actual Price'] - results['Predicted Price']
results.to_csv("predicted_vs_actual_prices.csv", index=False)

print("\n")
print(results.head(10))  # Show first 10 rows
print("\n")

print(f"MAE: ${mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Feature Importance 
# Get the trained XGBoost model from the pipeline
xgb_model = best_pipeline.named_steps['model']

# Get feature names after preprocessing
# This only works if OneHotEncoder was used with get_feature_names_out
preprocessor = best_pipeline.named_steps['preprocessor']
ohe = preprocessor.named_transformers_['cat']
num_features = preprocessor.transformers_[0][2]
cat_features = ohe.get_feature_names_out(preprocessor.transformers_[1][2])
all_features = np.concatenate([num_features, cat_features])

# Match importances to feature names
importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print top N features
feature_importance_df.to_csv("feature_importances.csv", index=False)
print(feature_importance_df.head(10))
print("\n")

