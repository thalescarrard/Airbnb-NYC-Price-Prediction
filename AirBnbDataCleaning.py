# This script cleans up the NY AirBnb csv file 

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from geopy.distance import geodesic
import numpy as np
from sklearn.cluster import KMeans



# Cleaning Steps 

# Load dataset
df = pd.read_csv("AB_NYC_2019.csv")  

# Basic info
print(df.shape)
print(df.columns)
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing 'reviews_per_month' with 0 (assumes no reviews = 0/month)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Convert 'last_review' to datetime format
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

# Drop duplicate rows (if any)
df = df.drop_duplicates()

# Filter out listings with non-positive or extreme prices
df = df[(df['price'] > 0) & (df['price'] < 1000)]

# Normalize categorical text fields
df['neighbourhood_group'] = df['neighbourhood_group'].str.strip().str.title()
df['room_type'] = df['room_type'].str.strip().str.title()

# Strip whitespace from 'name' and 'host_name' too
df['name'] = df['name'].astype(str).str.strip()
df['host_name'] = df['host_name'].astype(str).str.strip()

# Advanced feature engineering
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['days_since_last_review'] = (pd.Timestamp('2023-01-01') - df['last_review']).dt.days.fillna(9999)
df['name_length'] = df['name'].apply(lambda x: len(str(x).split()))
df['name_has_luxury'] = df['name'].str.lower().str.contains('luxury').astype(int)
df['log_host_listings'] = np.log1p(df['calculated_host_listings_count'])
df['is_multi_host'] = (df['calculated_host_listings_count'] > 1).astype(int)
host_reviews = df.groupby('host_name')['number_of_reviews'].mean()
df['host_avg_reviews'] = df['host_name'].map(host_reviews)
df['is_top_host'] = ((df['is_multi_host'] == 1) & (df['number_of_reviews'] > 50)).astype(int)

# Flag keywords in 'name'
keywords = ['luxury', 'modern', 'cozy', 'spacious']

for kw in keywords:
    df[f'name_has_{kw}'] = df['name'].str.lower().str.contains(kw).astype(int)

# Neighborhood average price and price difference
df['neighborhood_avg_price'] = df.groupby('neighbourhood')['price'].transform('mean')
df['price_vs_neighborhood_avg'] = df['price'] - df['neighborhood_avg_price']

# Distance to Manhattan center
manhattan_center = (40.7831, -73.9712)

def distance_to_manhattan(row):
    return geodesic((row['latitude'], row['longitude']), manhattan_center).miles

df['dist_to_manhattan'] = df.apply(distance_to_manhattan, axis=1)

# Binary flag for no reviews
df['no_reviews'] = (df['number_of_reviews'] == 0).astype(int)

# Cluster listings based on geolocation to find “hot zones”
# Decide on number of clusters, e.g., 8
coords = df[['latitude', 'longitude']].dropna()  # remove rows with missing coords

# Fit KMeans on coords
kmeans = KMeans(n_clusters=8, random_state=42)
df.loc[coords.index, 'geo_cluster'] = kmeans.fit_predict(coords)

# Make sure geo_cluster is integer type
df['geo_cluster'] = df['geo_cluster'].astype('Int64')

# One-hot encode geo_cluster to feed to ML models
df = pd.get_dummies(df, columns=['geo_cluster'], prefix='zone')

# Final sanity check
print(df.describe(include='all'))
print(df.isnull().sum())

# Save cleaned csv file
df.to_csv("cleaned_airbnb_nyc.csv", index=False)