# This script analyzes the NY AirBnb csv file 

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# EDA Steps

# Load dataset
df = pd.read_csv("AB_NYC_2019.csv")  

# Set seaborn theme
sns.set(style="whitegrid")

# Price Distribution
# Optional cap to 99th percentile
price_cap = df['price'].quantile(0.99)
plt.figure(figsize=(10, 6))
sns.histplot(df[df['price'] <= price_cap]['price'], bins=50, kde=True)
plt.title('Price Distribution (Capped at 99th Percentile)')
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.figtext(0.5, -0.05, 'Note: Listings above $1000 excluded for visibility.', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# Listing by Borough
borough_counts = df['neighbourhood_group'].value_counts().reset_index()
borough_counts.columns = ['neighbourhood', 'count']

fig = px.bar(
    borough_counts,
    x='neighbourhood', y='count',
    hover_data={'neighbourhood': True, 'count': True},
    title='Listings per Neighbourhood in NYC'
)
fig.update_layout(xaxis_title='Neighbourhood', yaxis_title='Number of Listings')
fig.show()

# Avergae Price by Room Type
avg_price_room = df.groupby('room_type')['price'].mean().reset_index()

fig = px.bar(
    avg_price_room,
    x='room_type', y='price',
    hover_data={'room_type': True, 'price': True},
    title='Average Price by Room Type'
)
fig.update_layout(yaxis_title='Average Price ($)', xaxis_title='Room Type')
fig.show()

# Price vs Number of Reviews
sample_df = df[df['price'] <= price_cap].sample(1000)

fig = px.scatter(
    sample_df,
    x='number_of_reviews', y='price',
    color='room_type',
    hover_data={'name': True, 'host_name': True, 'price': True, 'number_of_reviews': True},
    title='Price vs. Number of Reviews'
)
fig.update_layout(xaxis_title='Number of Reviews', yaxis_title='Price ($)')
fig.show()

# Interactive Map with Plotly
fig = px.scatter_mapbox(
    df.sample(1000),
    lat='latitude', lon='longitude',
    color='neighbourhood_group',
    size='price',
    hover_name='name',
    hover_data={'host_name': True, 'price': True, 'room_type': True, 'reviews_per_month': True},
    zoom=10,
    mapbox_style='carto-positron',
    title='NYC Airbnb Listings Map'
)
fig.show()

# Heatmap with Correlations
numeric_df = df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]
corr = numeric_df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

