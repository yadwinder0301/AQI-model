# For Data Handling
import pandas as pd
import numpy as np

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Algorithm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#Evaluation Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Datetime
from datetime import datetime

# Importing Dataset from Drive

# from google.colab import drive
# drive.mount('/content/drive')
df = pd.read_csv('city_day.csv')

# Only important columns

columns_to_keep = ['City', 'Date','PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
df_clean = df[columns_to_keep].copy()

# Dropping ALL rows with ANY missing values
df_clean = df_clean.dropna()

# Reset index after dropping
df_clean = df_clean.reset_index(drop=True)

df_clean.shape

# Converting Date to datetime and extracting features
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

df_clean['Year'] = df_clean['Date'].dt.year
df_clean['Month'] = df_clean['Date'].dt.month
df_clean['Day'] = df_clean['Date'].dt.day


# Drop original Date column (we extracted what we need)
df_clean = df_clean.drop('Date', axis=1)

df_clean.head()
# One-Hot Encoding for City column
df_clean = pd.get_dummies(df_clean, columns=['City'], drop_first=True)

df_clean.shape

# Distribution of all features

df_clean.hist(figsize=(14, 10), bins=30, edgecolor='black')
plt.suptitle('Distribution of Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.close() # plt.show()

# Correlation of numerical features with AQI

numerical_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'Year', 'Month', 'Day', 'AQI']
df_clean[numerical_cols].corr()['AQI'].sort_values(ascending=False)

# Correlation heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(df_clean[numerical_cols].corr(), annot=True, fmt='.2f', cmap='RdYlGn_r', square=True)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.close() # plt.show()

# AQI distribution with category colors

plt.figure(figsize=(12, 5))
plt.hist(df_clean['AQI'], bins=50, color='steelblue', edgecolor='black')
plt.axvline(50, color='green', linestyle='--', label='Good (0-50)')
plt.axvline(100, color='yellow', linestyle='--', label='Moderate (50-100)')
plt.axvline(200, color='orange', linestyle='--', label='Unhealthy (100-200)')
plt.axvline(300, color='red', linestyle='--', label='Very Unhealthy (200-300)')
plt.xlabel('AQI Value', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Distribution of AQI Values', fontsize=14, fontweight='bold')
plt.legend()
plt.close() # plt.show()

# Boxplots for all pollutants

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']

for i, col in enumerate(pollutants):
    row = i // 3
    col_idx = i % 3
    axes[row, col_idx].boxplot(df_clean[col])
    axes[row, col_idx].set_title(f'{col} Distribution', fontweight='bold')
    axes[row, col_idx].set_ylabel('Value')

plt.suptitle('Pollutant Distributions (Boxplots)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.close() # plt.show()

# Top 10 cities by average AQI (using original df before encoding)
city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.barh(city_aqi.index, city_aqi.values, color='coral', edgecolor='black')
plt.xlabel('Average AQI', fontweight='bold')
plt.ylabel('City', fontweight='bold')
plt.title('Top 10 Most Polluted Cities (by Average AQI)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.close() # plt.show()

# Monthly AQI trend
monthly_aqi = df_clean.groupby('Month')['AQI'].mean()

plt.figure(figsize=(10, 6))
plt.plot(monthly_aqi.index, monthly_aqi.values, marker='o', linewidth=2, markersize=8, color='purple')
plt.fill_between(monthly_aqi.index, monthly_aqi.values, alpha=0.3, color='purple')
plt.xlabel('Month', fontweight='bold')
plt.ylabel('Average AQI', fontweight='bold')
plt.title('Monthly AQI Trend', fontsize=14, fontweight='bold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(alpha=0.3)
plt.close() # plt.show()

# Yearly AQI trend
yearly_aqi = df_clean.groupby('Year')['AQI'].mean()

plt.figure(figsize=(10, 6))
plt.bar(yearly_aqi.index, yearly_aqi.values, color='teal', edgecolor='black')
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Average AQI', fontweight='bold')
plt.title('Yearly AQI Trend', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.close() # plt.show()

# Scatter plots: Pollutants vs AQI
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for i, col in enumerate(pollutants):
    row = i // 3
    col_idx = i % 3
    axes[row, col_idx].scatter(df_clean[col], df_clean['AQI'], alpha=0.5, color=colors[i], edgecolor='black')
    axes[row, col_idx].set_xlabel(col, fontweight='bold')
    axes[row, col_idx].set_ylabel('AQI', fontweight='bold')
    axes[row, col_idx].set_title(f'{col} vs AQI', fontweight='bold')

plt.suptitle('Pollutants vs AQI Relationship', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.close() # plt.show()

# Splitting features (X) and target (y)
X = df_clean.drop('AQI', axis=1)
y = df_clean['AQI']

X.shape, y.shape

# Splitting into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape

# Scaling features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape

# Linear Regression

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

print("Linear Regression R² Score:", r2_score(y_test, lr_pred))

# Decision Tree Regressor

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

print("Decision Tree R² Score:", r2_score(y_test, dt_pred))

# Random Forest Regressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print("Random Forest R² Score:", r2_score(y_test, rf_pred))

# Support Vector Regressor

svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
svr_pred = svr_model.predict(X_test_scaled)

print("SVR R² Score:", r2_score(y_test, svr_pred))

# K-Nearest Neighbors Regressor

knn_model = KNeighborsRegressor()
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

print("KNN R² Score:", r2_score(y_test, knn_pred))

# Actual vs Predicted comparison for all models
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

predictions = [lr_pred, dt_pred, rf_pred, svr_pred, knn_pred]
models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'KNN']
colors = ['red', 'blue', 'green', 'orange', 'purple']

for i, (pred, model) in enumerate(zip(predictions, models)):
    row = i // 3
    col_idx = i % 3
    axes[row, col_idx].scatter(y_test, pred, alpha=0.5, color=colors[i])
    axes[row, col_idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[row, col_idx].set_xlabel('Actual AQI', fontweight='bold')
    axes[row, col_idx].set_ylabel('Predicted AQI', fontweight='bold')
    axes[row, col_idx].set_title(f'{model}', fontweight='bold')

# Remove empty subplot
fig.delaxes(axes[1, 2])

plt.suptitle('Actual vs Predicted AQI - All Models', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.close() # plt.show()

# Feature Importance (Random Forest)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

feature_importance.head(10)

# Evaluation metrics for all models
results = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'KNN'],
    'R² Score': [
        r2_score(y_test, lr_pred),
        r2_score(y_test, dt_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test, svr_pred),
        r2_score(y_test, knn_pred)
    ],
    'MAE': [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, dt_pred),
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, svr_pred),
        mean_absolute_error(y_test, knn_pred)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, dt_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred)),
        np.sqrt(mean_squared_error(y_test, svr_pred)),
        np.sqrt(mean_squared_error(y_test, knn_pred))
    ]
}

eval_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
eval_df

# Metrics comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
models = ['LR', 'DT', 'RF', 'SVR', 'KNN']

# R² Score
r2_scores = [r2_score(y_test, lr_pred), r2_score(y_test, dt_pred),
             r2_score(y_test, rf_pred), r2_score(y_test, svr_pred),
             r2_score(y_test, knn_pred)]
axes[0].bar(models, r2_scores, color='green', edgecolor='black')
axes[0].set_title('R² Score (Higher is Better)', fontweight='bold')
axes[0].set_ylim(0, 1)

# MAE
mae_scores = [mean_absolute_error(y_test, lr_pred), mean_absolute_error(y_test, dt_pred),
              mean_absolute_error(y_test, rf_pred), mean_absolute_error(y_test, svr_pred),
              mean_absolute_error(y_test, knn_pred)]
axes[1].bar(models, mae_scores, color='orange', edgecolor='black')
axes[1].set_title('MAE (Lower is Better)', fontweight='bold')

# RMSE
rmse_scores = [np.sqrt(mean_squared_error(y_test, lr_pred)), np.sqrt(mean_squared_error(y_test, dt_pred)),
               np.sqrt(mean_squared_error(y_test, rf_pred)), np.sqrt(mean_squared_error(y_test, svr_pred)),
               np.sqrt(mean_squared_error(y_test, knn_pred))]
axes[2].bar(models, rmse_scores, color='red', edgecolor='black')
axes[2].set_title('RMSE (Lower is Better)', fontweight='bold')

plt.suptitle('Model Evaluation Metrics Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.close() # plt.show()

# Sample Predictions

sample_results = pd.DataFrame({
    'Actual AQI': y_test.values[:10],
    'Predicted AQI': rf_pred[:10],
    'Difference': abs(y_test.values[:10] - rf_pred[:10])
})

sample_results

# Check for outliers in AQI

plt.figure(figsize=(10, 5))
plt.boxplot(df_clean['AQI'], vert=False)
plt.xlabel('AQI Value')
plt.title('AQI Outliers Detection', fontweight='bold')
plt.close() # plt.show()
# Remove extreme outliers (AQI > 500 is rare/extreme)

df_clean_no_outliers = df_clean[df_clean['AQI'] <= 500].copy()

print(f"Before: {df_clean.shape[0]} rows")
print(f"After:  {df_clean_no_outliers.shape[0]} rows")
print(f"Removed: {df_clean.shape[0] - df_clean_no_outliers.shape[0]} outliers")

# Split features and target again
X = df_clean.drop('AQI', axis=1)
y = df_clean['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape

# Scale features again
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape

# Save the best model (Random Forest)

import pickle
import gzip

with gzip.open('aqi_model.pkl.gz', 'wb') as f:
    pickle.dump(rf_model, f)
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Create prediction function

def predict_aqi(pm25, pm10, no2, co, so2, o3, city_encoded, year, month, day):
    """
    Predict AQI based on input features
    """
    # Create input array (must match training feature order)
    input_data = [[pm25, pm10, no2, co, so2, o3] + city_encoded + [year, month, day]]

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = rf_model.predict(input_scaled)

    return round(prediction[0], 2)

# Example: Make a prediction
# Sample input values

sample_input = X_test.iloc[0].values.reshape(1, -1)
sample_scaled = scaler.transform(sample_input)
predicted_aqi = rf_model.predict(sample_scaled)[0]
actual_aqi = y_test.iloc[0]

print("="*60)
print("SAMPLE PREDICTION")
print("="*60)
print(f"Input Features: {X_test.iloc[0].to_dict()}")
print(f"\nPredicted AQI: {predicted_aqi:.2f}")
print(f"Actual AQI:    {actual_aqi:.2f}")
print(f"Difference:    {abs(predicted_aqi - actual_aqi):.2f}")
print("="*60)

# Interactive prediction (you can change these values)
print("🌍 CUSTOM AQI PREDICTION")
print("="*60)

# Change these values to test different scenarios
test_pm25 = 85.0
test_pm10 = 120.0
test_no2 = 45.0
test_co = 1.2
test_so2 = 15.0
test_o3 = 55.0

# Create input matching your feature set
# Note: You'll need to adjust city encoding based on your data
test_input = X_test.iloc[0].values.copy()  # Copy structure
test_input[0] = test_pm25
test_input[1] = test_pm10
test_input[2] = test_no2
test_input[3] = test_co
test_input[4] = test_so2
test_input[5] = test_o3

# Predict
test_scaled = scaler.transform([test_input])
predicted = rf_model.predict(test_scaled)[0]

print(f"PM2.5: {test_pm25}")
print(f"PM10:  {test_pm10}")
print(f"NO2:   {test_no2}")
print(f"CO:    {test_co}")
print(f"SO2:   {test_so2}")
print(f"O3:    {test_o3}")
print(f"\n🎯 Predicted AQI: {predicted:.2f}")

# AQI Category
if predicted <= 50:
    category = "Good 😊"
elif predicted <= 100:
    category = "Moderate 😐"
elif predicted <= 200:
    category = "Unhealthy ☹️"
elif predicted <= 300:
    category = "Very Unhealthy 😷"
else:
    category = "Hazardous ☠️"

print(f"Category: {category}")
print("="*60)

# Download the model files (optional - for future use)
# from google.colab import files

# files.download('aqi_model.pkl')
# files.download('scaler.pkl')

print("✅ Model files ready for download!")