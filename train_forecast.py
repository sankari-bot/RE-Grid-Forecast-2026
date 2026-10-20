import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
import joblib
import os

print("=== Training Imbalance Forecast Model - Extended Version ===\n")


if not os.path.exists('FINAL_Imbalance_Dataset.csv'):
    print("ERROR: FINAL_Imbalance_Dataset.csv not found! Run merging.py first")
    exit()

df = pd.read_csv('FINAL_Imbalance_Dataset.csv')
df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
df = df.dropna(subset=['time'])
df = df.sort_values('time').reset_index(drop=True)

print(f"Original: {df.shape}")



df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['month'] = df['time'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['day_of_year'] = df['time'].dt.dayofyear

if 'temp' in df.columns:
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df['temp'] = df['temp'].ffill()
if 'wind_speed' in df.columns:
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
    df['wind_speed'] = df['wind_speed'].ffill()


df['load_lag_1h'] = df['total_load_MW'].shift(1)
df['load_lag_2h'] = df['total_load_MW'].shift(2)
df['load_lag_24h'] = df['total_load_MW'].shift(24)
df['load_lag_168h'] = df['total_load_MW'].shift(168)  

df['gen_lag_1h'] = df['total_gen_MW'].shift(1)
df['gen_lag_24h'] = df['total_gen_MW'].shift(24)

df['imbalance_lag_1h'] = df['Imbalance_MW'].shift(1)
df['imbalance_lag_24h'] = df['Imbalance_MW'].shift(24)


df['imbalance_rolling_24h'] = df['Imbalance_MW'].rolling(window=24, min_periods=1).mean()
df['imbalance_rolling_168h'] = df['Imbalance_MW'].rolling(window=168, min_periods=1).mean()
df['load_rolling_24h'] = df['total_load_MW'].rolling(window=24, min_periods=1).mean()


lag_cols = ['load_lag_1h','load_lag_2h','load_lag_24h','load_lag_168h','gen_lag_1h','gen_lag_24h','imbalance_lag_1h','imbalance_lag_24h']
for col in lag_cols:
    if col in df.columns:
        df[col] = df[col].ffill()


df = df.iloc[168:].reset_index(drop=True)
print(f"After removing first week for lag: {df.shape}")


base_features = [
    'hour','day_of_week','month','is_weekend','day_of_year',
    'total_load_MW','total_gen_MW',
    'load_lag_1h','load_lag_2h','load_lag_24h','load_lag_168h',
    'gen_lag_1h','gen_lag_24h',
    'imbalance_lag_1h','imbalance_lag_24h',
    'imbalance_rolling_24h','imbalance_rolling_168h','load_rolling_24h',
    'temp','humidity','wind_speed','pressure'
]

features = [f for f in base_features if f in df.columns]

clean_features = []
for f in features:
    if df[f].isna().sum() / len(df) < 0.5:
        clean_features.append(f)
    else:
        print(f"Skipping {f} - too many NaNs")

features = clean_features
print(f"Final Features ({len(features)}): {features}")

df = df.dropna(subset=features + ['Imbalance_MW'])
print(f"Final Training Shape: {df.shape}")

if len(df) < 100:
    print("Dataset too small!")
    exit()


X = df[features]
y = df['Imbalance_MW']

split = int(len(df)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")


print("\n--- Training Random Forest ---")
model = RandomForestRegressor(
    n_estimators=200,      
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)


pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
acc = 100 - (mae / (y.abs().mean()+1) * 100)

print(f"\n=== MODEL PERFORMANCE (FOR REPORT) ===")
print(f"MAE: {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"R2 Score: {r2:.4f}")
print(f"Accuracy: {acc:.2f}%")
print(f"Mean Actual: {y.mean():.2f} MW | Mean Predicted: {pred.mean():.2f} MW")


def get_status(v):
    if v > 1000: return 'High Excess'
    elif v > 0: return 'Excess'
    elif v > -1000: return 'Deficit'
    else: return 'High Deficit'

if 'Grid_Status' not in df.columns:
    df['Grid_Status'] = df['Imbalance_MW'].apply(get_status)

y_test_status = y_test.apply(get_status)
pred_status = pd.Series(pred).apply(get_status)

print(f"\n=== STATUS CLASSIFICATION REPORT ===")
print(classification_report(y_test_status, pred_status))


out = df.iloc[split:].copy()
out['Predicted_Imbalance'] = pred
out['Prediction_Error'] = out['Imbalance_MW'] - out['Predicted_Imbalance']
out['Predicted_Status'] = pred_status.values
out['Actual_Status'] = y_test_status.values


out[['time','total_gen_MW','total_load_MW','Imbalance_MW','Predicted_Imbalance','Prediction_Error','Grid_Status','Predicted_Status']].to_csv('Forecast_Results.csv', index=False)


imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
imp.to_csv('Feature_Importance.csv', index=False)
print(f"\nTop 5 Features:\n{imp.head()}")


joblib.dump(model, 'grid_imbalance_model.pkl')
print("\nSaved: grid_imbalance_model.pkl")


print(f"\n=== 24H FUTURE FORECAST ===")
last_row = X.iloc[-1:].copy()
last_time = df['time'].iloc[-1]
future_forecast = []

for i in range(1, 49):  
    next_pred = model.predict(last_row)[0]
    last_time = last_time + pd.Timedelta(hours=1)
    status = get_status(next_pred)
    action = "CHARGE BATTERY" if next_pred > 500 else "DISCHARGE/IMPORT" if next_pred < -500 else "BALANCED"
    future_forecast.append([last_time, round(next_pred, 2), status, action])
    
    
    last_row['imbalance_lag_1h'] = next_pred
    last_row['hour'] = last_time.hour
    last_row['day_of_week'] = last_time.dayofweek
    last_row['month'] = last_time.month

forecast_df = pd.DataFrame(future_forecast, columns=['Future_Time','Predicted_Imbalance_MW','Grid_Status','Recommended_Action'])
forecast_df.to_csv('24h_Forecast.csv', index=False)
print(forecast_df.head(24).to_string(index=False))


print(f"\n=== FILES CREATED - COPY THIS FOR REPORT ===")
print(f"1. Forecast_Results.csv - {len(out)} rows")
print(f"2. Feature_Importance.csv")
print(f"3. 24h_Forecast.csv - Next 48h prediction")
print(f"4. grid_imbalance_model.pkl - Trained model")
print(f"\n--- FOR FINAL REPORT TABLE ---")
print(f"Model: Random Forest | MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW | R2: {r2:.4f} | Accuracy: {acc:.2f}%")
print("\nSUCCESS! All expected outputs generated!")

