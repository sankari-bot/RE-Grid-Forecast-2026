import pandas as pd
import os
import numpy as np

print("=== Grid Imbalance Project - Data Merging Started ===\n")


files = ['energy_dataset.csv', 'weather_features.csv', 'Renewable.csv', 'continuous dataset.csv', 'weekly pre-dispatch forecast.csv']
for f in files:
    print(f"{'Found' if os.path.exists(f) else 'Not Found'}: {f}")

energy = pd.read_csv('energy_dataset.csv')
weather = pd.read_csv('weather_features.csv')

print(f"\nEnergy Shape: {energy.shape}")
print(f"Weather Shape: {weather.shape}")


energy['time'] = pd.to_datetime(energy['time'], utc=True, errors='coerce')
weather['time'] = pd.to_datetime(weather['dt_iso'], utc=True, errors='coerce')

energy = energy.dropna(subset=['time'])
weather = weather.dropna(subset=['time'])


if 'city_name' in weather.columns:
    print(f"Cities: {weather['city_name'].unique()[:5]}")
    agg_dict = {}
    for col in ['temp', 'humidity', 'wind_speed', 'pressure', 'clouds_all', 'rain_1h', 'temp_min', 'temp_max']:
        if col in weather.columns:
            agg_dict[col] = 'mean'
    if agg_dict:
        weather = weather.groupby('time').agg(agg_dict).reset_index()
        print(f"Weather aggregated: {weather.shape}")


energy['time'] = energy['time'].dt.floor('h')
weather['time'] = weather['time'].dt.floor('h')
energy = energy.sort_values('time')
weather = weather.sort_values('time')


df = pd.merge(energy, weather, on='time', how='inner')
print(f"\nAfter Energy+Weather Merge: {df.shape}")

if df.empty:
    print("Trying merge_asof with 1h tolerance...")
    df = pd.merge_asof(energy, weather, on='time', direction='nearest', tolerance=pd.Timedelta('1h'))


if os.path.exists('Renewable.csv'):
    try:
        renew = pd.read_csv('Renewable.csv')
        
        r_time = 'time' if 'time' in renew.columns else next((c for c in renew.columns if 'time' in c.lower() or 'date' in c.lower()), None)
        if r_time:
            renew['time'] = pd.to_datetime(renew[r_time], utc=True, errors='coerce')
            renew['time'] = renew['time'].dt.floor('h')
            renew = renew.dropna(subset=['time'])
            df = pd.merge(df, renew, on='time', how='left', suffixes=('', '_renew'))
            print(f"After Renewable Merge: {df.shape}")
    except Exception as e:
        print(f"Renewable merge failed: {e}")


load_col = next((c for c in df.columns if 'load' in c.lower() and 'actual' in c.lower()), None)
if not load_col:
    load_col = next((c for c in df.columns if 'total load' in c.lower()), None)

gen_cols = [c for c in df.columns if 'generation' in c.lower()]
print(f"\nUsing Load Column: {load_col}")
print(f"Found {len(gen_cols)} Generation Columns: {gen_cols[:5]}")

if not load_col or not gen_cols:
    raise ValueError(f"Columns not found! All columns: {df.columns.tolist()[:20]}")


df[gen_cols] = df[gen_cols].apply(pd.to_numeric, errors='coerce')
df[load_col] = pd.to_numeric(df[load_col], errors='coerce')

df['total_gen_MW'] = df[gen_cols].sum(axis=1)
df['total_load_MW'] = df[load_col]
df['Imbalance_MW'] = df['total_gen_MW'] - df['total_load_MW']

def get_status(x):
    if x > 1000: return 'High Excess'
    elif x > 0: return 'Excess'
    elif x > -1000: return 'Deficit'
    else: return 'High Deficit'

df['Grid_Status'] = df['Imbalance_MW'].apply(get_status)


df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['month'] = df['time'].dt.month
df['day_of_year'] = df['time'].dt.dayofyear
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)


if 'temp' in df.columns:
    df['temp_load_interaction'] = df['temp'] * df['total_load_MW'] / 1000
if 'wind_speed' in df.columns:
    df['wind_gen_proxy'] = df['wind_speed'] * 10  


df = df.sort_values('time')
df['load_lag_1h'] = df['total_load_MW'].shift(1)
df['load_lag_24h'] = df['total_load_MW'].shift(24)
df['gen_lag_1h'] = df['total_gen_MW'].shift(1)
df['imbalance_lag_1h'] = df['Imbalance_MW'].shift(1)
df['imbalance_rolling_24h'] = df['Imbalance_MW'].rolling(24).mean()


if os.path.exists('continuous dataset.csv'):
    try:
        cont = pd.read_csv('continuous dataset.csv')
        print(f"\nContinuous dataset found: {cont.shape} - You can merge it later for real-time stream simulation")
    except: pass

if os.path.exists('weekly pre-dispatch forecast.csv'):
    try:
        pre = pd.read_csv('weekly pre-dispatch forecast.csv')
        print(f"Pre-dispatch forecast found: {pre.shape} - Use for validation")
    except: pass


df = df.dropna(subset=['total_load_MW', 'total_gen_MW'])

first_cols = ['time', 'total_gen_MW', 'total_load_MW', 'Imbalance_MW', 'Grid_Status', 'hour', 'day_of_week', 'month']
other_cols = [c for c in df.columns if c not in first_cols]
df = df[first_cols + other_cols]

df.to_csv('FINAL_Imbalance_Dataset.csv', index=False)
df.head(100).to_csv('Sample_Check.csv', index=False)


df_ml = df.dropna()
df_ml.to_csv('FINAL_Imbalance_Dataset_ML_Ready.csv', index=False)

print("\n=== MERGING SUCCESS! ===")
print(f"Final Dataset Shape: {df.shape}")
print(f"ML Ready Shape: {df_ml.shape}")
print(f"\nImbalance Stats:\n{df['Imbalance_MW'].describe()}")
print(f"\nGrid Status Count:\n{df['Grid_Status'].value_counts()}")
print("\nFiles Created:")
print("1. FINAL_Imbalance_Dataset.csv - Main dataset")
print("2. FINAL_Imbalance_Dataset_ML_Ready.csv - For model training (no NaN)")
print("3. Sample_Check.csv - Quick check")
print("\n--- Ready for next step: model training ---")


