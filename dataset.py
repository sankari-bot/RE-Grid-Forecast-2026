import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
import pickle
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, r2_score
from sklearn.decomposition import PCA
print("Loading Smart Grid CSVs...")
energy = pd.read_csv('energy_dataset.csv', low_memory=False)
weather = pd.read_csv('weather_features.csv', low_memory=False)
print(f"Energy columns: {energy.columns.tolist()[:8]}")
print(f"Weather columns: {weather.columns.tolist()[:8]}")
print(f"Energy: {len(energy)}, Weather: {len(weather)}")
energy['time'] = pd.to_datetime(energy['time'], utc=True, errors='coerce')
weather['time'] = pd.to_datetime(weather['dt_iso'], utc=True, errors='coerce')
energy = energy.dropna(subset=['time'])
weather = weather.dropna(subset=['time'])
if 'city_name' in weather.columns:
    agg = {c: 'mean' for c in ['temp','humidity','wind_speed','pressure','clouds_all'] if c in weather.columns}
    weather = weather.groupby('time').agg(agg).reset_index()
    print(f"Weather aggregated: {len(weather)}")
energy['time'] = energy['time'].dt.floor('h')
weather['time'] = weather['time'].dt.floor('h')
df = pd.merge(energy.sort_values('time'), weather.sort_values('time'), on='time', how='inner')
df = df.drop_duplicates(subset=['time'], keep='first')
df = df.sort_values('time').reset_index(drop=True)
print(f"Merged Smart Grid: {len(df)}")

load_col = next((c for c in df.columns if 'load' in c.lower() and 'actual' in c.lower()), None)
gen_cols = [c for c in df.columns if 'generation' in c.lower()]
df[gen_cols] = df[gen_cols].apply(pd.to_numeric, errors='coerce')
df['total_gen_MW'] = df[gen_cols].sum(axis=1)
df['total_load_MW'] = pd.to_numeric(df[load_col], errors='coerce')
df['Imbalance_MW'] = df['total_gen_MW'] - df['total_load_MW']

def get_status(x):
    if x > 1000: return 3
    elif x > 0: return 2
    elif x > -1000: return 1
    else: return 0

df['Grid_Status'] = df['Imbalance_MW'].apply(get_status)
df['status_label'] = df['Imbalance_MW'].apply(lambda x: 'High Excess' if x>1000 else 'Excess' if x>0 else 'Deficit' if x>-1000 else 'High Deficit')

df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek 
df['month'] = df['time'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

df['load_lag_1h'] = df['total_load_MW'].shift(1)
df['load_lag_24h'] = df['total_load_MW'].shift(24)
df['imbalance_lag_1h'] = df['Imbalance_MW'].shift(1)
df['imbalance_rolling_24h'] = df['Imbalance_MW'].rolling(24, min_periods=1).mean()
df[['load_lag_1h','load_lag_24h','imbalance_lag_1h']] = df[['load_lag_1h','load_lag_24h','imbalance_lag_1h']].ffill()



features = ['hour','day_of_week','month','is_weekend','total_load_MW','total_gen_MW','load_lag_1h','load_lag_24h','imbalance_rolling_24h','temp','wind_speed']
features = [f for f in features if f in df.columns]
df = df.dropna(subset=features)
print(f"Final Smart Grid Dataset: {len(df)} rows")
print(f"Grid Status Count:\n{df['status_label'].value_counts()}")
X = df[features]
y_reg = df['Imbalance_MW']
y_class = df['Grid_Status']
X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(X, y_reg, y_class, test_size=0.2, random_state=42)
reg_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
acc_percent = 100 - (mae / (y_reg.abs().mean()+1) * 100)

print(f"\nRegression - MAE: {mae:.2f} MW, RMSE: {rmse:.2f} MW, R2: {r2:.4f}, Accuracy: {acc_percent:.2f}%")

class_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
class_model.fit(X_train, y_class_train)
y_class_pred = class_model.predict(X_test)
class_acc = accuracy_score(y_class_test, y_class_pred)
print(f"Classification - Grid Status Accuracy: {class_acc*100:.2f}%")

pickle.dump(reg_model, open('model.pk1','wb'))
pickle.dump(class_model, open('status_model.pk1','wb'))
print("Saved model.pk1 and status_model.pk1")

os.makedirs('static', exist_ok=True)

plt.figure(figsize=(8,4))
plt.plot(y_reg_test.values[:200], label='Actual Imbalance', color='#ef4444')
plt.plot(y_reg_pred[:200], label='Predicted Imbalance', color='#22c55e')
plt.title(f'Line Plot - Actual vs Predicted\nMAE: {mae:.2f} MW | R2: {r2:.2f}')
plt.xlabel('Time Steps'); plt.ylabel('Imbalance MW'); plt.legend()
plt.savefig('static/line_plot.png', bbox_inches='tight', dpi=150)
plt.close()

plt.figure(figsize=(4,4))
plt.bar(['Accuracy'], [acc_percent], color='#22c55e', width=0.5)
plt.ylim(0, 100)
plt.title(f'Model Accuracy\n{acc_percent:.2f}%', fontweight='bold', pad=20)
plt.ylabel('%')
plt.text(0, acc_percent/2, f'{acc_percent:.2f}%', ha='center', va='center', fontweight='bold', fontsize=20, color='white')
plt.savefig('static/bar_plot.png', bbox_inches='tight', dpi=150)
plt.close()

plt.figure(figsize=(5,4))
plt.hist(df['Imbalance_MW'], bins=50, color='#3b82f6', edgecolor='white')
plt.title('Histogram - Grid Imbalance Distribution'); plt.xlabel('Imbalance MW'); plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--', label='Balanced')
plt.legend()
plt.savefig('static/hist.png', bbox_inches='tight', dpi=120)
plt.close()

plt.figure(figsize=(5,3))
plt.plot(df['total_load_MW'].tail(500).values, color='#f59e0b')
plt.title('Smart Grid Load Profile - Last 500h'); plt.xlabel('Hours'); plt.ylabel('Load MW')
plt.savefig('static/word_count.png', bbox_inches='tight', dpi=120)
plt.close()

status_counts = df['status_label'].value_counts()
plt.figure(figsize=(5,5))
plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', explode=[0.05]*len(status_counts), shadow=True, startangle=90, textprops={'fontweight': 'bold'})
plt.title(f'PIE CHART - Grid Status\nTotal: {len(df)} hours', fontweight='bold')
plt.savefig('static/pie_chart.png', bbox_inches='tight', dpi=150)

print("Creating scatter.png ...")
X_sample = df[features].sample(n=530, random_state=42)
y_sample = df.loc[X_sample.index, 'Grid_Status']
X_2d = PCA(n_components=2).fit_transform(X_sample)

labels = {0:'High Deficit', 1:'Deficit', 2:'Excess', 3:'high Excess'}
plt.figure(figsize=(12,6))
for k in [0,1,2,3]:
    mask = y_sample==k
    if mask.sum()>0:
        plt.scatter(X_2d[mask,0], X_2d[mask,1], label=f'{labels[k]} ({mask.sum()})', alpha=0.7, s=30)

plt.title('Scatter Plot - Smart Grid Features + PCA', fontsize=14, pad=15)
plt.xlabel('PCA Component 1'); plt.ylabel('PCA Component 2')
plt.legend(title='Grid Status', loc='upper right')
plt.tight_layout()
plt.savefig('static/scatter.png', bbox_inches='tight', dpi=200)
plt.savefig('static/scatterplot.png', bbox_inches='tight', dpi=200)
plt.close()


cm = confusion_matrix(y_class_test, y_class_pred)
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title(f'Confusion Matrix - Grid Status\nAccuracy: {class_acc*100:.1f}%')
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,cm[i,j], ha='center', va='center', fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.xticks([0,1,2,3],['H.Def','Def','Exc','H.Exc']); plt.yticks([0,1,2,3],['H.Def','Def','Exc','H.Exc'])
plt.savefig('static/confusion.png', bbox_inches='tight', dpi=150)
plt.close()

print("\n=== ALL DONE ===")
print(f"Regression Accuracy: {acc_percent:.2f}%")
print(f"Status Accuracy: {class_acc*100:.2f}%")



