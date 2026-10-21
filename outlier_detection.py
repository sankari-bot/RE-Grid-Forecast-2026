import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import IsolationForest

np.random.seed(42)
os.makedirs('static', exist_ok=True)


data = {
    'Generation_MW': list(np.random.randint(4500, 5500, 95)) + [8000, 1000, 9000, 500, 7500],  
    'Load_MW': list(np.random.randint(4400, 5400, 95)) + [2000, 7000, 1500, 8000, 1000],
    'Wind_MW': np.random.randint(500, 1500, 100),
    'Solar_MW': np.random.randint(300, 1200, 100),
}
data['Imbalance_MW'] = np.array(data['Generation_MW']) - np.array(data['Load_MW'])
df = pd.DataFrame(data)

print("=== OUTLIER DETECTION FOR RE-GRID PROJECT ===")


Q1 = df['Imbalance_MW'].quantile(0.25)
Q3 = df['Imbalance_MW'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df['IQR_Outlier'] = (df['Imbalance_MW'] < lower) | (df['Imbalance_MW'] > upper)


from scipy import stats
df['Z_Score'] = np.abs(stats.zscore(df['Imbalance_MW']))
df['Z_Outlier'] = df['Z_Score'] > 3


iso = IsolationForest(contamination=0.05, random_state=42)
df['IF_Outlier'] = iso.fit_predict(df[['Generation_MW','Load_MW','Imbalance_MW']])
df['IF_Outlier'] = df['IF_Outlier'] == -1

print(f"Total: {len(df)} | IQR Outliers: {df['IQR_Outlier'].sum()} | Z Outliers: {df['Z_Outlier'].sum()} | IsolationForest: {df['IF_Outlier'].sum()}")
print(df[df['IF_Outlier']][['Generation_MW','Load_MW','Imbalance_MW']])


plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
sns.boxplot(y=df['Imbalance_MW'], color='lightgreen', linewidth=2, fliersize=8, flierprops=dict(marker='o', markerfacecolor='red', markersize=10))
plt.title("BoxPlot - Outliers in RED (200MW Normal, 8000MW Outlier)", fontweight='bold')
plt.ylabel("Imbalance (MW)")
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
colors = np.where(df['IF_Outlier'], 'red', 'green')
plt.scatter(df.index, df['Imbalance_MW'], c=colors, s=80, edgecolor='black', alpha=0.7)
plt.axhline(upper, color='red', linestyle='--', label=f'Upper {upper:.0f}')
plt.axhline(lower, color='red', linestyle='--', label=f'Lower {lower:.0f}')
plt.axhline(200, color='green', linestyle='-', label='Normal 200MW Excess')
plt.title("IsolationForest - RED = Outlier / GREEN = Normal", fontweight='bold')
plt.xlabel("Time Index")
plt.ylabel("Imbalance (MW)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('static/outlier_detection.png', dpi=300, bbox_inches='tight')
plt.show()


df_clean = df[~df['IF_Outlier']].copy()
df_clean.to_csv('FINAL_Imbalance_Dataset_Cleaned.csv', index=False)
print(f"\nCleaned: {len(df_clean)} rows saved | static/outlier_detection.png ready for Viva!")

