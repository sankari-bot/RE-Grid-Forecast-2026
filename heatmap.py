import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs('static', exist_ok=True)

data = {
    'Generation_MW': np.random.randint(4000, 6000, 100),
    'Load_MW': np.random.randint(4200, 5800, 100),
    'Wind_MW': np.random.randint(500, 1500, 100),
    'Solar_MW': np.random.randint(300, 1200, 100),
    'Temperature_C': np.random.randint(20, 40, 100),
    'Humidity_%': np.random.randint(40, 90, 100),
}
data['Imbalance_MW'] = np.array(data['Generation_MW']) - np.array(data['Load_MW'])
data['Excess_MW'] = np.where(data['Imbalance_MW'] > 0, data['Imbalance_MW'], 0)
data['Deficit_MW'] = np.where(data['Imbalance_MW'] < 0, np.abs(data['Imbalance_MW']), 0)

df = pd.DataFrame(data)
corr = df.corr()


plt.figure(figsize=(12, 9))

cmap = sns.diverging_palette(10, 130, as_cmap=True)

ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, 
                 linewidths=1.5, linecolor='black',
                 annot_kws={"size": 10, "weight": "bold"},
                 cbar_kws={"shrink": 0.8}, 
                 vmin=-1, vmax=1)


plt.title("HEATMAP - Renewable Energy Grid Imbalance Forecasting\n(200MW GREEN Excess | -800MW RED Deficit)", 
          fontsize=13, fontweight='bold', pad=12)

for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_edgecolor('black')

plt.xticks(rotation=30, ha='right', fontsize=9, fontweight='bold')
plt.yticks(rotation=0, fontsize=9, fontweight='bold')


plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig('static/heatmap.png', dpi=300)
plt.show()

