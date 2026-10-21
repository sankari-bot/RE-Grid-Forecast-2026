import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

np.random.seed(42)

df = pd.DataFrame({
    'index': list(range(100))*2,
    'length': list(np.random.randint(600,900,100)) + list(np.random.randint(1400,1900,100)),
    'label': ['Deficit (Low)']*100 + ['Excess (High)']*100
})

os.makedirs('static', exist_ok=True)

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="label", y="length", hue="label",
            palette={"Deficit (Low)":"red","Excess (High)":"green"},
            linewidth=2, fliersize=5, width=0.6)

plt.setp(plt.gca().artists, edgecolor='black', linewidth=2)

plt.title("Box Plot - Renewable Energy Imbalance Forecasting", fontsize=14, fontweight='bold')
plt.xlabel("Grid Status")
plt.ylabel("Imbalance (MW)")
plt.legend(title="Grid Status")
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('static/box_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved: static/box_plot.png | RED=Deficit 600-900, GREEN=Excess 1400-1900")