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
sns.lineplot(data=df, x="index", y="length", hue="label", 
             palette={"Deficit (Low)":"red","Excess (High)":"green"}, 
             marker='o', markersize=4, linewidth=2)

plt.title("Renewable Energy Imbalance Forecasting", fontsize=14, fontweight='bold')
plt.xlabel("Time Index")
plt.ylabel("Imbalance (MW)")
plt.legend(title="Grid Status")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/line_plot.png', dpi=300)
plt.show()

