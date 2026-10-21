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
sns.histplot(data=df, x="length", hue="label", 
             palette={"Deficit (Low)":"red","Excess (High)":"green"},
             bins=20, kde=True, alpha=0.6)

plt.title("Histogram - Renewable Energy Imbalance Forecasting", fontsize=14, fontweight='bold')
plt.xlabel("Imbalance (MW)")
plt.ylabel("Frequency")
plt.legend(title="Grid Status")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/histogram.png', dpi=300)
plt.show()

