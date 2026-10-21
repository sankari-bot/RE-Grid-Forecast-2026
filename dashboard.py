import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
n=200
time = pd.date_range('2026-01-01', periods=n, freq='h')
df = pd.DataFrame({
    'Time': time,
    'Generation_MW': np.random.randint(4000,6000,n),
    'Load_MW': np.random.randint(4200,5800,n),
    'Wind_MW': np.random.randint(500,1500,n),
    'Solar_MW': np.random.randint(300,1200,n),
})
df['Imbalance_MW'] = df['Generation_MW'] - df['Load_MW']
df['Status'] = np.where(df['Imbalance_MW']>0, 'GREEN Excess', 'RED Deficit')
df['Excess_MW'] = np.where(df['Imbalance_MW']>0, df['Imbalance_MW'], 0)
df['Deficit_MW'] = np.where(df['Imbalance_MW']<0, np.abs(df['Imbalance_MW']), 0)
corr = df[['Generation_MW','Load_MW','Wind_MW','Solar_MW','Imbalance_MW','Excess_MW','Deficit_MW']].corr()


fig, axs = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
fig.suptitle("RE Grid Imbalance - 6-in-1 Analysis Dashboard\nHow It Works: Generation - Load = Imbalance (200MW GREEN | -800MW RED)", 
             fontsize=13, fontweight='bold')


axs[0,0].plot(df['Time'][:50], df['Generation_MW'][:50], label='Generation', color='green', linewidth=1.5)
axs[0,0].plot(df['Time'][:50], df['Load_MW'][:50], label='Load', color='red', linewidth=1.5)
axs[0,0].set_title("1. LINEPLOT - Gen vs Load", fontweight='bold', fontsize=11)
axs[0,0].legend(fontsize=8)
axs[0,0].tick_params(axis='x', rotation=20, labelsize=7)


counts = df['Status'].value_counts()
axs[0,1].bar(counts.index, counts.values, color=['#00C853','#D50000'], edgecolor='black', linewidth=1.5)
axs[0,1].set_title("2. BARPLOT - Excess vs Deficit", fontweight='bold', fontsize=11)


colors = np.where(df['Imbalance_MW']>0, 'green', 'red')
axs[0,2].scatter(df['Generation_MW'], df['Imbalance_MW'], c=colors, alpha=0.6, edgecolor='black', s=20)
axs[0,2].axhline(y=200, color='green', linestyle='--', linewidth=1)
axs[0,2].axhline(y=-800, color='red', linestyle='--', linewidth=1)
axs[0,2].set_title("3. SCATTER - Gen vs Imbalance", fontweight='bold', fontsize=11)
axs[0,2].set_xlabel("Generation_MW", fontsize=8)
axs[0,2].set_ylabel("Imbalance_MW", fontsize=8)


axs[1,0].hist(df['Imbalance_MW'], bins=30, color='orange', edgecolor='black')
axs[1,0].axvline(x=200, color='green', linewidth=2, label='GREEN 200')
axs[1,0].axvline(x=-800, color='red', linewidth=2, label='RED -800')
axs[1,0].set_title("4. HISTOGRAM - Imbalance Distribution", fontweight='bold', fontsize=11)
axs[1,0].legend(fontsize=7)
axs[1,0].set_xlabel("Imbalance_MW", fontsize=8)


im = axs[1,1].imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
axs[1,1].set_xticks(range(len(corr.columns)))
axs[1,1].set_yticks(range(len(corr.columns)))
axs[1,1].set_xticklabels(corr.columns, rotation=30, ha='right', fontsize=6, fontweight='bold')
axs[1,1].set_yticklabels(corr.columns, fontsize=6, fontweight='bold')
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        axs[1,1].text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=5, fontweight='bold', color="white" if abs(corr.values[i,j])>0.5 else "black")
axs[1,1].set_title("5. HEATMAP - Correlation", fontweight='bold', fontsize=11)


data_box = [df['Wind_MW'], df['Solar_MW'], df['Generation_MW']/4, df['Imbalance_MW']+1000]
axs[1,2].boxplot(data_box, patch_artist=True, tick_labels=['Wind','Solar','Gen/4','Imb+1k'])
axs[1,2].set_title("6. BOXPLOT - RE Distribution", fontweight='bold', fontsize=11)
axs[1,2].set_ylabel("MW", fontsize=8)

plt.savefig('static/dashboard_6in1.png', dpi=300)
plt.show()

