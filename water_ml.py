import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. LOAD & PREPROCESS ──────────────────────────────────────────────────────
df = pd.read_csv('daily_water_consumption.csv')

le_gender   = LabelEncoder()
le_activity = LabelEncoder()
le_city     = LabelEncoder()

df['Gender_enc']   = le_gender.fit_transform(df['Gender'])
df['Activity_enc'] = le_activity.fit_transform(df['Activity_Level'])
df['City_enc']     = le_city.fit_transform(df['City'])

features = ['Age', 'Gender_enc', 'Temperature_C', 'Activity_enc', 'City_enc']
X = df[features]
y = df['Water_Consumed_Liters']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 2. TRAIN MODELS ───────────────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR':               SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

results     = {}
predictions = {}

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results[name]     = {'RMSE': round(rmse,4), 'MAE': round(mae,4),
                         'MSE':  round(mse,4),  'R2':  round(r2,4)}
    predictions[name] = y_pred

print("\n", pd.DataFrame(results).T)

# ── 3. VISUALIZATION ─────────────────────────────────────────────────────────
colors      = {'Linear Regression': '#4C72B0', 'Random Forest': '#55A868', 'SVR': '#C44E52'}
model_names = list(results.keys())

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Water Consumption Prediction — Model Comparison',
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Row 1 — Error metrics bar charts
for i, metric in enumerate(['RMSE', 'MAE', 'MSE']):
    ax   = fig.add_subplot(gs[0, i])
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(model_names, vals,
                  color=[colors[m] for m in model_names], width=0.5, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=11)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

# Row 2 — Actual vs Predicted scatter
sample   = min(500, len(y_test))
y_sample = y_test.values[:sample]

for i, name in enumerate(model_names):
    ax       = fig.add_subplot(gs[1, i])
    y_pred_s = predictions[name][:sample]
    ax.scatter(y_sample, y_pred_s, alpha=0.4, color=colors[name], s=12)
    mn, mx = y_sample.min(), y_sample.max()
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.2, label='Perfect fit')
    ax.set_xlabel('Actual (Liters)', fontsize=9)
    ax.set_ylabel('Predicted (Liters)', fontsize=9)
    ax.set_title(f'{name}\nActual vs Predicted', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.05, 0.92, f'R2={results[name]["R2"]}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Row 3 left — R2 bar chart
ax_r2   = fig.add_subplot(gs[2, :2])
r2_vals = [results[m]['R2'] for m in model_names]
bars    = ax_r2.bar(model_names, r2_vals,
                    color=[colors[m] for m in model_names], width=0.4, edgecolor='white')
for bar, val in zip(bars, r2_vals):
    ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_r2.set_title('R2 Score Comparison (higher = better)', fontweight='bold', fontsize=11)
ax_r2.set_ylabel('R2 Score')
ax_r2.set_ylim(0, 1.15)
ax_r2.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax_r2.set_xticks(range(len(model_names)))
ax_r2.set_xticklabels(model_names, fontsize=10)
ax_r2.grid(axis='y', alpha=0.3)
ax_r2.spines[['top', 'right']].set_visible(False)

# Row 3 right — Feature Importance (Random Forest)
ax_fi       = fig.add_subplot(gs[2, 2])
rf_model    = models['Random Forest']
importances = rf_model.feature_importances_
feat_labels = ['Age', 'Gender', 'Temperature', 'Activity Level', 'City']
sorted_idx  = np.argsort(importances)
ax_fi.barh([feat_labels[i] for i in sorted_idx],
           [importances[i] for i in sorted_idx],
           color='#55A868', edgecolor='white')
ax_fi.set_title('Feature Importance\n(Random Forest)', fontweight='bold', fontsize=10)
ax_fi.set_xlabel('Importance Score')
ax_fi.grid(axis='x', alpha=0.3)
ax_fi.spines[['top', 'right']].set_visible(False)

# ── SAVE & SHOW ───────────────────────────────────────────────────────────────
plt.savefig('water_ml_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\nPlot saved as 'water_ml_results.png'")

# Uncomment baris di bawah jika ingin popup window (perlu GUI / Tkinter)
# matplotlib.use('TkAgg')
# plt.show()
