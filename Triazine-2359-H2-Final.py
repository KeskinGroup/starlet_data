import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np  # Needed for plotting
tpot_data = pd.read_csv('C:/ReddCOF-Data/Triazine Models/Triazine-2359-H2.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('H2-1 bar (mol/kg)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['H2-1 bar (mol/kg)'], train_size=0.80, test_size=0.20, random_state=42)

# Average CV score on the training set was: -8.570829760415197e-06
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    RidgeCV()
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
y_pred_train = exported_pipeline.predict(training_features)
preds = exported_pipeline.predict(testing_features)

# Create a list to store the metrics
metrics = []

# Calculate metrics
metrics.append({'Metric': 'R2', 'Train': r2_score(training_target, y_pred_train), 'Test': r2_score(testing_target, preds)})
metrics.append({'Metric': 'MSE', 'Train': mean_squared_error(training_target, y_pred_train), 'Test': mean_squared_error(testing_target, preds)})
metrics.append({'Metric': 'MAE', 'Train': mean_absolute_error(training_target, y_pred_train), 'Test': mean_absolute_error(testing_target, preds)})
metrics.append({'Metric': 'RMSE', 'Train': math.sqrt(mean_squared_error(training_target, y_pred_train)), 'Test': math.sqrt(mean_squared_error(testing_target, preds))})
metrics.append({'Metric': 'SRCC', 'Train': spearmanr(training_target, y_pred_train)[0], 'Test': spearmanr(testing_target, preds)[0]})

# Create a DataFrame from the list of metrics
metrics_df = pd.DataFrame(metrics)

# Print the DataFrame
print(metrics_df)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Scatter plot for training data
training_scatter = ax.scatter(training_target, y_pred_train, color="blue", label='Training Data')
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')

# Scatter plot for testing data
testing_scatter = ax.scatter(testing_target, preds, color="red", label='Testing Data')
ax.set_xlabel('Simulated')
ax.set_ylabel('ML-predicted')

# Plot the x=y line
x = np.linspace(min(min(training_target), min(testing_target)), max(max(training_target), max(testing_target)), 100)
ax.plot(x, x, color='black', linestyle='--', label='_nolegend_')

# Add legend with only "Training Data" and "Testing Data"
handles = [training_scatter, testing_scatter]
labels = [handle.get_label() for handle in handles]
ax.legend(handles=handles, labels=labels)

# Show the plot
plt.show()

# ==== SHAP  ====
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline  # <-- eklendi

# 0) Kaynak X (hız için örnekleme)
X_source = training_features if 'training_features' in globals() else features
X_disp = X_source.sample(n=min(1500, len(X_source)), random_state=42).copy()
cols = X_disp.columns

# 1) Model fit değilse fit et (güvenlik)
try:
    _ = exported_pipeline.get_booster()
except Exception:
    exported_pipeline.fit(training_features, training_target)

# 2) SHAP değerleri (Pipeline için callable, aksi halde TreeExplainer)
try:
    if isinstance(exported_pipeline, Pipeline):
        # TreeExplainer Pipeline'ı desteklemediği için callable kullan
        explainer = shap.Explainer(exported_pipeline.predict, X_disp)
        sv = explainer(X_disp)
        shap_values = sv.values
    else:
        explainer = shap.TreeExplainer(exported_pipeline)
        shap_values = explainer.shap_values(X_disp)
except Exception:
    # Genel yedek: callable varsa onu kullan
    model_callable = exported_pipeline.predict if hasattr(exported_pipeline, "predict") else exported_pipeline
    explainer = shap.Explainer(model_callable, X_disp)
    sv = explainer(X_disp)
    shap_values = sv.values

# (Nadiren) liste dönerse ilkini al
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# 3) Kendi sıralamamız (mean|SHAP|) ve çizim
order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
shap_values_ord = shap_values[:, order]
X_disp_ord = X_disp.iloc[:, order]
names_ord = [cols[i] for i in order]

plt.figure(figsize=(6, 4), dpi=150)
shap.summary_plot(
    shap_values_ord,
    X_disp_ord,
    feature_names=names_ord,
    show=False,
    max_display=5,
    sort=False  # sıralamayı biz verdik
)
plt.tight_layout()
os.makedirs("Figures", exist_ok=True)
plt.savefig("Figures/Triazine_SHAP_beeswarm_H2_XGB.png", dpi=300, bbox_inches="tight")
plt.show()

# 4) Ortalama |SHAP| önemleri (kayıt)
imp = pd.Series(np.abs(shap_values).mean(axis=0), index=cols).sort_values(ascending=False)
print("\nTop features by mean(|SHAP|):\n", imp.head(20))
imp.to_csv("Figures/SHAP_importances_H2_XGB.csv")

# (Opsiyonel) bar özeti:
plt.figure(figsize=(5, 3.6), dpi=300)
shap.summary_plot(shap_values_ord, X_disp_ord, plot_type="bar", show=False, max_display=20, sort=False)
plt.tight_layout(); plt.savefig("Figures/Triazine_SHAP_H2.png", dpi=1200, bbox_inches="tight"); plt.show()

