import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_excel('E:/ZIF8IL/MLmodels/V11/rawdatamodels/CO2_01BAR.xlsx',  dtype=np.float64)
features = tpot_data.drop('CO2 Uptake (mol/kg) - 0.1 bar', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['CO2 Uptake (mol/kg) - 0.1 bar'], train_size=0.80, test_size=0.20, random_state=42)

# Average CV score on the training set was: -0.0011820607523277593
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=8, max_features=0.5, min_samples_leaf=1, min_samples_split=9, n_estimators=100, subsample=0.8500000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
y_pred_train = exported_pipeline.predict(training_features.values)
preds = exported_pipeline.predict(testing_features.values)


# Calculate and print SRCC
srcc_train, _ = spearmanr(training_target, y_pred_train)
srcc_test, _ = spearmanr(testing_target, preds)

print('SRCC_Train: %.3f' % srcc_train)
print('SRCC_Test: %.3f' % srcc_test)

# ACCURACY
print('R2_Train: %.3f' % r2_score(training_target, y_pred_train))
print('R2_Test: %.3f' % r2_score(testing_target, preds))
print('MSE_Train: %.10f' % mean_squared_error(training_target, y_pred_train))
print('MSE_Test: %.10f' % mean_squared_error(testing_target, preds))
print('MAE_Train: %.10f' % mean_absolute_error(training_target, y_pred_train))
print('MAE_Test: %.10f' % mean_absolute_error(testing_target, preds))
mse_train = mean_squared_error(training_target, y_pred_train)
rmse_train = math.sqrt(mse_train)
mse_test = mean_squared_error(testing_target, preds)
rmse_test = math.sqrt(mse_test)

print('RMSE_Train: %.7f' % rmse_train)
print('RMSE_Test: %.7f' % rmse_test)

fig, ax = plt.subplots(figsize=(6, 4.9), dpi=300)  # Set the figsize and dpi
plt.scatter(training_target, y_pred_train, color="#00a0d6", edgecolors='black', s=100)
plt.xlabel('truevalues_train')
plt.ylabel('predictedvalues_train')
plt.xlim(0, 0.70) # Set the x-axis limits 
plt.ylim(0, 0.70) # Set the y-axis limits 
plt.scatter(testing_target, preds, color="#f03592", edgecolors='black', s=100)
plt.xlabel(r'Simulated $\mathrm{N}_{\mathrm{CO}_2}$ (mol/kg)', fontsize=20)
plt.ylabel(r'ML-predicted $\mathrm{N}_{\mathrm{CO}_2}$ (mol/kg)',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tick_params(axis='both', which='major', width=2)

# Increase the border thickness of the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
# Add a legend to the plot
    # Add a legend to the plot
    plt.legend(fontsize=14, frameon=False, loc='upper right')
    plt.text(0.02, 0.98, '(a) 0.1 bar', transform=ax.transAxes, ha='left', va='top', fontsize=18)
    text = '$\mathrm{R}^2$:0.803\nRMSE:0.034\nMAE:0.020\nSRCC:%.3f' % srcc_test
    plt.text(0.99, 0.03, text, transform=ax.transAxes, ha='right', va='bottom', multialignment='left', fontsize=16, color='#f03592')
    text = '$\mathrm{R}^2$:0.979\nRMSE:0.010\nMAE:0.03\nSRCC:%.3f' % srcc_train
    plt.text(0.03, 0.90, text, transform=ax.transAxes, ha='left', va='top', multialignment='left', fontsize=16, color='#00a0d6')
plt.show()


