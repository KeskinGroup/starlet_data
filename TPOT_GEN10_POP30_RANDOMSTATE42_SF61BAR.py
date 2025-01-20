import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from scipy.stats import spearmanr
from scipy.stats import spearmanr, gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_excel('SF6_1BAR.xlsx',  dtype=np.float64)
features = tpot_data.drop('SF6 Uptake 1 bar (mol/kg)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['SF6 Uptake 1 bar (mol/kg)'], train_size=0.80, test_size=0.20, random_state=42)

# Average CV score on the training set was: -0.37325180699903926
exported_pipeline = make_pipeline(
    ZeroCount(),
    RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=3, min_samples_split=6, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
y_pred_train=exported_pipeline.predict(training_features)
preds = exported_pipeline.predict(testing_features)


#ACCURACY
print('R2_Train: %.3f' % r2_score(training_target, y_pred_train))
print('R2_Test: %.3f' % r2_score(testing_target, preds))
print('MSE_Train: %.10f' % mean_squared_error(training_target, y_pred_train))
print('MSE_Test: %.10f' %mean_squared_error(testing_target, preds))
print('MAE_Train: %.10f' % mean_absolute_error(training_target, y_pred_train))
print('MAE_Test: %.10f' %mean_absolute_error(testing_target, preds))
mse_train = mean_squared_error(training_target, y_pred_train)
rmse_train = math.sqrt(mse_train)
mse_test = mean_squared_error(testing_target, preds)
rmse_test = math.sqrt(mse_test)

print('RMSE_Train: %.7f' % rmse_train)
print('RMSE_Test: %.7f' % rmse_test)

coef1, p = spearmanr(training_target, y_pred_train)
coef2, p = spearmanr(testing_target, preds)

print('SRCC_Train: %.3f' % coef1)
print('SRCC_Test: %.3f' % coef2)

# Combine scatter plots
df_train = pd.DataFrame({'Target': training_target, 'Predictions': y_pred_train, 'Set': 'Training'})
df_test = pd.DataFrame({'Target': testing_target, 'Predictions': preds, 'Set': 'Test'})
combined_df = pd.concat([df_test, df_train])

# Create a joint plot with scatter and histograms
palette = {'Training': '#8a2be2', 'Test': '#065535'}
g = sns.JointGrid(data=combined_df, x='Target', y='Predictions', hue='Set', space=0, ratio=4, palette=palette, height=8)
g = g.plot(sns.scatterplot, sns.histplot, alpha=1, edgecolor='black')

# Calculate the KDE manually
kde_train = gaussian_kde(training_target)
kde_test = gaussian_kde(testing_target)
x = np.linspace(min(combined_df['Target']), max(combined_df['Target']), 1000)
y_train = kde_train(x)
y_test = kde_test(x)

# Normalize the KDE
max_height_x = max([h.get_height() for h in g.ax_marg_x.patches])
scale_train = max_height_x / max(y_train)
scale_test = max_height_x / max(y_test)

# Plot scaled KDEs on the marginal histograms
g.ax_marg_x.plot(x, y_train * scale_train, color='#8a2be2', linewidth=3)
g.ax_marg_x.plot(x, y_test * scale_test, color='#065535', linewidth=3)

# For the y-axis marginal (right)
y_values = np.linspace(min(combined_df['Predictions']), max(combined_df['Predictions']), 1000)
y_density_train = kde_train(y_values)
y_density_test = kde_test(y_values)
max_width_y = max([h.get_width() for h in g.ax_marg_y.patches])
scale_y_train = max_width_y / max(y_density_train)
scale_y_test = max_width_y / max(y_density_test)

g.ax_marg_y.plot(y_density_train * scale_y_train, y_values, color='#8a2be2', linewidth=3)
g.ax_marg_y.plot(y_density_test * scale_y_test, y_values, color='#065535', linewidth=3)

# Set axis labels with subscript using LaTeX formatting
g.set_axis_labels(r'Simulated N$_{SF_6}$ (mol/kg)', r'Predicted N$_{SF_6}$ (mol/kg)', fontsize=27)

# Set both axes to the same scale
scale_min = -0.1
scale_max = 7.5  # Ensure the same range for both axes
g.ax_joint.set_xlim(scale_min, scale_max)
g.ax_joint.set_ylim(scale_min, scale_max)
g.ax_joint.set_aspect('equal', 'box')  # Ensure equal scaling

# Set custom ticks to include 3.5
ticks = np.arange(0, 7.5, 1)
g.ax_joint.set_xticks(ticks)
g.ax_joint.set_yticks(ticks)


# Increase the size of scatter plot markers
g.ax_joint.collections[0].set_sizes([100])

# Set font size for x and y-axis tick labels
g.ax_joint.tick_params(axis='both', which='major', labelsize=23)

# Set font size for x and y-axis tick labels and increase the size of tick marks
g.ax_joint.tick_params(axis='both', which='major', labelsize=22, length=8, width=2)  # Increase the font size and tick size



# Increase the thickness of the axis lines
g.ax_joint.spines['bottom'].set_linewidth(2)
g.ax_joint.spines['left'].set_linewidth(2)
g.ax_joint.spines['top'].set_linewidth(2)
g.ax_joint.spines['right'].set_linewidth(2)

# Add x=y line
x = np.linspace(scale_min, scale_max, 100)
g.ax_joint.plot(x, x, color='black', linestyle='--', label='_nolegend_')

# Remove the legend
g.ax_joint.legend_.remove()

# Add custom legend manually
plt.text(0.02, 0.98, '(b) 1 bar ', transform=g.ax_joint.transAxes, fontsize=23, verticalalignment='top')

# Add R², RMSE, and MAE values manually
plt.text(0.01, 0.85, f'R²: {r2_score(training_target, y_pred_train):.2f}', fontsize=22, color='#8a2be2', transform=g.ax_joint.transAxes)
plt.text(0.65, 0.12, f'R²: {r2_score(testing_target, preds):.2f}', fontsize=22, color='#065535', transform=g.ax_joint.transAxes)
plt.text(0.01, 0.80, f'RMSE: {rmse_train:.3f}', fontsize=22, color='#8a2be2', transform=g.ax_joint.transAxes)
plt.text(0.65, 0.07, f'RMSE: {rmse_test:.3f}', fontsize=22, color='#065535', transform=g.ax_joint.transAxes)
plt.text(0.01, 0.75, f'MAE: {mean_absolute_error(training_target, y_pred_train):.3f}', fontsize=22, color='#8a2be2', transform=g.ax_joint.transAxes)
plt.text(0.65, 0.02, f'MAE: {mean_absolute_error(testing_target, preds):.3f}', fontsize=22, color='#065535', transform=g.ax_joint.transAxes)

plt.show()
