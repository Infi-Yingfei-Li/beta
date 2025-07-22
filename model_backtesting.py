#%%
import os, sys, copy, scipy, datetime, tqdm, collections, itertools, pickle, json
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from contextlib import redirect_stdout

import model_lib

if __name__ == "__main__":
    sys.path.append(os.path.join(os.getcwd(), '..'))

import utils

file_name = os.path.join(os.getcwd(), "..", "feature_engineer", "feature_engineer_summary.pkl")
with open(file_name, "rb") as f:
    feature_selection_summary = pickle.load(f)
feature_pool = feature_selection_summary["selected_feature"]

log = {}

#%% becnhmark
feature = ["beta", "return_", "volume"]
feature_label = "baseline"
model = model_lib.model_linear_regression(feature, feature_label)
time_hist, mse_hist = model.model_performance(is_stock_dependent=False, auto_select_feature=True)
log["baseline"] = np.nanmean(mse_hist)

#%%
feature = feature_pool[0:5]
feature_label = "top5_feature"
model = model_lib.model_linear_regression(feature, feature_label)
time_hist, mse_hist = model.model_performance(is_stock_dependent=False, auto_select_feature=True)
plt.plot(time_hist, mse_hist, "-o", label="False, True")

time_hist, mse_hist = model.model_performance(is_stock_dependent=False, auto_select_feature=False)
plt.plot(time_hist, mse_hist, "-o", label="False, False")
log["linear_regression"] = np.nanmean(mse_hist)

time_hist, mse_hist = model.model_performance(is_stock_dependent=True, auto_select_feature=False)
plt.plot(time_hist, mse_hist, "-o", label="True, False")

time_hist, mse_hist = model.model_performance(is_stock_dependent=True, auto_select_feature=True)
plt.plot(time_hist, mse_hist, "-o", label="True, True")

plt.legend(loc="upper left", title="is_stock_dependent, is_autoselect_feature", framealpha=0)
plt.ylim(0.08, 0.22)
plt.ylabel("MSE"); plt.xlabel("Time")
plt.title("Linear Regression")

#%% ridge lasso regression
feature = feature_pool[0:5]
feature_label = "top5_feature"
regularization = "ridge"
model = model_lib.model_ridge_lasso_regression(feature, feature_label, regularization)
time_hist, mse_hist = model.model_performance(is_stock_dependent=False)
plt.plot(time_hist, mse_hist, "-o", label="ridge, all stock")
log["ridge_regression"] = np.nanmean(mse_hist)

time_hist, mse_hist = model.model_performance(is_stock_dependent=True)
plt.plot(time_hist, mse_hist, "-o", label="ridge, single stock")

regularization = "lasso"
model = model_lib.model_ridge_lasso_regression(feature, feature_label, regularization)
time_hist, mse_hist = model.model_performance(is_stock_dependent=False)
plt.plot(time_hist, mse_hist, "-o", label="lasso, all stock")
log["lasso_regression"] = np.nanmean(mse_hist)

time_hist, mse_hist = model.model_performance(is_stock_dependent=True)
plt.plot(time_hist, mse_hist, "-o", label="lasso, single stock")

plt.ylabel("MSE"); plt.xlabel("Time")
plt.ylim(0.08, 0.22)
plt.legend(title="type, is_stock_dependent")
plt.title("Ridge and Lasso Regression")

#%% principal component regression
feature = feature_pool[0:5]
model = model_lib.model_principal_component_regression(feature, "top5_feature")
time_hist, mse_hist = model.model_performance(is_stock_dependent=False)
plt.plot(time_hist, mse_hist, "-o", label="False")
log["principal_component_regression"] = np.nanmean(mse_hist)

time_hist, mse_hist = model.model_performance(is_stock_dependent=True)
plt.plot(time_hist, mse_hist, "-o", label="True")
plt.legend(title="is_stock_dependent")
plt.ylabel("MSE"); plt.xlabel("Time")
plt.ylim(0.08, 0.22)
plt.title("Principal Component Regression")

#%% least angle regression
feature = feature_pool[0:5]
model = model_lib.model_least_angle_regression(feature, "top5_feature")
time_hist, mse_hist = model.model_performance(is_stock_dependent=False)
plt.plot(time_hist, mse_hist, "-o", label="False")
log["least_angle_regression"] = np.nanmean(mse_hist)

time_hist, mse_hist = model.model_performance(is_stock_dependent=True)
plt.plot(time_hist, mse_hist, "-o", label="True")
plt.legend(title="is_stock_dependent")
plt.ylabel("MSE"); plt.xlabel("Time")
plt.ylim(0.08, 0.22)
plt.title("Least Angle Regression")

#%% partial least square regression
feature = feature_pool[0:5]
model = model_lib.model_partial_least_square_regression(feature, "top5_feature")
time_hist, mse_hist = model.model_performance(is_stock_dependent=False)
plt.plot(time_hist, mse_hist, "-o", label="False")
log["partial_least_square_regression"] = np.nanmean(mse_hist)

time_hist, mse_hist = model.model_performance(is_stock_dependent=True)
plt.plot(time_hist, mse_hist, "-o", label="True")
plt.legend(title="is_stock_dependent")
plt.ylabel("MSE"); plt.xlabel("Time")
plt.ylim(0.08, 0.22)
plt.title("Partial Least Square Regression")

#%% random forest regression
feature = feature_pool[0:5]
model = model_lib.model_tree_based_regression(feature, "top5_feature", is_gradient_boost=False)
time_hist, mse_hist = model.model_performance(is_stock_dependent=False)
plt.plot(time_hist, mse_hist, "-o", label="False")
log["random_forest_regression"] = np.nanmean(mse_hist)

time_hist, mse_hist = model.model_performance(is_stock_dependent=True)
plt.plot(time_hist, mse_hist, "-o", label="True")
plt.legend(title="is_stock_dependent")
plt.ylabel("MSE"); plt.xlabel("Time")
plt.ylim(0.08, 0.22)
plt.title("Random Forest Regression")

#%% gradient boosting regression
feature = feature_pool[0:5]
model = model_lib.model_tree_based_regression(feature, "top5_feature", is_gradient_boost=True)
time_hist, mse_hist = model.model_performance(is_stock_dependent=False)
plt.plot(time_hist, mse_hist, "-o", label="False")
log["gb_random_forest_regression"] = np.nanmean(mse_hist)

time_hist, mse_hist = model.model_performance(is_stock_dependent=True)
plt.plot(time_hist, mse_hist, "-o", label="True")
plt.legend(title="is_stock_dependent")
plt.ylabel("MSE"); plt.xlabel("Time")
plt.ylim(0.08, 0.22)
plt.title("Gradient Boosting Regression")

#%% neural networks
feature = feature_pool[0:5]
model = model_lib.model_neural_networks(feature, "top5_feature")
time_hist, mse_hist = model.model_performance()
log["neural_networks"] = np.nanmean(mse_hist)
plt.plot(time_hist, mse_hist, "-o")
plt.legend()
plt.ylabel("MSE"); plt.xlabel("Time")
plt.ylim(0.08, 0.22)
plt.title("Neural Networks")

#%% summary
item = list(log.items())
item.sort(key=lambda x: x[1], reverse=True)
plt.figure(figsize=(6, 4))
plt.barh([x[0] for x in item], [x[1] for x in item])
plt.vlines(x=log["baseline"], ymin=-0.5, ymax=len(item)-0.5, color="red", linestyles='--', label="baseline")
plt.xlabel("MSE"); plt.ylabel("Model")
plt.legend(loc="lower left")
plt.title("Model Comparison")

#%%
baseline = log["baseline"]
labels = [x[0] for x in item]
values = [100*(baseline-x[1])/baseline for x in item]

for i, v in item:
    print("model: {}, MSE: {:.4f}, improvement: {:.4f}%".format(i, v, 100*(baseline-v)/baseline))

fig = plt.figure(figsize=(8, 4))
fig, (axl, axr) = plt.subplots(ncols=2, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 4]})

axl.barh(labels, values, color='C0')
axl.set_xlim(-17, -12)
axl.spines["right"].set_visible(False)
axl.set_ylabel("Model")

for i in range(len(labels)):
    axl.axhline(y=i, color='k', linewidth=1, alpha=0.3)
    axr.axhline(y=i, color='k', linewidth=1, alpha=0.3)

axr.barh(labels, values, color='C1')
axr.set_yticks([])
axr.set_yticklabels([])
axr.set_xlim(0, 8)
axr.spines["left"].set_visible(False)
axr.set_xlabel("MSE Improvement (%)")

d = 0.015
bboxl = axl.get_position()
bboxr = axr.get_position()
ratioL = bboxl.width / bboxl.height
ratioR = bboxr.width / bboxr.height

axl.plot((1 - d/ratioL, 1 + d/ratioL), (-d, +d), transform=axl.transAxes, color='k', clip_on=False)
axl.plot((1 - d/ratioL, 1 + d/ratioL), (1 - d, 1 + d), transform=axl.transAxes, color='k', clip_on=False)

axr.plot((-d/ratioR, +d/ratioR), (-d, +d), transform=axr.transAxes, color='k', clip_on=False)
axr.plot((-d/ratioR, +d/ratioR), (1 - d, 1 + d), transform=axr.transAxes, color='k', clip_on=False)

plt.tight_layout()

from matplotlib.patches import Rectangle

bboxL = axl.get_position()
bboxR = axr.get_position()
x_left = bboxL.x1
x_right = bboxR.x0
y_bottom = bboxL.y0
y_top = bboxL.y1

# Add rectangle to the figure
fig.patches.append(Rectangle(
    (x_left, y_bottom),               # (x, y) bottom-left in figure coords
    x_right - x_left,                 # width
    y_top - y_bottom,                 # height
    transform=fig.transFigure,
    color='lightgray',
    zorder=2,
    clip_on=False,
    alpha=0.5
))

plt.savefig(os.path.join(os.getcwd(), "results", "model_comparison.pdf"), dpi=300, bbox_inches='tight')

#%%
plt.figure(figsize=(6, 4))
model = model_lib.model_linear_regression(feature, feature_label)
time_hist, mse_hist = model.model_performance(is_stock_dependent=False, auto_select_feature=False)
plt.plot(time_hist, mse_hist, "-o", label="linear_regression")

model = model_lib.model_tree_based_regression(feature, "top5_feature", is_gradient_boost=True)
time_hist, mse_hist = model.model_performance(is_stock_dependent=False)
plt.plot(time_hist, mse_hist, "-o", label="gb_random_forest_regression")
plt.legend()
plt.ylabel("MSE"); plt.xlabel("Time")
plt.ylim(0.04, 0.22)
plt.title("Linear regression vs Gradient Boosting Regression")

# %%
file_name = os.path.join(os.getcwd(), "results", "model_performance_summary.pkl")
with open(file_name, "wb") as f:
    pickle.dump(log, f)

# %%
