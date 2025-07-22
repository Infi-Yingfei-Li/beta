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

file_name = os.path.join(os.path.dirname(__file__), "..", "feature_engineer", "feature_engineer_summary.pkl")
with open(file_name, "rb") as f:
    feature_selection_summary = pickle.load(f)
feature_pool = feature_selection_summary["selected_feature"]

log = collections.defaultdict(dict)

#%% Linear regression
current_feature = ["beta", "return_", "volume"]
feature_label = "baseline"
model = model_lib.model_linear_regression(current_feature, feature_label)
model.model_performance(is_stock_dependent=False, auto_select_feature=True)
model.model_performance(is_stock_dependent=False, auto_select_feature=False)
model.model_performance(is_stock_dependent=True, auto_select_feature=False)
model.model_performance(is_stock_dependent=True, auto_select_feature=True)
file_name = model.model_performance_file_name
with open(file_name, "rb") as f:
    model_performance_summary = pickle.load(f)
log[feature_label] = model_performance_summary

plt.figure(figsize=(6, 4))
plt.plot(model_performance_summary["stock_dependent: False, auto_select_feature: True"]["time_hist"], model_performance_summary["stock_dependent: False, auto_select_feature: True"]["mse_hist"], "-o", label="False, True")
plt.plot(model_performance_summary["stock_dependent: False, auto_select_feature: False"]["time_hist"], model_performance_summary["stock_dependent: False, auto_select_feature: False"]["mse_hist"], "-o", label="False, False")
plt.plot(model_performance_summary["stock_dependent: True, auto_select_feature: True"]["time_hist"], model_performance_summary["stock_dependent: True, auto_select_feature: True"]["mse_hist"], "-o", label="True, True")
plt.plot(model_performance_summary["stock_dependent: True, auto_select_feature: False"]["time_hist"], model_performance_summary["stock_dependent: True, auto_select_feature: False"]["mse_hist"], "-o", label="True, False")
plt.legend(loc="upper left", title="coef_stock_depend auto_select_feature", framealpha=0)
plt.ylim(0.08, 0.22)
plt.title("Baseline Model Performance")

#%%
current_feature_pool = feature_pool[0:2]
feature_label = "top2_feature"
model = model_lib.model_linear_regression(current_feature_pool, feature_label)
model.model_performance(is_stock_dependent=False, auto_select_feature=True)
model.model_performance(is_stock_dependent=False, auto_select_feature=False)
model.model_performance(is_stock_dependent=True, auto_select_feature=False)
model.model_performance(is_stock_dependent=True, auto_select_feature=True)
file_name = model.model_performance_file_name
with open(file_name, "rb") as f:
    model_performance_summary = pickle.load(f)
log[feature_label] = model_performance_summary

plt.figure(figsize=(6, 4))
plt.plot(model_performance_summary["stock_dependent: False, auto_select_feature: True"]["time_hist"], model_performance_summary["stock_dependent: False, auto_select_feature: True"]["mse_hist"], "-o", label="False, True")
plt.plot(model_performance_summary["stock_dependent: False, auto_select_feature: False"]["time_hist"], model_performance_summary["stock_dependent: False, auto_select_feature: False"]["mse_hist"], "-o", label="False, False")
plt.plot(model_performance_summary["stock_dependent: True, auto_select_feature: True"]["time_hist"], model_performance_summary["stock_dependent: True, auto_select_feature: True"]["mse_hist"], "-o", label="True, True")
plt.plot(model_performance_summary["stock_dependent: True, auto_select_feature: False"]["time_hist"], model_performance_summary["stock_dependent: True, auto_select_feature: False"]["mse_hist"], "-o", label="True, False")
plt.legend(loc="upper left", title="coef_stock_depend auto_select_feature", framealpha=0)
plt.ylim(0.08, 0.22)
plt.title("Top 2 Feature Model Performance")

#%%
current_feature_pool = feature_pool[0:5]
feature_label = "top5_feature"
model = model_lib.model_linear_regression(current_feature_pool, feature_label)
model.model_performance(is_stock_dependent=False, auto_select_feature=True)
model.model_performance(is_stock_dependent=False, auto_select_feature=False)
model.model_performance(is_stock_dependent=True, auto_select_feature=False)
model.model_performance(is_stock_dependent=True, auto_select_feature=True)
file_name = model.model_performance_file_name
with open(file_name, "rb") as f:
    model_performance_summary = pickle.load(f)
log[feature_label] = model_performance_summary

plt.figure(figsize=(6, 4))
plt.plot(model_performance_summary["stock_dependent: False, auto_select_feature: True"]["time_hist"], model_performance_summary["stock_dependent: False, auto_select_feature: True"]["mse_hist"], "-o", label="False, True")
plt.plot(model_performance_summary["stock_dependent: False, auto_select_feature: False"]["time_hist"], model_performance_summary["stock_dependent: False, auto_select_feature: False"]["mse_hist"], "-o", label="False, False")
plt.plot(model_performance_summary["stock_dependent: True, auto_select_feature: True"]["time_hist"], model_performance_summary["stock_dependent: True, auto_select_feature: True"]["mse_hist"], "-o", label="True, True")
plt.plot(model_performance_summary["stock_dependent: True, auto_select_feature: False"]["time_hist"], model_performance_summary["stock_dependent: True, auto_select_feature: False"]["mse_hist"], "-o", label="True, False")
plt.legend(loc="upper left", title="coef_stock_depend auto_select_feature", framealpha=0)
plt.ylim(0.08, 0.22)
plt.title("Top 5 Feature Model Performance")

#%%
current_feature_pool = feature_pool[0:14]
feature_label = "top14_feature"
model = model_lib.model_linear_regression(current_feature_pool, feature_label)
model.model_performance(is_stock_dependent=False, auto_select_feature=True)
model.model_performance(is_stock_dependent=False, auto_select_feature=False)
model.model_performance(is_stock_dependent=True, auto_select_feature=False)
model.model_performance(is_stock_dependent=True, auto_select_feature=True)
file_name = model.model_performance_file_name
with open(file_name, "rb") as f:
    model_performance_summary = pickle.load(f)
log[feature_label] = model_performance_summary

plt.figure(figsize=(6, 4))
plt.plot(model_performance_summary["stock_dependent: False, auto_select_feature: True"]["time_hist"], model_performance_summary["stock_dependent: False, auto_select_feature: True"]["mse_hist"], "-o", label="False, True")
plt.plot(model_performance_summary["stock_dependent: False, auto_select_feature: False"]["time_hist"], model_performance_summary["stock_dependent: False, auto_select_feature: False"]["mse_hist"], "-o", label="False, False")
plt.plot(model_performance_summary["stock_dependent: True, auto_select_feature: True"]["time_hist"], model_performance_summary["stock_dependent: True, auto_select_feature: True"]["mse_hist"], "-o", label="True, True")
plt.plot(model_performance_summary["stock_dependent: True, auto_select_feature: False"]["time_hist"], model_performance_summary["stock_dependent: True, auto_select_feature: False"]["mse_hist"], "-o", label="True, False")
plt.legend(loc="upper left", title="coef_stock_depend auto_select_feature", framealpha=0)
plt.ylim(0.08, 0.22)
plt.title("Top 14 Feature Model Performance")


#%%
k1 = ["baseline", "top2_feature", "top5_feature", "top14_feature"]
k2 = ["stock_dependent: False, auto_select_feature: True", 
      "stock_dependent: False, auto_select_feature: False", 
      "stock_dependent: True, auto_select_feature: True", 
      "stock_dependent: True, auto_select_feature: False"]
label = ["False, True", "False, False", "True, True", "True, False"]
ar = np.zeros((len(k1), len(k2)))
for i in range(len(k1)):
    for j in range(len(k2)):
        ar[i, j] = np.nanmean(log[k1[i]][k2[j]]["mse_hist"])

width = 0.2
x = np.arange(len(k1))
for i in range(len(k2)):
    plt.bar(x + i*width, ar[:, i], width=width, label=label[i])

plt.xticks(x + width * (len(k2) - 1) / 2, k1)
plt.xlabel("Feature Set")
plt.ylabel("MSE")
plt.legend(title="stock_dependent, auto_select_feature")
plt.tight_layout()
plt.show()

#%%
mse_summary_1 = []
mse_summary_2 = []

for i in range(len(k1)):
    mse_summary_1.append(np.nanmean(log[k1[i]]["stock_dependent: False, auto_select_feature: True"]["mse_hist"]))
    mse_summary_2.append(np.nanmean(log[k1[i]]["stock_dependent: False, auto_select_feature: False"]["mse_hist"]))

plt.plot(k1, mse_summary_1, "-o", label="False, True")
plt.plot(k1, mse_summary_2, "-o", label="False, False")
plt.legend(title="stock_dependent, auto_select_feature", framealpha=0)
plt.ylabel("MSE"); plt.xlabel("Feature Set")

# %%
def feature_selection_MSE(selected_feature):
    file_name = os.path.join(os.getcwd(), "../data/data_transformed.npz")
    data = np.load(file_name, allow_pickle=True)
    code_list = data["code_list"].tolist(); time_axis = data["time_axis"].tolist(); forward_beta = data["forward_beta"]
    N = len(code_list); T = len(time_axis)

    file_name = os.path.join(os.getcwd(), "../data/feature_dict.pkl")
    with open(file_name, "rb") as f:
        feature_dict = pickle.load(f)

    time_hist = []; mse_hist = []; model_result_hist = []
    t_idx = np.searchsorted(time_axis, datetime.datetime(2012, 1, 1))

    while True:
        t_train_start_idx = t_idx - 251
        t_train_end_idx = t_idx
        t_test_start_idx = t_train_end_idx + 1
        t_test_end_idx = min(t_test_start_idx + 124, T - 1)

        if t_test_start_idx >= T:
            break

        X = np.zeros((N, T, len(selected_feature))); X[:] = np.nan
        for i in range(len(selected_feature)):
            if selected_feature[i] in feature_dict:
                X[:, :, i] = feature_dict[selected_feature[i]]
            else:
                raise Exception("Feature {} not found.".format(selected_feature[i]))

        X_train = X[:, t_train_start_idx:(t_train_end_idx + 1), :]
        Y_train = forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)]
        data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
        data_train = data_train.swapaxes(0, 1).reshape((-1, len(selected_feature)+1)) # make T become the principal axis so that during train-valid data split, respect causality
        data_train = data_train[~np.isnan(data_train).any(axis=1), :]
        data_train = data_train[~np.isinf(data_train).any(axis=1), :]
        X_train = data_train[:, 0:len(selected_feature)]; Y_train = data_train[:, [len(selected_feature)]]

        X_test = X[:, t_test_start_idx:(t_test_end_idx + 1), :]
        Y_test = forward_beta[:, t_test_start_idx:(t_test_end_idx + 1)]
        data_test = np.concatenate((X_test, Y_test[:, :, np.newaxis]), axis=2)
        data_test = data_test.swapaxes(0, 1).reshape((-1, len(selected_feature)+1))
        data_test = data_test[~np.isnan(data_test).any(axis=1), :]
        data_test = data_test[~np.isinf(data_test).any(axis=1), :]
        X_test = data_test[:, 0:len(selected_feature)]; Y_test = data_test[:, [len(selected_feature)]]

        model = utils.linear_regression_vanilla(X_train, Y_train, X_columns=selected_feature, is_normalize=True)
        model.fit(is_output=False)
        if X_test.shape[0] > 0:
            time_hist.append(time_axis[t_train_start_idx])
            model_result_hist.append([model.ols.params[1:], model.ols.bse[1:], model.ols.pvalues[1:]])
            Y_pred = model.predict(X_test)
            mse_hist.append(np.mean(np.power(Y_pred - Y_test, 2)))
        t_idx += 125

    plt.figure(figsize=(6, 3))
    for i in range(len(model_result_hist)):
        plt.errorbar(selected_feature, model_result_hist[i][0], yerr=model_result_hist[i][1], label=datetime.datetime.strftime(time_hist[i], "%Y-%m-%d"), capsize=5)
    plt.ylabel("Coefficient"); plt.xlabel("Feature")
    plt.hlines(0, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], colors="black", linestyles="dashed")
    plt.legend(ncol=2)
    plt.xticks(rotation=45)

current_feature_pool = feature_pool[0:5]
feature_label = "top5_feature"
_ = feature_selection_MSE(current_feature_pool)






