#%%
import os, sys, copy, scipy, datetime, tqdm, collections, itertools, pickle
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    sys.path.append(os.path.join(os.getcwd(), '..'))
import utils

file_name = os.path.join(os.getcwd(), "../data/data_transformed.npz")
data = np.load(file_name, allow_pickle=True)
code_list = data["code_list"].tolist(); ticker_list = data["ticker_list"].tolist(); time_axis = data["time_axis"].tolist()
adjclose = data["adjclose"]; volume = data["volume"]; return_ = data["return_"]
beta = data["beta"]; forward_beta = data["forward_beta"]
SPY_adjclose = data["SPY_adjclose"]; SPY_volume = data["SPY_volume"]; SPY_return = data["SPY_return"]
N = len(code_list); T = len(time_axis)

file_name = os.path.join(os.getcwd(), "../data/feature_dict.pkl")
with open(file_name, "rb") as f:
    feature_dict = pickle.load(f)

feature_pool = ["beta", 
                "beta_20d", "beta_120d", "beta_mean_revert_20d", "idiosyncratic_vol_20d", "bear_beta_45d"]

mse_model_initial = utils.feature_selection_MSE(feature_pool)

#%%
#model = utils.linear_model_diagnosis(feature_pool, year=2012, colinearity=True)
#model = utils.linear_model_diagnosis(feature_pool, year=2013, colinearity=True)
#model = utils.linear_model_diagnosis(feature_pool, year=2014, colinearity=True)
#model = utils.linear_model_diagnosis(feature_pool, year=2015, colinearity=True)

#%%
#model = utils.linear_model_diagnosis(feature_pool, year=2012, nonlinearity=True)
#model = utils.linear_model_diagnosis(feature_pool, year=2013, nonlinearity=True)
#model = utils.linear_model_diagnosis(feature_pool, year=2014, nonlinearity=True)
#model = utils.linear_model_diagnosis(feature_pool, year=2015, nonlinearity=True)

#%% 
'''
Conclusion:
(1) Feature to remove:
    - None

(2) Feature to add (non-linearity):
    - beta***2 (AIC, BIC, CV_error)
    - beta*idiosyncratic_vol_20d (AIC, BIC, CV_error)
    - beta*bear_beta_45d (AIC, BIC, CV_error)
    - mean_revert_20d*idiosyncratic_vol_20d (AIC, BIC, CV_error)
'''

feature_dict["beta**2"] = np.power(beta, 2)
feature_dict["beta*idiosyncratic_vol_20d"] = np.multiply(beta, feature_dict["idiosyncratic_vol_20d"])
feature_dict["beta*bear_beta_45d"] = np.multiply(beta, feature_dict["bear_beta_45d"])
feature_dict["mean_revert_20d*idiosyncratic_vol_20d"] = np.multiply(feature_dict["beta_mean_revert_20d"], feature_dict["idiosyncratic_vol_20d"])

file_name = os.path.join(os.getcwd(), "../data/feature_dict.pkl")
with open(file_name, "wb") as f:
    pickle.dump(feature_dict, f)

feature_column = ["beta", 
                  "beta_20d", "beta_120d", "beta_mean_revert_20d", "idiosyncratic_vol_20d", "bear_beta_45d",
                  "beta**2", "beta*idiosyncratic_vol_20d", "beta*bear_beta_45d", "mean_revert_20d*idiosyncratic_vol_20d"]

mse_model_add_nonlinear = utils.feature_selection_MSE(feature_column)

feature_validation = []
model = utils.linear_model_diagnosis(feature_column, year=2012, feature_selection=True)
feature_validation.append(model.feature_selection_best_subset_summary[4][0])

model = utils.linear_model_diagnosis(feature_column, year=2013, feature_selection=True)
feature_validation.append(model.feature_selection_best_subset_summary[6][0])

model = utils.linear_model_diagnosis(feature_column, year=2014, feature_selection=True)
feature_validation.append(model.feature_selection_best_subset_summary[5][0])

model = utils.linear_model_diagnosis(feature_column, year=2015, feature_selection=True)
feature_validation.append(model.feature_selection_best_subset_summary[6][0])

#%%
feature_column = ["beta", 
                  "beta_20d", "beta_120d", "beta_mean_revert_20d", "idiosyncratic_vol_20d", "bear_beta_45d",
                  "beta**2", "beta*idiosyncratic_vol_20d", "beta*bear_beta_45d", "mean_revert_20d*idiosyncratic_vol_20d"]

mse_model_feature_selection = utils.feature_selection_MSE(feature_column)

#%%
plt.plot(["initial", "add_nonlinear", "feature_selection"], 
         [mse_model_initial, mse_model_add_nonlinear, mse_model_feature_selection], marker='o')
plt.title("MSE of different models")
plt.xlabel("Model")
plt.ylabel("MSE")

feature_vote = collections.defaultdict(int)
for i in [item for sublist in feature_validation for item in sublist]:
    feature_vote[feature_column[i]] += 1

file_name = os.path.join(os.getcwd(), "feature_vote_market_correlation.pkl")
with open(file_name, "wb") as f:
    pickle.dump(feature_vote, f)




# %%
