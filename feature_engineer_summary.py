#%%
import os, sys, copy, scipy, datetime, tqdm, collections, itertools, pickle
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import subprocess

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

#%%
'''
file_name = os.path.join(os.getcwd(), "feature_vote.pkl")
if os.path.exists(file_name):
    os.remove(file_name)

scripts = [
    'return_based_features.py',
    'SPY_return_based_features.py',
    'volatility_based_features.py',
    'market_correlation_features.py',
    'volume_price_features.py'
]

for script in scripts:
    print(f'Running {script}...')
    try:
        result = subprocess.run(['python', script], capture_output=True, text=True)
        plt.close('all')
    except:
        raise Exception(f"Error running {script}")
'''

#%%
file_name_list = ["feature_vote_return.pkl", "feature_vote_SPY_return.pkl", "feature_vote_volatility.pkl", "feature_vote_market_correlation.pkl", "feature_vote_volume_price.pkl"]
feature_vote_summary = collections.defaultdict(int)
for file_name in file_name_list:
    with open(os.path.join(os.getcwd(), file_name), "rb") as f:
        feature_vote = pickle.load(f)
        for key, value in feature_vote.items():
            feature_vote_summary[key] += value

feature_vote_summary = sorted(feature_vote_summary.items(), key=lambda x: x[1], reverse=True)
plt.figure(figsize=(10, 4))
plt.bar([feature_vote_summary[i][0] for i in range(len(feature_vote_summary))], [feature_vote_summary[i][1] for i in range(len(feature_vote_summary))], color='grey')
selected_feature = [feature_vote_summary[i][0] for i in range(len(feature_vote_summary)) if feature_vote_summary[i][1] >= 10]
plt.bar(selected_feature, [feature_vote_summary[i][1] for i in range(len(feature_vote_summary)) if feature_vote_summary[i][0] in selected_feature], label="Tier 1")
selected_feature = [feature_vote_summary[i][0] for i in range(len(feature_vote_summary)) if feature_vote_summary[i][1] < 10 and feature_vote_summary[i][1] >= 4]
plt.bar(selected_feature, [feature_vote_summary[i][1] for i in range(len(feature_vote_summary)) if feature_vote_summary[i][0] in selected_feature], label="Tier 2")
selected_feature = [feature_vote_summary[i][0] for i in range(len(feature_vote_summary)) if feature_vote_summary[i][1] < 4 and feature_vote_summary[i][1] >=3]
plt.bar(selected_feature, [feature_vote_summary[i][1] for i in range(len(feature_vote_summary)) if feature_vote_summary[i][0] in selected_feature], label="Tier 3")
plt.xlabel("Feature")
plt.ylabel("Vote Count")
plt.legend()
plt.xticks(rotation=90)

#%%
#_ = utils.linear_model_diagnosis(selected_feature, year=2012, nonlinearity=True)
_ = utils.linear_model_diagnosis(selected_feature, year=2012, colinearity=True)
selected_feature.remove("beta*idiosyncratic_vol_20d")

#%%
_ = utils.linear_model_diagnosis(selected_feature, year=2012, colinearity=True)

#%%
_ = utils.linear_model_diagnosis(selected_feature, year=2012, nonlinearity=True)

#%%
_ = utils.linear_model_diagnosis(selected_feature, year=2012, outlier=True)

#%%
_ = utils.linear_model_diagnosis(selected_feature, year=2012, homoscedasticity=True)

#%%
_ = utils.linear_model_diagnosis(selected_feature, year=2012, residual_normality_independence=True)

#%%
_ = utils.linear_model_diagnosis(selected_feature, year=2012, feature_selection=True)

#%%
_ = utils.linear_model_diagnosis(selected_feature, year=2012, visualize=True)

# %%
selected_feature = [feature_vote_summary[i][0] for i in range(len(feature_vote_summary)) if feature_vote_summary[i][1] >= 3]
file_name = os.path.join(os.path.dirname(__file__), "feature_engineer_summary.pkl")
with open(file_name, "wb") as f:
    pickle.dump({"feature_vote": feature_vote_summary, "selected_feature": selected_feature}, f)

print("selected feature:")
print(selected_feature)

#%%





