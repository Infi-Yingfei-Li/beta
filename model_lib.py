#%%
import os, sys, copy, scipy, datetime, tqdm, collections, itertools, pickle, json
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), '..'))
import utils

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional

from contextlib import redirect_stdout

#%%
class model_linear_regression:
    def __init__(self, feature, feature_label):
        self.feature = feature
        self.feature_label = feature_label

        self.t_update_freq = 125

        file_name = os.path.join(os.getcwd(), "../data", "data_transformed.npz")
        data = np.load(file_name, allow_pickle=True)
        self.code_list = data["code_list"].tolist(); self.time_axis = data["time_axis"].tolist()
        self.forward_beta = data["forward_beta"]
        self.N = len(self.code_list); self.T = len(self.time_axis)

        file_name = os.path.join(os.getcwd(), "../data", "feature_dict.pkl")
        with open(file_name, "rb") as f:
            self.feature_dict = pickle.load(f)

    def predict(self, X, t, auto_select_feature=True):
        '''
        params:
            X: np.ndarray, shape (N, p)
            t: datetime
        return:
            Y_pred: np.ndarray, shape (N, 1)
        '''
        if X.shape[1] != len(self.feature):
            raise Exception("Feature number mismatch. Expected {}, got {}.".format(len(self.feature), X.shape[1]))
        t_idx = np.searchsorted(self.time_axis, t)
        if auto_select_feature:
            if not hasattr(self, "optimal_feature_idx"):
                raise Exception("Evaluate model performance first to select hyperparameters.")
            for key in self.optimal_feature_idx:
                if t_idx >= key[0] and t_idx <= key[1]:
                    selected_feature_idx = self.optimal_feature_idx[key]
                    break
        else:
            selected_feature_idx = list(np.arange(len(self.feature)))
        
        t_train_start_idx = t_idx - 252
        t_train_end_idx = t_idx - 1
        data = np.zeros((self.N, self.T, len(self.feature))); data[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                data[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))

        X_train = data[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
        Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
        data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
        data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
        data_train = data_train[~np.isnan(data_train).any(axis=1), :]
        data_train = data_train[~np.isinf(data_train).any(axis=1), :]
        X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]
        X_train = X_train[:, selected_feature_idx]
        model = utils.linear_regression_vanilla(X_train, Y_train, X_columns=[self.feature[i] for i in selected_feature_idx], is_normalize=True)
        model.fit(is_output=False)
        Y_pred = model.predict(X).reshape(-1, 1)
        return Y_pred

    def model_performance(self, is_stock_dependent=False, auto_select_feature=True):
        print("Evaluating model performance - is_stock_dependent: {}, auto_select_feature: {}".format(is_stock_dependent, auto_select_feature))
        file_name = os.path.join(os.getcwd(), "results", "linear_regression_{}.pkl".format(self.feature_label))
        self.model_performance_file_name = file_name
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
            key = "stock_dependent: {}, auto_select_feature: {}".format(is_stock_dependent, auto_select_feature)
            if key in model_performance_summary:
                if (not is_stock_dependent) and auto_select_feature:
                    self.optimal_feature_idx = model_performance_summary["optimal_feature_idx"]
                return model_performance_summary[key]["time_hist"], model_performance_summary[key]["mse_hist"]

        time_hist = []; mse_hist = []; self.optimal_feature_idx = collections.defaultdict(list)
        t_idx = np.searchsorted(self.time_axis, datetime.datetime(2012, 1, 1))

        X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                X[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))
        
        plt.ioff()
        if is_stock_dependent:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                if t_test_start_idx >= self.T:
                    break
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                mse_temp = []
                for stock_idx in range(self.N):
                    X_train = X[stock_idx, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                    Y_train = self.forward_beta[stock_idx, t_train_start_idx:(t_train_end_idx + 1)].copy()
                    data_train = np.concatenate((X_train, Y_train[:, np.newaxis]), axis=1)
                    data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                    data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                    X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                    X_test = X[stock_idx, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                    Y_test = self.forward_beta[stock_idx, t_test_start_idx:(t_test_end_idx + 1)].copy()
                    data_test = np.concatenate((X_test, Y_test[:, np.newaxis]), axis=1)
                    data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                    data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                    X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                    if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                        continue

                    if auto_select_feature:
                        model = utils.linear_regression_vanilla(X_train, Y_train, X_columns=self.feature, is_normalize=True)
                        model.fit(is_output=False)
                        with open(os.devnull, "w") as f, redirect_stdout(f):
                            model.feature_selection_best_subset()
                            plt.close("all")
                        R2_oos = [model.feature_selection_best_subset_summary[i][4] for i in np.arange(1, model.p+1, 1)]
                        opt_feature_num = np.arange(1, model.p+1, 1)[np.argmax(R2_oos)]
                        selected_feature_idx = model.feature_selection_best_subset_summary[opt_feature_num][0]
                        selected_feature = [self.feature[i] for i in selected_feature_idx]
                        X_train = X_train[:, selected_feature_idx]
                        X_test = X_test[:, selected_feature_idx]
                    model = utils.linear_regression_vanilla(X_train, Y_train, X_columns=selected_feature if auto_select_feature else self.feature, is_normalize=True)
                    model.fit(is_output=False)
                    Y_pred = model.predict(X_test).reshape(-1, 1)
                    mse_temp.append(np.mean((Y_test - Y_pred) ** 2))

                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean(mse_temp))
                t_idx += self.t_update_freq

        else:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                if t_test_start_idx >= self.T:
                    break

                X_train = X[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
                data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
                data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                X_test = X[:, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                Y_test = self.forward_beta[:, t_test_start_idx:(t_test_end_idx + 1)].copy()
                data_test = np.concatenate((X_test, Y_test[:, :, np.newaxis]), axis=2)
                data_test = data_test.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                    break

                if auto_select_feature:
                    model = utils.linear_regression_vanilla(X_train, Y_train, X_columns=self.feature, is_normalize=True)
                    model.fit(is_output=False)
                    with open(os.devnull, "w") as f, redirect_stdout(f):
                        model.feature_selection_best_subset()
                        plt.close("all")
                    R2_oos = [model.feature_selection_best_subset_summary[i][4] for i in np.arange(1, model.p+1, 1)]
                    opt_feature_num = np.arange(1, model.p+1, 1)[np.argmax(R2_oos)]
                    selected_feature_idx = model.feature_selection_best_subset_summary[opt_feature_num][0]
                    self.optimal_feature_idx[(t_test_start_idx, t_test_end_idx)] = selected_feature_idx
                    selected_feature = [self.feature[i] for i in selected_feature_idx]
                    X_train = X_train[:, selected_feature_idx]
                    X_test = X_test[:, selected_feature_idx]

                model = utils.linear_regression_vanilla(X_train, Y_train, X_columns=selected_feature if auto_select_feature else self.feature, is_normalize=True)
                model.fit(is_output=False)
                Y_pred = model.predict(X_test).reshape(-1, 1)
                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean((Y_test - Y_pred) ** 2))
                t_idx += self.t_update_freq

        plt.ion()
        file_name = os.path.join(os.getcwd(), "results", "linear_regression_{}.pkl".format(self.feature_label))
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
        else:
            model_performance_summary = {}

        key = "stock_dependent: {}, auto_select_feature: {}".format(is_stock_dependent, auto_select_feature)
        if (not is_stock_dependent) and auto_select_feature:
            model_performance_summary["optimal_feature_idx"] = self.optimal_feature_idx
        model_performance_summary[key] = {"time_hist": time_hist, "mse_hist": mse_hist}
        with open(file_name, "wb") as f:
            pickle.dump(model_performance_summary, f)

        return time_hist, mse_hist

#%%
class model_ridge_lasso_regression:
    def __init__(self, feature, feature_label, regularization):
        self.feature = feature
        self.feature_label = feature_label
        self.regularization = regularization

        self.t_update_freq = 125

        file_name = os.path.join(os.getcwd(), "..", "data", "data_transformed.npz")
        data = np.load(file_name, allow_pickle=True)
        self.code_list = data["code_list"].tolist(); self.time_axis = data["time_axis"].tolist()
        self.forward_beta = data["forward_beta"]
        self.N = len(self.code_list); self.T = len(self.time_axis)

        file_name = os.path.join(os.getcwd(), "..", "data", "feature_dict.pkl")
        with open(file_name, "rb") as f:
            self.feature_dict = pickle.load(f)

    def predict(self, X, t):
        '''
        params:
            X: np.ndarray, shape (N, p)
            t: datetime
        return:
            Y_pred: np.ndarray, shape (N, 1)
        '''

        if not hasattr(self, "optimal_alpha_hist"):
            raise Exception("Evaluate model performance first to select hyperparameters.")
        if X.shape[1] != len(self.feature):
            raise Exception("Feature number mismatch. Expected {}, got {}.".format(len(self.feature), X.shape[1]))

        t_idx = np.searchsorted(self.time_axis, t)
        for key in self.optimal_alpha_hist:
            if t_idx >= key[0] and t_idx <= key[1]:
                alpha_opt = self.optimal_alpha_hist[key]
                break
        
        t_train_start_idx = t_idx - 252
        t_train_end_idx = t_idx - 1
        data = np.zeros((self.N, self.T, len(self.feature))); data[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                data[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))

        X_train = data[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
        Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
        data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
        data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
        data_train = data_train[~np.isnan(data_train).any(axis=1), :]
        data_train = data_train[~np.isinf(data_train).any(axis=1), :]
        X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]
        model = utils.ridge_lasso_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True, regularization=self.regularization)
        model.fit(alpha=alpha_opt)
        Y_pred = model.predict(X).reshape(-1, 1)
        return Y_pred

    def model_performance(self, is_stock_dependent=False):
        print("Evaluating model performance - is_stock_dependent: {}".format(is_stock_dependent))
        file_name = os.path.join(os.getcwd(), "results", "{}_regression_{}.pkl".format(self.regularization, self.feature_label))
        self.model_performance_file_name = file_name
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
            key = "stock_dependent: {}".format(is_stock_dependent)
            if key in model_performance_summary:
                if (not is_stock_dependent):
                    self.optimal_alpha_hist = model_performance_summary["optimal_alpha_hist"]
                return model_performance_summary[key]["time_hist"], model_performance_summary[key]["mse_hist"]

        time_hist = []; mse_hist = []; self.optimal_alpha_hist = dict()
        t_idx = np.searchsorted(self.time_axis, datetime.datetime(2012, 1, 1))

        X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                X[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))
        
        plt.ioff()
        if is_stock_dependent:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                if t_test_start_idx >= self.T:
                    break
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                mse_temp = []
                for stock_idx in range(self.N):
                    X_train = X[stock_idx, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                    Y_train = self.forward_beta[stock_idx, t_train_start_idx:(t_train_end_idx + 1)].copy()
                    data_train = np.concatenate((X_train, Y_train[:, np.newaxis]), axis=1)
                    data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                    data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                    X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                    X_test = X[stock_idx, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                    Y_test = self.forward_beta[stock_idx, t_test_start_idx:(t_test_end_idx + 1)].copy()
                    data_test = np.concatenate((X_test, Y_test[:, np.newaxis]), axis=1)
                    data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                    data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                    X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                    if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                        continue

                    model = utils.ridge_lasso_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True, regularization=self.regularization)
                    model.fit()
                    Y_pred = model.predict(X_test).reshape(-1, 1)
                    mse_temp.append(np.mean((Y_test - Y_pred) ** 2))

                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean(mse_temp))
                t_idx += self.t_update_freq

        else:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                if t_test_start_idx >= self.T:
                    break

                X_train = X[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
                data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
                data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                X_test = X[:, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                Y_test = self.forward_beta[:, t_test_start_idx:(t_test_end_idx + 1)].copy()
                data_test = np.concatenate((X_test, Y_test[:, :, np.newaxis]), axis=2)
                data_test = data_test.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                    break

                model = utils.ridge_lasso_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True, regularization=self.regularization)
                model.fit()
                Y_pred = model.predict(X_test).reshape((-1, 1))
                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean((Y_test - Y_pred) ** 2))
                self.optimal_alpha_hist[(t_test_start_idx, t_test_end_idx)] = model.alpha
                t_idx += self.t_update_freq

        plt.ion()
        file_name = os.path.join(os.getcwd(), "results", "{}_regression_{}.pkl".format(self.regularization, self.feature_label))
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
        else:
            model_performance_summary = {}

        key = "stock_dependent: {}".format(is_stock_dependent)
        if not is_stock_dependent:
            model_performance_summary["optimal_alpha_hist"] = self.optimal_alpha_hist
        model_performance_summary[key] = {"time_hist": time_hist, "mse_hist": mse_hist}
        with open(file_name, "wb") as f:
            pickle.dump(model_performance_summary, f)

        return time_hist, mse_hist

#%%
class model_principal_component_regression:
    def __init__(self, feature, feature_label):
        self.feature = feature
        self.feature_label = feature_label

        self.t_update_freq = 125

        file_name = os.path.join(os.getcwd(), "..", "data", "data_transformed.npz")
        data = np.load(file_name, allow_pickle=True)
        self.code_list = data["code_list"].tolist(); self.time_axis = data["time_axis"].tolist()
        self.forward_beta = data["forward_beta"]
        self.N = len(self.code_list); self.T = len(self.time_axis)

        file_name = os.path.join(os.getcwd(), "..", "data", "feature_dict.pkl")
        with open(file_name, "rb") as f:
            self.feature_dict = pickle.load(f)

    def predict(self, X, t):
        '''
        params:
            X: np.ndarray, shape (N, p)
            t: datetime
        return:
            Y_pred: np.ndarray, shape (N, 1)
        '''
        if not hasattr(self, "optimal_factor_num_hist"):
            raise Exception("Evaluate model performance first to select hyperparameters.")
        if X.shape[1] != len(self.feature):
            raise Exception("Feature number mismatch. Expected {}, got {}.".format(len(self.feature), X.shape[1]))

        t_idx = np.searchsorted(self.time_axis, t)
        for key in self.optimal_factor_num_hist:
            if t_idx >= key[0] and t_idx <= key[1]:
                factor_num_opt = self.optimal_factor_num_hist[key]
                break
        
        t_train_start_idx = t_idx - 252
        t_train_end_idx = t_idx - 1
        data = np.zeros((self.N, self.T, len(self.feature))); data[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                data[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))

        X_train = data[:, t_train_start_idx:(t_train_end_idx + 1), :]
        Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
        data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
        data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
        data_train = data_train[~np.isnan(data_train).any(axis=1), :]
        data_train = data_train[~np.isinf(data_train).any(axis=1), :]
        X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]
        model = utils.principal_component_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True)
        model.fit(factor_num=factor_num_opt, is_plot=False)
        Y_pred = model.predict(X).reshape(-1, 1)
        return Y_pred

    def model_performance(self, is_stock_dependent=False):
        print("Evaluating model performance - is_stock_dependent: {}".format(is_stock_dependent))
        file_name = os.path.join(os.getcwd(), "results", "principal_component_regression_{}.pkl".format(self.feature_label))
        self.model_performance_file_name = file_name
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
            key = "stock_dependent: {}".format(is_stock_dependent)
            if key in model_performance_summary:
                if (not is_stock_dependent):
                    self.optimal_factor_num_hist = model_performance_summary["optimal_factor_num_hist"]
                return model_performance_summary[key]["time_hist"], model_performance_summary[key]["mse_hist"]

        time_hist = []; mse_hist = []; self.optimal_factor_num_hist = dict()
        t_idx = np.searchsorted(self.time_axis, datetime.datetime(2012, 1, 1))

        X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                X[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))
        
        plt.ioff()
        if is_stock_dependent:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                if t_test_start_idx >= self.T:
                    break
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                mse_temp = []
                for stock_idx in range(self.N):
                    X_train = X[stock_idx, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                    Y_train = self.forward_beta[stock_idx, t_train_start_idx:(t_train_end_idx + 1)].copy()
                    data_train = np.concatenate((X_train, Y_train[:, np.newaxis]), axis=1)
                    data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                    data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                    X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                    X_test = X[stock_idx, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                    Y_test = self.forward_beta[stock_idx, t_test_start_idx:(t_test_end_idx + 1)].copy()
                    data_test = np.concatenate((X_test, Y_test[:, np.newaxis]), axis=1)
                    data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                    data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                    X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                    if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                        continue

                    model = utils.principal_component_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True)
                    model.fit(is_plot=False)
                    Y_pred = model.predict(X_test).reshape(-1, 1)
                    mse_temp.append(np.mean((Y_test - Y_pred) ** 2))

                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean(mse_temp))
                t_idx += self.t_update_freq

        else:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                if t_test_start_idx >= self.T:
                    break

                X_train = X[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
                data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
                data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                X_test = X[:, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                Y_test = self.forward_beta[:, t_test_start_idx:(t_test_end_idx + 1)].copy()
                data_test = np.concatenate((X_test, Y_test[:, :, np.newaxis]), axis=2)
                data_test = data_test.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                    break

                model = utils.principal_component_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True)
                model.fit(is_plot=False)
                Y_pred = model.predict(X_test).reshape((-1, 1))
                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean((Y_test - Y_pred) ** 2))
                self.optimal_factor_num_hist[(t_test_start_idx, t_test_end_idx)] = model.optimal_factor_number
                t_idx += self.t_update_freq

        plt.ion()
        file_name = os.path.join(os.getcwd(), "results", "principal_component_regression_{}.pkl".format(self.feature_label))
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
        else:
            model_performance_summary = {}

        key = "stock_dependent: {}".format(is_stock_dependent)
        if not is_stock_dependent:
            model_performance_summary["optimal_factor_num_hist"] = self.optimal_factor_num_hist
        model_performance_summary[key] = {"time_hist": time_hist, "mse_hist": mse_hist}
        with open(file_name, "wb") as f:
            pickle.dump(model_performance_summary, f)

        return time_hist, mse_hist

#%%
class model_least_angle_regression:
    def __init__(self, feature, feature_label):
        self.feature = feature
        self.feature_label = feature_label

        self.t_update_freq = 125

        file_name = os.path.join(os.getcwd(), "..", "data", "data_transformed.npz")
        data = np.load(file_name, allow_pickle=True)
        self.code_list = data["code_list"].tolist(); self.time_axis = data["time_axis"].tolist()
        self.forward_beta = data["forward_beta"]
        self.N = len(self.code_list); self.T = len(self.time_axis)

        file_name = os.path.join(os.getcwd(), "..", "data", "feature_dict.pkl")
        with open(file_name, "rb") as f:
            self.feature_dict = pickle.load(f)

    def predict(self, X, t):
        '''
        params:
            X: np.ndarray, shape (N, p)
            t: datetime
        return:
            Y_pred: np.ndarray, shape (N, 1)
        '''
        if not hasattr(self, "optimal_factor_num_hist"):
            raise Exception("Evaluate model performance first to select hyperparameters.")
        if X.shape[1] != len(self.feature):
            raise Exception("Feature number mismatch. Expected {}, got {}.".format(len(self.feature), X.shape[1]))

        t_idx = np.searchsorted(self.time_axis, t)
        for key in self.optimal_factor_num_hist:
            if t_idx >= key[0] and t_idx <= key[1]:
                factor_num_opt = self.optimal_factor_num_hist[key]
                break
        
        t_train_start_idx = t_idx - 252
        t_train_end_idx = t_idx - 1
        data = np.zeros((self.N, self.T, len(self.feature))); data[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                data[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))

        X_train = data[:, t_train_start_idx:(t_train_end_idx + 1), :]
        Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
        data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
        data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
        data_train = data_train[~np.isnan(data_train).any(axis=1), :]
        data_train = data_train[~np.isinf(data_train).any(axis=1), :]
        X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]
        model = utils.least_angle_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True)
        model.fit(factor_num=factor_num_opt)
        Y_pred = model.predict(X).reshape(-1, 1)
        return Y_pred

    def model_performance(self, is_stock_dependent=False):
        print("Evaluating model performance - is_stock_dependent: {}".format(is_stock_dependent))
        file_name = os.path.join(os.getcwd(), "results", "least_angle_regression_{}.pkl".format(self.feature_label))
        self.model_performance_file_name = file_name
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
            key = "stock_dependent: {}".format(is_stock_dependent)
            if key in model_performance_summary:
                if (not is_stock_dependent):
                    self.optimal_factor_num_hist = model_performance_summary["optimal_factor_num_hist"]
                return model_performance_summary[key]["time_hist"], model_performance_summary[key]["mse_hist"]

        time_hist = []; mse_hist = []; self.optimal_factor_num_hist = dict()
        t_idx = np.searchsorted(self.time_axis, datetime.datetime(2012, 1, 1))

        X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                X[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))
        
        plt.ioff()
        if is_stock_dependent:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                if t_test_start_idx >= self.T:
                    break
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                mse_temp = []
                for stock_idx in range(self.N):
                    X_train = X[stock_idx, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                    Y_train = self.forward_beta[stock_idx, t_train_start_idx:(t_train_end_idx + 1)].copy()
                    data_train = np.concatenate((X_train, Y_train[:, np.newaxis]), axis=1)
                    data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                    data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                    X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                    X_test = X[stock_idx, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                    Y_test = self.forward_beta[stock_idx, t_test_start_idx:(t_test_end_idx + 1)].copy()
                    data_test = np.concatenate((X_test, Y_test[:, np.newaxis]), axis=1)
                    data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                    data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                    X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                    if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                        continue

                    model = utils.least_angle_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True)
                    model.fit()
                    Y_pred = model.predict(X_test).reshape(-1, 1)
                    mse_temp.append(np.mean((Y_test - Y_pred) ** 2))

                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean(mse_temp))
                t_idx += self.t_update_freq

        else:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                if t_test_start_idx >= self.T:
                    break

                X_train = X[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
                data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
                data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                X_test = X[:, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                Y_test = self.forward_beta[:, t_test_start_idx:(t_test_end_idx + 1)].copy()
                data_test = np.concatenate((X_test, Y_test[:, :, np.newaxis]), axis=2)
                data_test = data_test.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                    break

                model = utils.least_angle_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True)
                model.fit()
                Y_pred = model.predict(X_test).reshape((-1, 1))
                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean((Y_test - Y_pred) ** 2))
                self.optimal_factor_num_hist[(t_test_start_idx, t_test_end_idx)] = model.optimal_factor_number
                t_idx += self.t_update_freq

        plt.ion()
        file_name = os.path.join(os.getcwd(), "results", "least_angle_regression_{}.pkl".format(self.feature_label))
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
        else:
            model_performance_summary = {}

        key = "stock_dependent: {}".format(is_stock_dependent)
        if not is_stock_dependent:
            model_performance_summary["optimal_factor_num_hist"] = self.optimal_factor_num_hist
        model_performance_summary[key] = {"time_hist": time_hist, "mse_hist": mse_hist}
        with open(file_name, "wb") as f:
            pickle.dump(model_performance_summary, f)

        return time_hist, mse_hist



#%%
class model_partial_least_square_regression:
    def __init__(self, feature, feature_label):
        self.feature = feature
        self.feature_label = feature_label

        self.t_update_freq = 125

        file_name = os.path.join(os.getcwd(), "..", "data", "data_transformed.npz")
        data = np.load(file_name, allow_pickle=True)
        self.code_list = data["code_list"].tolist(); self.time_axis = data["time_axis"].tolist()
        self.forward_beta = data["forward_beta"]
        self.N = len(self.code_list); self.T = len(self.time_axis)

        file_name = os.path.join(os.getcwd(), "..", "data", "feature_dict.pkl")
        with open(file_name, "rb") as f:
            self.feature_dict = pickle.load(f)

    def predict(self, X, t):
        '''
        params:
            X: np.ndarray, shape (N, p)
            t: datetime
        return:
            Y_pred: np.ndarray, shape (N, 1)
        '''
        if not hasattr(self, "optimal_factor_num_hist"):
            raise Exception("Evaluate model performance first to select hyperparameters.")
        if X.shape[1] != len(self.feature):
            raise Exception("Feature number mismatch. Expected {}, got {}.".format(len(self.feature), X.shape[1]))

        t_idx = np.searchsorted(self.time_axis, t)
        for key in self.optimal_factor_num_hist:
            if t_idx >= key[0] and t_idx <= key[1]:
                factor_num_opt = self.optimal_factor_num_hist[key]
                break
        
        t_train_start_idx = t_idx - 252
        t_train_end_idx = t_idx - 1
        data = np.zeros((self.N, self.T, len(self.feature))); data[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                data[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))

        X_train = data[:, t_train_start_idx:(t_train_end_idx + 1), :]
        Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
        data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
        data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
        data_train = data_train[~np.isnan(data_train).any(axis=1), :]
        data_train = data_train[~np.isinf(data_train).any(axis=1), :]
        X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]
        model = utils.partial_least_square_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True)
        model.fit(factor_num=factor_num_opt, is_plot=False)
        Y_pred = model.predict(X).reshape(-1, 1)
        return Y_pred

    def model_performance(self, is_stock_dependent=False):
        print("Evaluating model performance - is_stock_dependent: {}".format(is_stock_dependent))
        file_name = os.path.join(os.getcwd(), "results", "partial_least_square_regression_{}.pkl".format(self.feature_label))
        self.model_performance_file_name = file_name
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
            key = "stock_dependent: {}".format(is_stock_dependent)
            if key in model_performance_summary:
                if (not is_stock_dependent):
                    self.optimal_factor_num_hist = model_performance_summary["optimal_factor_num_hist"]
                return model_performance_summary[key]["time_hist"], model_performance_summary[key]["mse_hist"]

        time_hist = []; mse_hist = []; self.optimal_factor_num_hist = dict()
        t_idx = np.searchsorted(self.time_axis, datetime.datetime(2012, 1, 1))

        X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                X[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))
        
        plt.ioff()
        if is_stock_dependent:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                if t_test_start_idx >= self.T:
                    break
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                mse_temp = []
                for stock_idx in range(self.N):
                    X_train = X[stock_idx, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                    Y_train = self.forward_beta[stock_idx, t_train_start_idx:(t_train_end_idx + 1)].copy()
                    data_train = np.concatenate((X_train, Y_train[:, np.newaxis]), axis=1)
                    data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                    data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                    X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                    X_test = X[stock_idx, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                    Y_test = self.forward_beta[stock_idx, t_test_start_idx:(t_test_end_idx + 1)].copy()
                    data_test = np.concatenate((X_test, Y_test[:, np.newaxis]), axis=1)
                    data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                    data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                    X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                    if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                        continue

                    model = utils.partial_least_square_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True)
                    model.fit(is_plot=False)
                    Y_pred = model.predict(X_test).reshape(-1, 1)
                    mse_temp.append(np.mean((Y_test - Y_pred) ** 2))

                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean(mse_temp))
                t_idx += self.t_update_freq

        else:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                if t_test_start_idx >= self.T:
                    break

                X_train = X[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
                data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
                data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                X_test = X[:, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                Y_test = self.forward_beta[:, t_test_start_idx:(t_test_end_idx + 1)].copy()
                data_test = np.concatenate((X_test, Y_test[:, :, np.newaxis]), axis=2)
                data_test = data_test.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                    break

                model = utils.partial_least_square_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True)
                model.fit(is_plot=False)
                Y_pred = model.predict(X_test).reshape((-1, 1))
                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean((Y_test - Y_pred) ** 2))
                self.optimal_factor_num_hist[(t_test_start_idx, t_test_end_idx)] = model.optimal_factor_number
                t_idx += self.t_update_freq

        plt.ion()
        file_name = os.path.join(os.getcwd(), "results", "partial_least_square_regression_{}.pkl".format(self.feature_label))
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
        else:
            model_performance_summary = {}

        key = "stock_dependent: {}".format(is_stock_dependent)
        if not is_stock_dependent:
            model_performance_summary["optimal_factor_num_hist"] = self.optimal_factor_num_hist
        model_performance_summary[key] = {"time_hist": time_hist, "mse_hist": mse_hist}
        with open(file_name, "wb") as f:
            pickle.dump(model_performance_summary, f)

        return time_hist, mse_hist


# %%
class model_tree_based_regression:
    def __init__(self, feature, feature_label, is_gradient_boost):
        self.feature = feature
        self.feature_label = feature_label
        self.is_gradient_boost = is_gradient_boost

        self.t_update_freq = 125
        if self.is_gradient_boost:
            self.params_dict = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}
        else:
            self.params_dict = {"n_estimators": 100, "max_depth": 5}

        file_name = os.path.join(os.getcwd(), "..", "data", "data_transformed.npz")
        data = np.load(file_name, allow_pickle=True)
        self.code_list = data["code_list"].tolist(); self.time_axis = data["time_axis"].tolist()
        self.forward_beta = data["forward_beta"]
        self.N = len(self.code_list); self.T = len(self.time_axis)

        file_name = os.path.join(os.getcwd(), "..", "data", "feature_dict.pkl")
        with open(file_name, "rb") as f:
            self.feature_dict = pickle.load(f)

    def predict(self, X, t):
        '''
        params:
            X: np.ndarray, shape (N, p)
            t: datetime
        return:
            Y_pred: np.ndarray, shape (N, 1)
        '''
        if X.shape[1] != len(self.feature):
            raise Exception("Feature number mismatch. Expected {}, got {}.".format(len(self.feature), X.shape[1]))

        file_name = os.path.join(os.getcwd(), "results", "{}random_forest_regression_{}.pkl".format("gb_" if self.is_gradient_boost else "",self.feature_label))
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
            model_estimated_params = model_performance_summary["model_estimated_params"]
        else:
            raise Exception("Model performance file not found. Please evaluate model performance first.")

        t_idx = np.searchsorted(self.time_axis, t)
        for key in model_estimated_params:
            if t_idx >= key[0] and t_idx <= key[1]:
                model = model_estimated_params[key]
                break
        Y_pred = model.predict(X).reshape(-1, 1)
        return Y_pred

    def model_performance(self, is_stock_dependent=False):
        print("Evaluating model performance - is_stock_dependent: {}".format(is_stock_dependent))
        file_name = os.path.join(os.getcwd(), "results", "{}random_forest_regression_{}.pkl".format("gb_" if self.is_gradient_boost else "",self.feature_label))
        self.model_performance_file_name = file_name
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
            key = "stock_dependent: {}".format(is_stock_dependent)
            if key in model_performance_summary:
                return model_performance_summary[key]["time_hist"], model_performance_summary[key]["mse_hist"]

        time_hist = []; mse_hist = []
        t_idx = np.searchsorted(self.time_axis, datetime.datetime(2012, 1, 1))

        X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                X[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))
        
        plt.ioff()
        if is_stock_dependent:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                if t_test_start_idx >= self.T:
                    break
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                mse_temp = []
                for stock_idx in range(self.N):
                    X_train = X[stock_idx, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                    Y_train = self.forward_beta[stock_idx, t_train_start_idx:(t_train_end_idx + 1)].copy()
                    data_train = np.concatenate((X_train, Y_train[:, np.newaxis]), axis=1)
                    data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                    data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                    X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                    X_test = X[stock_idx, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                    Y_test = self.forward_beta[stock_idx, t_test_start_idx:(t_test_end_idx + 1)].copy()
                    data_test = np.concatenate((X_test, Y_test[:, np.newaxis]), axis=1)
                    data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                    data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                    X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                    if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                        continue

                    model = utils.random_forest_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True, is_gradient_boost=self.is_gradient_boost)
                    model.fit(params_dict=self.params_dict)
                    Y_pred = model.predict(X_test).reshape(-1, 1)
                    mse_temp.append(np.mean((Y_test - Y_pred) ** 2))

                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean(mse_temp))
                t_idx += self.t_update_freq

        else:
            model_estimated_params = {}
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                if t_test_start_idx >= self.T:
                    break

                X_train = X[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
                data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
                data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                X_test = X[:, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                Y_test = self.forward_beta[:, t_test_start_idx:(t_test_end_idx + 1)].copy()
                data_test = np.concatenate((X_test, Y_test[:, :, np.newaxis]), axis=2)
                data_test = data_test.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                    break

                model = utils.random_forest_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True, is_gradient_boost=self.is_gradient_boost)
                model.fit(params_dict=self.params_dict)
                Y_pred = model.predict(X_test).reshape((-1, 1))
                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean((Y_test - Y_pred) ** 2))
                model_estimated_params[(t_test_start_idx, t_test_end_idx)] = model
                t_idx += self.t_update_freq
        plt.ion()

        file_name = os.path.join(os.getcwd(), "results", "{}random_forest_regression_{}.pkl".format("gb_" if self.is_gradient_boost else "",self.feature_label))
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
        else:
            model_performance_summary = {}

        key = "stock_dependent: {}".format(is_stock_dependent)
        model_performance_summary[key] = {"time_hist": time_hist, "mse_hist": mse_hist}
        if not is_stock_dependent:
            model_performance_summary["model_estimated_params"] = model_estimated_params
        with open(file_name, "wb") as f:
            pickle.dump(model_performance_summary, f)

        return time_hist, mse_hist

#%%
class model_neural_networks:
    def __init__(self, feature, feature_label, nn_type="cnn_transformer"):
        self.feature = feature
        self.feature_label = feature_label
        self.nn_type = nn_type

        self.t_update_freq = 125
        self.CNN_transformer_config = {"looking_back_window": 24,
                                    "epoch_max": 100,
                                    "learning_rate": 1e-3,
                                    "CNN_input_channels": len(self.feature),
                                    "CNN_output_channels": 2*len(self.feature),
                                    "CNN_kernel_size": 2,
                                    "CNN_drop_out_rate": 0.25,
                                    "transformer_input_channels": 2*len(self.feature),
                                    "transformer_hidden_channels": 4*len(self.feature),
                                    "transformer_output_channels": 2*len(self.feature),
                                    "transformer_head_num": len(self.feature),
                                    "transformer_drop_out_rate": 0.25
                                }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.neural_networks_dict = dict()

        file_name = os.path.join(os.getcwd(), "..", "data", "data_transformed.npz")
        data = np.load(file_name, allow_pickle=True)
        self.code_list = data["code_list"].tolist(); self.time_axis = data["time_axis"].tolist()
        self.forward_beta = data["forward_beta"]
        self.N = len(self.code_list); self.T = len(self.time_axis)

        file_name = os.path.join(os.getcwd(), "..", "data", "feature_dict.pkl")
        with open(file_name, "rb") as f:
            self.feature_dict = pickle.load(f)

    def predict(self, X, t):
        '''
        params:
            X: np.ndarray, shape (N, T, p)
            t: datetime
        return:
            Y_pred: np.ndarray, shape (N, 1)
        '''
        if X.shape[2] != len(self.feature):
            raise Exception("Feature number mismatch. Expected {}, got {}.".format(len(self.feature), X.shape[2]))
        t_idx = np.searchsorted(self.time_axis, t)
        for key in self.neural_networks_dict:
            if t_idx >= key[0] and t_idx <= key[1]:
                t_train_start_idx, t_train_end_idx, t_test_start_idx, t_test_end_idx = self.neural_networks_dict[key]
                break
        self.train(t_train_start_idx, t_train_end_idx, t_test_start_idx, t_test_end_idx)
        self.model.eval()
        X = torch.from_numpy(X).to(torch.float32).to(self.device)
        X = X.permute(0, 2, 1)
        Y_pred = self.model.forward(X)
        return Y_pred.cpu().detach().numpy().reshape(-1, 1)

    def train(self, t_train_start_idx, t_train_end_idx, t_test_start_idx, t_test_end_idx):
        X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                X[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))
        X = torch.from_numpy(X).to(torch.float32).to(self.device)
        Y = torch.from_numpy(self.forward_beta).to(torch.float32).to(self.device)

        self.model = utils.CNN_transformer(CNN_input_channels = self.CNN_transformer_config["CNN_input_channels"], 
                                    CNN_output_channels = self.CNN_transformer_config["CNN_output_channels"], 
                                    CNN_kernel_size = self.CNN_transformer_config["CNN_kernel_size"], 
                                    CNN_drop_out_rate = self.CNN_transformer_config["CNN_drop_out_rate"], 
                                    transformer_input_channels = self.CNN_transformer_config["transformer_input_channels"], 
                                    transformer_hidden_channels = self.CNN_transformer_config["transformer_hidden_channels"], 
                                    transformer_output_channels = self.CNN_transformer_config["transformer_output_channels"], 
                                    transformer_head_num = self.CNN_transformer_config["transformer_head_num"], 
                                    transformer_drop_out_rate = self.CNN_transformer_config["transformer_drop_out_rate"])

        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.CNN_transformer_config["learning_rate"])

        network_file_name = os.path.join(os.getcwd(), "results", "NN_params_{}_t_test_idx_{}_{}.pt".format(self.feature_label, t_test_start_idx, t_test_end_idx))
        if os.path.exists(network_file_name):
            checkpoint = torch.load(network_file_name)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
        else:
            epoch = 0

        epoch_max = 10000
        self.model.train()
        for _ in tqdm.tqdm(range(epoch_max - epoch)):
                t_idx_select = np.random.uniform(t_train_start_idx, t_train_end_idx, size=1).astype(int)[0]
                X_train = X[:, (t_idx_select - self.CNN_transformer_config["looking_back_window"] + 1):(t_idx_select + 1), :]
                Y_train = Y[:, (t_idx_select - self.CNN_transformer_config["looking_back_window"] + 1):(t_idx_select + 1)]
                stock_idx = torch.isnan(X_train).any(axis=2).any(axis=1)
                X_train = X_train[~stock_idx, :, :]; Y_train = Y_train[~stock_idx, :]
                stock_idx = torch.isnan(Y_train).any(axis=1)
                X_train = X_train[~stock_idx, :, :]; Y_train = Y_train[~stock_idx, :]
                X_train = X_train.permute(0, 2, 1)
                target = torch.mean((Y_train - self.model.forward(X_train)) ** 2)
                optimizer.zero_grad()
                target.backward()
                optimizer.step()

        torch.save({"epoch": epoch_max,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, network_file_name)

        self.model.eval()
        self.neural_networks_dict[(t_test_start_idx, t_test_end_idx)] = (t_train_start_idx, t_train_end_idx, t_test_start_idx, t_test_end_idx)

    def model_performance(self):
        print("Evaluating model performance")
        file_name = os.path.join(os.getcwd(), "results", "neural_networks.pkl")
        time_hist = []; mse_hist = []; self.model_dict = dict()
        t_idx = np.searchsorted(self.time_axis, datetime.datetime(2012, 1, 1))
        plt.ioff()
        while True:
            t_train_start_idx = t_idx - 251
            t_train_end_idx = t_idx
            t_test_start_idx = t_train_end_idx + 1
            t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
            if t_test_start_idx >= self.T:
                break
            print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))
            self.train(t_train_start_idx, t_train_end_idx, t_test_start_idx, t_test_end_idx)
            self.model.eval()

            X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
            for i in range(len(self.feature)):
                if self.feature[i] in self.feature_dict:
                    X[:, :, i] = self.feature_dict[self.feature[i]]
                else:
                    raise Exception("Feature {} not found.".format(self.feature[i]))
            X = torch.from_numpy(X).to(torch.float32).to(self.device)
            Y = torch.from_numpy(self.forward_beta).to(torch.float32).to(self.device)

            mse_temp = []
            for t_idx_temp in range(t_test_start_idx, t_test_end_idx + 1):
                X_test = X[:, (t_idx_temp - self.CNN_transformer_config["looking_back_window"] + 1):(t_idx_temp + 1), :]
                Y_test = Y[:, (t_idx_temp - self.CNN_transformer_config["looking_back_window"] + 1):(t_idx_temp + 1)]
                stock_idx = torch.isnan(X_test).any(axis=2).any(axis=1)
                X_test = X_test[~stock_idx, :, :]; Y_test = Y_test[~stock_idx, :]
                stock_idx = torch.isnan(Y_test).any(axis=1)
                X_test = X_test[~stock_idx, :, :]; Y_test = Y_test[~stock_idx, :]
                if X_test.shape[0] == 0:
                    continue
                X_test = X_test.permute(0, 2, 1)
                Y_pred = self.model.forward(X_test)
                mse_temp.append(float(torch.mean((Y_test - Y_pred) ** 2).detach().cpu().numpy()))

            time_hist.append(self.time_axis[t_test_start_idx])
            mse_hist.append(np.mean(mse_temp))
            t_idx += self.t_update_freq

        plt.ion()
        return time_hist, mse_hist

#%%
class model_support_vector_regression:
    def __init__(self, feature, feature_label, kernel="rbf"):
        self.feature = feature
        self.feature_label = feature_label
        self.kernel = kernel

        self.t_update_freq = 125

        file_name = os.path.join(os.getcwd(), "..", "data", "data_transformed.npz")
        data = np.load(file_name, allow_pickle=True)
        self.code_list = data["code_list"].tolist(); self.time_axis = data["time_axis"].tolist()
        self.forward_beta = data["forward_beta"]
        self.N = len(self.code_list); self.T = len(self.time_axis)

        file_name = os.path.join(os.getcwd(), "data", "feature_dict.pkl")
        with open(file_name, "rb") as f:
            self.feature_dict = pickle.load(f)

    def predict(self, X, t):
        '''
        params:
            X: np.ndarray, shape (N, p)
            t: datetime
        return:
            Y_pred: np.ndarray, shape (N, 1)
        '''
        if not hasattr(self, "optimal_factor_num_hist"):
            raise Exception("Evaluate model performance first to select hyperparameters.")
        if X.shape[1] != len(self.feature):
            raise Exception("Feature number mismatch. Expected {}, got {}.".format(len(self.feature), X.shape[1]))

        t_idx = np.searchsorted(self.time_axis, t)
        for key in self.optimal_factor_num_hist:
            if t_idx >= key[0] and t_idx <= key[1]:
                factor_num_opt = self.optimal_factor_num_hist[key]
                break
        
        t_train_start_idx = t_idx - 252
        t_train_end_idx = t_idx - 1
        X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                X[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))

        X_train = X[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
        Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
        data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
        data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
        data_train = data_train[~np.isnan(data_train).any(axis=1), :]
        data_train = data_train[~np.isinf(data_train).any(axis=1), :]
        X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]
        model = utils.support_vector_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True, kernel=self.kernel)
        model.fit(C=factor_num_opt)
        Y_pred = model.predict(X).reshape(-1, 1)
        return Y_pred

    def model_performance(self, is_stock_dependent=False):
        print("Evaluating model performance - is_stock_dependent: {}".format(is_stock_dependent))
        file_name = os.path.join(os.getcwd(), "results", "support_vector_regression_{}.pkl".format(self.feature_label))
        self.model_performance_file_name = file_name
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
            key = "stock_dependent: {}".format(is_stock_dependent)
            if key in model_performance_summary:
                if (not is_stock_dependent):
                    self.optimal_factor_num_hist = model_performance_summary["optimal_factor_num_hist"]
                return model_performance_summary[key]["time_hist"], model_performance_summary[key]["mse_hist"]

        time_hist = []; mse_hist = []; self.optimal_factor_num_hist = dict()
        t_idx = np.searchsorted(self.time_axis, datetime.datetime(2012, 1, 1))

        X = np.zeros((self.N, self.T, len(self.feature))); X[:] = np.nan
        for i in range(len(self.feature)):
            if self.feature[i] in self.feature_dict:
                X[:, :, i] = self.feature_dict[self.feature[i]]
            else:
                raise Exception("Feature {} not found.".format(self.feature[i]))
        
        plt.ioff()
        if is_stock_dependent:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                if t_test_start_idx >= self.T:
                    break
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                mse_temp = []
                for stock_idx in range(self.N):
                    X_train = X[stock_idx, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                    Y_train = self.forward_beta[stock_idx, t_train_start_idx:(t_train_end_idx + 1)].copy()
                    data_train = np.concatenate((X_train, Y_train[:, np.newaxis]), axis=1)
                    data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                    data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                    X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]

                    X_test = X[stock_idx, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                    Y_test = self.forward_beta[stock_idx, t_test_start_idx:(t_test_end_idx + 1)].copy()
                    data_test = np.concatenate((X_test, Y_test[:, np.newaxis]), axis=1)
                    data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                    data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                    X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                    if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                        continue

                    model = utils.support_vector_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True, kernel=self.kernel)
                    model.fit()
                    Y_pred = model.predict(X_test).reshape(-1, 1)
                    mse_temp.append(np.mean((Y_test - Y_pred) ** 2))

                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean(mse_temp))
                t_idx += self.t_update_freq

        else:
            while True:
                t_train_start_idx = t_idx - 251
                t_train_end_idx = t_idx
                t_test_start_idx = t_train_end_idx + 1
                t_test_end_idx = min(t_test_start_idx + self.t_update_freq - 1, self.T - 1)
                print("training period: {} - {}; testing period: {} - {}".format(self.time_axis[t_train_start_idx], self.time_axis[t_train_end_idx], self.time_axis[t_test_start_idx], self.time_axis[t_test_end_idx]))

                if t_test_start_idx >= self.T:
                    break

                X_train = X[:, t_train_start_idx:(t_train_end_idx + 1), :].copy()
                Y_train = self.forward_beta[:, t_train_start_idx:(t_train_end_idx + 1)].copy()
                data_train = np.concatenate((X_train, Y_train[:, :, np.newaxis]), axis=2)
                data_train = data_train.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_train = data_train[~np.isnan(data_train).any(axis=1), :]
                data_train = data_train[~np.isinf(data_train).any(axis=1), :]
                X_train = data_train[:, 0:len(self.feature)]; Y_train = data_train[:, [len(self.feature)]]
                print("X_train shape: ", X_train.shape)

                X_test = X[:, t_test_start_idx:(t_test_end_idx + 1), :].copy()
                Y_test = self.forward_beta[:, t_test_start_idx:(t_test_end_idx + 1)].copy()
                data_test = np.concatenate((X_test, Y_test[:, :, np.newaxis]), axis=2)
                data_test = data_test.swapaxes(0, 1).reshape((-1, len(self.feature)+1))
                data_test = data_test[~np.isnan(data_test).any(axis=1), :]
                data_test = data_test[~np.isinf(data_test).any(axis=1), :]
                X_test = data_test[:, 0:len(self.feature)]; Y_test = data_test[:, [len(self.feature)]]
                if X_test.shape[0] == 0 or X_train.shape[0] <= 100:
                    break

                model = utils.support_vector_regression(X_train, Y_train, X_columns=self.feature, is_normalize=True, kernel=self.kernel)
                model.fit()
                Y_pred = model.predict(X_test).reshape((-1, 1))
                time_hist.append(self.time_axis[t_test_start_idx])
                mse_hist.append(np.mean((Y_test - Y_pred) ** 2))
                self.optimal_factor_num_hist[(t_test_start_idx, t_test_end_idx)] = model.opt_params
                t_idx += self.t_update_freq

        plt.ion()
        file_name = os.path.join(os.getcwd(), "results", "support_vector_regression_{}.pkl".format(self.feature_label))
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model_performance_summary = pickle.load(f)
        else:
            model_performance_summary = {}

        key = "stock_dependent: {}".format(is_stock_dependent)
        if not is_stock_dependent:
            model_performance_summary["optimal_factor_num_hist"] = self.optimal_factor_num_hist
        model_performance_summary[key] = {"time_hist": time_hist, "mse_hist": mse_hist}
        with open(file_name, "wb") as f:
            pickle.dump(model_performance_summary, f)

        return time_hist, mse_hist




#%%




