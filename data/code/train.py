# %%
# General imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping


warnings.filterwarnings('ignore')

DIR = "../"
DIR_DATA_PRE = DIR + "user_data/preprocessed/"
DIR_MODEL_LGB = DIR + "user_data/model/"


# %%
def save_model_callback(model_prefix, period=100):
    def callback(env):
        if env.iteration % period == 0:
            # 保存每轮模型
            model_name = f"{model_prefix}_iteration_{env.iteration}.txt"
            env.model.save_model(model_name)
    return callback


def train(
    model_name,
    train_data,
    test_data,
    iter_num=None,
    show_importance=False
):
    X_train = train_data
    X_test = test_data
    y_train = X_train['quantity']
    y_test = X_test['quantity']
    X_train = X_train.drop(['date', 'quantity'], axis=1)
    X_test = X_test.drop(['date', 'quantity'], axis=1)
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=True)

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'tweedie',
        'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'subsample': 0.5,
        'subsample_freq': 1,
        'learning_rate': 0.015,
        'num_leaves': 2**11-1,
        'min_data_in_leaf': 2**12-1,
        'feature_fraction': 0.5,
        'max_bin': 100,
        'n_estimators': 3000,
        'boost_from_average': False,
        'verbose': -1,
        # 'device': 'gpu'
    } 

    if iter_num is not None:
        lgb_params['num_iterations'] = iter_num
    
    np.random.seed(23)
    random.seed(23)
    
    callbacks = [log_evaluation(period=10), early_stopping(stopping_rounds=30), save_model_callback(DIR_MODEL_LGB + model_name, period=100)]

    model = lgb.train(lgb_params, train_data, valid_sets=[train_data, test_data], callbacks=callbacks)

    if show_importance:
        lgb.plot_importance(model, max_num_features=30, figsize=(12, 6))
        plt.title("Featurertances")
        plt.show()

    model.save_model(DIR_MODEL_LGB + f'{model_name}_final.txt')

# %%
print("partial_30 train start!")
data_merged = pd.read_pickle(DIR_DATA_PRE + "data_merged.pkl")
train_data = data_merged[(data_merged['date'] >= pd.to_datetime("2023-08-01")) & (data_merged['date'] < pd.to_datetime("2023-09-01"))].copy()
test_data = data_merged[(data_merged['date'] < pd.to_datetime("2022-09-01"))].copy()
del data_merged
train(
    "partial_30",
    train_data,
    test_data,
)
del train_data, test_data
print("partial_30 train finished!")

# %%
print("partial_90 train start!")
data_merged = pd.read_pickle(DIR_DATA_PRE + "data_merged.pkl")
train_data = data_merged[(data_merged['date'] >= pd.to_datetime("2023-06-01")) & (data_merged['date'] < pd.to_datetime("2023-09-01"))].copy()
test_data = data_merged[(data_merged['date'] < pd.to_datetime("2022-09-01"))].copy()
del data_merged
train(
    "partial_90",
    train_data,
    test_data,
)
del train_data, test_data
print("partial_90 train finished!")

# %%
print("partial_365 train start!")
data_merged = pd.read_pickle(DIR_DATA_PRE + "data_merged.pkl")
train_data = data_merged[(data_merged['date'] >= pd.to_datetime("2022-09-01")) & (data_merged['date'] < pd.to_datetime("2023-09-01"))].copy()
test_data = data_merged[(data_merged['date'] < pd.to_datetime("2022-09-01"))].copy()
del data_merged
train(
    "partial_365",
    train_data,
    test_data,
)
del train_data, test_data
print("partial_365 train finished!")

# %%
print("all train start!")
data_merged = pd.read_pickle(DIR_DATA_PRE + "data_merged.pkl")
train_data = data_merged[(data_merged['date'] < pd.to_datetime("2023-09-01"))].copy()
test_data = data_merged[(data_merged['date'] < pd.to_datetime("2022-09-01"))].copy()
del data_merged
train(
    "all",
    train_data,
    test_data,
    iter_num=1400
)
del train_data, test_data
print("all train finished!")
