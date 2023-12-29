# %%
# General imports
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
import pickle


warnings.filterwarnings('ignore')

DIR = "../"
DIR_DATA_PRE = DIR + "user_data/preprocessed/"
DIR_MODEL_LGB = DIR + "user_data/model/"
DIR_SUBMIT = DIR + "prediction_result/"

# %%
data_merged = pd.read_pickle(DIR_DATA_PRE + "data_merged.pkl")

# %%
def test(model_name):
    # 加载模型
    model = lgb.Booster(model_file=DIR_MODEL_LGB + model_name)
    # 加载要预测的数据
    data_pred = data_merged[data_merged['date'] >= '2023-09-01']
    # 预测
    data_pred['quantity'] = model.predict(data_pred.drop(["quantity", "date"], axis=1))
    # 只保留需要的列
    data_pred = data_pred[['store_id', 'sku_id', 'date', 'channel', 'quantity']]
    return data_pred

# %%
print('Predicting using partial_30_final.txt...')
pred_30 = test('partial_30_final.txt')

# %%
print('Predicting using partial_90_final.txt...')
pred_90 = test('partial_90_final.txt')

# %%
print('Predicting using partial_365_final.txt...')
pred_365 = test('partial_365_final.txt')

# %%
print('Predicting using all_final.txt...')
pred_all = test('all_final.txt')

# %%
pred_list = [pred_365.copy(), pred_90.copy(), pred_30.copy(), pred_all.copy()]
pred_avg = pred_list[0].copy()
for pred in pred_list[1:]:
    pred_avg['quantity'] += pred['quantity']
pred_avg['quantity'] /= len(pred_list)
pred_list.append(pred_avg)

# %%
with open(DIR_DATA_PRE + 'pred_list.pkl', 'wb') as f:
    pickle.dump(pred_list, f)
del pred_list

# %%
pred_list = pickle.load(open(DIR_DATA_PRE + 'pred_list.pkl', 'rb'))
pred_avg = pred_list[-1]

# %%
def form_test_results(
    pred_data,
    date_start='2023-09-01',
    date_end='2023-09-14',
):
    # store_id从1-12，sku_id从1-1000，order_time从2023-08-18到2023-08-31
    store_ids = list(range(1, 13))
    sku_ids = list(range(1, 1001))
    dates = pd.date_range(date_start, date_end).date
    # 生成store_id, sku_id, order_time的笛卡尔积
    cartesian = pd.MultiIndex.from_product([store_ids, sku_ids, dates], names=['store_id', 'sku_id', 'date'])
    data_now_stocks = pd.DataFrame(cartesian.to_frame(), columns=['store_id', 'sku_id', 'date']).reset_index(drop=True)
    data_now_stocks["date"] = pd.to_datetime(data_now_stocks["date"]).dt.date
    # 合并预测结果
    pred_data['date'] = pd.to_datetime(pred_data['date']).dt.date
    pred_data = pred_data.pivot_table(
        index=['store_id', 'sku_id', 'date'], 
        columns=['channel'], 
        values='quantity', fill_value=0).reset_index()
    pred_data.columns = ['store_id', 'sku_id', 'date', 'x_k', 'x_m']
    # pred_data['x_k'], pred_data['x_m'] = pred_data['x_m'], pred_data['x_k']
    data_now_stocks = pd.merge(data_now_stocks, pred_data, how='left')
    return data_now_stocks

# %%
def allocate(
    test_results
):
    # TODO: 根据物品一起出现的频次来分配
    mask_kzero = (test_results['x_k'] == 0) & (test_results['x_m'] > 0)
    test_results.loc[mask_kzero, 'x_k'] = test_results.loc[mask_kzero, 'x_m']
    test_results.loc[mask_kzero, 'x_m'] = 0
    # 计算每个门店每天的总库存
    test_results['x'] = test_results['x_k'] + test_results['x_m']
    # mask_exceed = test_results['x'] > results_max['x']
    # test_results.loc[mask_exceed, 'x_k'] = np.ceil(test_results.loc[mask_exceed, 'x_k'] * results_max.loc[mask_exceed, 'x'] / test_results.loc[mask_exceed, 'x'])
    # test_results.loc[mask_exceed, 'x'] = results_max.loc[mask_exceed, 'x']
    # test_results.loc[mask_exceed, 'x_m'] = results_max.loc[mask_exceed, 'x'] - test_results.loc[mask_exceed, 'x_k']
    # print("exceed num:", mask_exceed.sum())
    x_sum = test_results.groupby(['store_id', 'date'], as_index=False).agg({'x': 'sum'})
    x_sum.columns = ['store_id', 'date', 'x_sum']
    test_results = pd.merge(test_results, x_sum, how='left')
    # 按照x_m排序
    test_results.sort_values(['store_id', 'date', 'x_m'], ascending=[True, True, False], inplace=True)
    test_results['x_m_cumsum'] = test_results.groupby(['store_id', 'date'])['x_m'].cumsum()
    # 分配
    mask_need = (test_results['x_m_cumsum'] <= test_results['x_sum'] * 0.4) & np.tile(np.array([True] * 200 + [False] * 800), test_results.shape[0] // 1000)
    # print("need num:", mask_need.sum())
    test_results.loc[~mask_need, 'x_k'] = test_results.loc[~mask_need, 'x']
    test_results.loc[~mask_need, 'x_m'] = 0
    # 重新排序返回
    test_results.sort_values(['store_id', 'sku_id', 'date'], inplace=True)
    return test_results[['store_id', 'sku_id', 'date', 'x_k', 'x_m']]

# %%
test_results = pred_avg.copy()
# 分配
test_results['quantity'] = test_results['quantity'].apply(lambda x: np.ceil(x * (np.exp(-x/10) + 1.23)) if x > 0 else 0)
test_results = form_test_results(test_results)
test_results.fillna(0, inplace=True)
test_results = allocate(test_results)

# %%
test_results.to_csv(DIR_SUBMIT + "result.csv", index=False)

print("prediction finished!\nthe result has been exported to the `prediction_result` folder!")
