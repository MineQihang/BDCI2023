# %%
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings('ignore')

DIR = "../"
DIR_DATA_RAW = DIR + "raw_data/"
DIR_DATA_PRE = DIR + "user_data/preprocessed/"

# %% [markdown]
# tools
# %%

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# %% [markdown]
# # 1. 单个文件数据
# 
# ## 1.1 `sku_info.csv`
# 
# 包含sku_id,item_first_cate_cd,item_second_cate_cd,item_third_cate_cd,brand_code
# 
# 先暂时不做处理

# %%
print("processing sku_info.csv...")
data_sku_info = pd.read_csv(DIR_DATA_RAW + "sku_info.csv")
data_sku_info = reduce_mem_usage(data_sku_info)
data_sku_info.to_pickle(DIR_DATA_PRE + "data_sku_info.pkl")
del data_sku_info

# %% [markdown]
# ## 1.2 `sku_price_and_status.csv`
# 
# 包含store_id,sku_id,date,salable_status,stock_status,original_price
# 
# 对从最小日期到最大日期出现original_price缺失的情况进行向前填充

# %%
print("processing sku_price_and_status.csv...")
data_sku_price_and_status = pd.read_csv(DIR_DATA_RAW + "sku_price_and_status.csv")
data_sku_price_and_status['date'] = pd.to_datetime(data_sku_price_and_status['date'])
# 按照store_id, sku_id进行分组，然后补齐从最小日期到最大日期的数据
data_sku_price_and_status = data_sku_price_and_status.groupby(["store_id", "sku_id"]).apply(lambda x: x.set_index("date").resample("D").ffill()).drop(["store_id", "sku_id"], axis=1).reset_index()
data_sku_price_and_status['date'] = pd.to_datetime(data_sku_price_and_status['date']).dt.date
data_sku_price_and_status.drop(["salable_status", "stock_status"], axis=1, inplace=True)
# 原价格的统计量
for operation in ['max', 'min', 'mean', 'std', 'median', 'skew']:
    data_sku_price_and_status['price_' + operation] = data_sku_price_and_status.groupby(['store_id', 'sku_id'])['original_price'].transform(operation)
# 原价格的分位数
for quantile in [0.25, 0.75, 0.50, 0.05, 0.95, 0.10, 0.90]:
    data_sku_price_and_status['price_quantile_' + str(int(quantile * 100))] = data_sku_price_and_status.groupby(['store_id', 'sku_id'])['original_price'].transform(lambda x: x.quantile(quantile))
# 归一化
data_sku_price_and_status['price_norm'] = data_sku_price_and_status['original_price'] / data_sku_price_and_status['price_max']
# 商品的价格数量（商品有多少种价格）
data_sku_price_and_status['price_nunique'] = data_sku_price_and_status.groupby(['store_id', 'sku_id'])['original_price'].transform('nunique')
# 相同价格的商品数量
data_sku_price_and_status['sku_nunique'] = data_sku_price_and_status.groupby(['store_id', 'original_price'])['sku_id'].transform('nunique')
# 价格变化
data_sku_price_and_status['price_momentum'] = data_sku_price_and_status['original_price'] / data_sku_price_and_status.groupby(['store_id', 'sku_id'])['original_price'].transform(lambda x: x.shift(1)).fillna(0)
# 价格在全年、全月的变化
data_sku_price_and_status['year'] = pd.to_datetime(data_sku_price_and_status['date']).dt.year
data_sku_price_and_status['month'] = pd.to_datetime(data_sku_price_and_status['date']).dt.month
data_sku_price_and_status['price_momentum_m'] = data_sku_price_and_status['original_price'] / data_sku_price_and_status.groupby(['store_id', 'sku_id', 'month'])['original_price'].transform('mean')
data_sku_price_and_status['price_momentum_y'] = data_sku_price_and_status['original_price'] / data_sku_price_and_status.groupby(['store_id', 'sku_id', 'year'])['original_price'].transform('mean')
data_sku_price_and_status = reduce_mem_usage(data_sku_price_and_status)
data_sku_price_and_status.to_pickle(DIR_DATA_PRE + "data_sku_price_and_status.pkl")
del data_sku_price_and_status

# %% [markdown]
# ## 1.3 `sku_prom.csv`
# 
# 包含store_id,sku_id,date,promotion_id,curr_day,total_days,promotion_type,threshold,discount_off,channel
# 

# %%
print("processing sku_prom.csv...")
data_sku_prom = pd.read_csv(DIR_DATA_RAW + "sku_prom.csv")
data_sku_prom['date'] = pd.to_datetime(data_sku_prom['date']).dt.date
data_sku_prom["row_num"] = data_sku_prom.groupby(["store_id", "sku_id", "date", "channel"]).cumcount()
data_sku_prom = data_sku_prom.set_index(["store_id", "sku_id", "date", "channel", "row_num"]).unstack(fill_value=0, level=-1)
data_sku_prom.columns = [f'{col}_{row_num}' for col, row_num in data_sku_prom.columns]
data_sku_prom.reset_index(inplace=True)
data_sku_prom = reduce_mem_usage(data_sku_prom)
data_sku_prom.to_pickle(DIR_DATA_PRE + "data_sku_prom.pkl")
del data_sku_prom

# %% [markdown]
# ## 1.4 `sku_sales.csv`
# 
# 包含order_id,store_id,sku_id,order_time,quantity,channel
# 
# 我们对quantity求和即可

# %%
print("processing sku_sales.csv...")
data_sku_sales = pd.read_csv(DIR_DATA_RAW + "sku_sales.csv")
data_sku_sales['order_time'] = pd.to_datetime(data_sku_sales['order_time'])
data_sku_sales['date'] = data_sku_sales['order_time'].dt.date
data_sku_sales = data_sku_sales.drop(["order_id", "order_time"], axis=1).groupby(["store_id", "sku_id", "date", "channel"]).sum().reset_index()
data_sku_sales = reduce_mem_usage(data_sku_sales)
data_sku_sales.to_pickle(DIR_DATA_PRE + "data_sku_sales.pkl")
del data_sku_sales

# %% [markdown]
# ## 1.5 `store_weather.csv`
# 
# 包含store_id,date,weather_type,min_temperature,max_temperature
# 

# %%
print("processing store_weather.csv...")
data_store_weather = pd.read_csv(DIR_DATA_RAW + "store_weather.csv")
data_store_weather['date'] = pd.to_datetime(data_store_weather['date']).dt.date
data_store_weather = reduce_mem_usage(data_store_weather)
data_store_weather.to_pickle(DIR_DATA_PRE + "data_store_weather.pkl")
del data_store_weather

# %% [markdown]
# # 2. 合并数据

# %%
print("merging data...")
data_sku_info = pd.read_pickle(DIR_DATA_PRE + "data_sku_info.pkl")
data_sku_price_and_status = pd.read_pickle(DIR_DATA_PRE + "data_sku_price_and_status.pkl")
data_sku_prom = pd.read_pickle(DIR_DATA_PRE + "data_sku_prom.pkl")
data_sku_sales = pd.read_pickle(DIR_DATA_PRE + "data_sku_sales.pkl")
data_store_weather = pd.read_pickle(DIR_DATA_PRE + "data_store_weather.pkl")
data_merged = pd.merge(data_sku_price_and_status, data_sku_info, on=["sku_id"], how="left")
data_merged = pd.merge(data_merged, pd.DataFrame({"channel": [1, 2]}), how="cross")
del data_sku_price_and_status, data_sku_info
data_merged = pd.merge(data_merged, data_store_weather, on=["store_id", "date"], how="left")
del data_store_weather
data_merged = pd.merge(data_merged, data_sku_sales, on=["store_id", "sku_id", "date", "channel"], how="left").fillna(0)
del data_sku_sales
data_merged = pd.merge(data_merged, data_sku_prom, on=["store_id", "sku_id", "date", "channel"], how="left").fillna(0)
del data_sku_prom

# %%
print("processing data_merged...")
PREDICT_WINDOW = 14

for day in [1,3,5,7,14,21,30,60,90]:
    print("processing day: ", day)
    data_merged[f'sales_{day}_pw_mean'] = data_merged.groupby(['store_id', 'sku_id', 'channel'])['quantity'].transform(lambda x: x.shift(PREDICT_WINDOW).rolling(day).mean())
    data_merged[f'sales_{day}_pw_std'] = data_merged.groupby(['store_id', 'sku_id', 'channel'])['quantity'].transform(lambda x: x.shift(PREDICT_WINDOW).rolling(day).std())
    data_merged[f'sales_{day}_pw_mean_change'] = data_merged.groupby(['store_id', 'sku_id', 'channel'])['quantity'].transform(lambda x: x.shift(PREDICT_WINDOW).rolling(day).mean().pct_change())

data_merged.fillna(0, inplace=True)
data_merged = reduce_mem_usage(data_merged)

# %%
# 分类编码
cols = [
    ['store_id'],
    ['sku_id'],
    ['channel'],
    ['store_id', 'sku_id'],
    ['store_id', 'channel'],
    ['sku_id', 'channel'],
    ['store_id', 'sku_id', 'channel'],
    ['item_first_cate_cd'],
    ['item_first_cate_cd', 'item_second_cate_cd'],
    ['item_first_cate_cd', 'item_second_cate_cd', 'item_third_cate_cd'],
    ['brand_code'],
    ['item_first_cate_cd', 'item_second_cate_cd', 'item_third_cate_cd', 'brand_code']
]

for col in cols:
    print("encoding:", col)
    col_name = '_'+'_'.join(col)+'_'
    data_merged['enc'+col_name+'mean'] = data_merged.groupby(col)['original_price'].transform('mean').astype(np.float16)
    data_merged['enc'+col_name+'std'] = data_merged.groupby(col)['original_price'].transform('std').astype(np.float16)


# %%
print("processing date...")
data_merged['date'] = pd.to_datetime(data_merged['date'])
data_merged['year'] = data_merged['date'].dt.year
data_merged['month'] = data_merged['date'].dt.month
data_merged['day'] = data_merged['date'].dt.day
data_merged['dayofweek'] = data_merged['date'].dt.dayofweek
data_merged['dayofyear'] = data_merged['date'].dt.dayofyear
data_merged['quarter'] = data_merged['date'].dt.quarter
data_merged['is_month_start'] = data_merged['date'].dt.is_month_start
data_merged['is_month_end'] = data_merged['date'].dt.is_month_end
data_merged['is_quarter_start'] = data_merged['date'].dt.is_quarter_start
data_merged['is_quarter_end'] = data_merged['date'].dt.is_quarter_end
data_merged['is_year_start'] = data_merged['date'].dt.is_year_start
data_merged['is_year_end'] = data_merged['date'].dt.is_year_end
data_merged['days_in_month'] = data_merged['date'].dt.days_in_month
data_merged['is_leap_year'] = data_merged['date'].dt.is_leap_year

# %%
data_merged = reduce_mem_usage(data_merged)
data_merged.to_pickle(DIR_DATA_PRE + "data_merged.pkl")

# %%
print("preprocessing finished!")
