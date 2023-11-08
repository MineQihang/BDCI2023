import numpy as np
import pandas as pd
import os


STORE_NUM = 12
SKU_NUM = 1000
CHANNEL_NUM = 2
MAX_PRICE = 100


def get_sku_info(path="data/sku_info.csv", sku_id=None):
    data_sku_info = pd.read_csv(path)
    if sku_id is not None:
        data_sku_info = data_sku_info[data_sku_info["sku_id"] == sku_id]
    return data_sku_info

def get_pre_sku_info(path="data/sku_info.csv", sku_id=None):
    data_sku_info = get_sku_info(path, sku_id)
    if sku_id is None:
        # 将item_first_cate_cd, item_second_cate_cd, item_third_cate_cd, brand_code转化为one-hot编码
        data_sku_info = pd.get_dummies(data_sku_info, columns=["item_first_cate_cd", "item_second_cate_cd", "item_third_cate_cd", "brand_code"])
    else:
        data_sku_info.drop(["item_first_cate_cd", "item_second_cate_cd", "item_third_cate_cd", "brand_code"], axis=1, inplace=True)
    return data_sku_info

def get_sku_prices(path="data/sku_price_and_status.csv", sku_id=None, store_id=None, start_date=None, end_date=None):
    data_sku_prices = pd.read_csv(path)
    data_sku_prices["date"] = pd.to_datetime(data_sku_prices["date"]).dt.date
    global MAX_PRICE
    MAX_PRICE = data_sku_prices["original_price"].max()
    if sku_id is not None:
        data_sku_prices = data_sku_prices[data_sku_prices["sku_id"] == sku_id]
    if store_id is not None:
        data_sku_prices = data_sku_prices[data_sku_prices["store_id"] == store_id]
    if start_date is not None:
        data_sku_prices = data_sku_prices[data_sku_prices["date"] >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        data_sku_prices = data_sku_prices[data_sku_prices["date"] <= pd.to_datetime(end_date).date()]
    return data_sku_prices

def get_pre_sku_prices(path="data/sku_price_and_status.csv", sku_id=None, store_id=None, start_date=None, end_date=None):
    data_sku_prices = get_sku_prices(path, sku_id, store_id, start_date, end_date)
    # 将salable_status和stock_status转为one-hot编码
    data_sku_prices = pd.get_dummies(data_sku_prices, columns=["salable_status", "stock_status"])
    # 将price归一化
    data_sku_prices["original_price"] = data_sku_prices["original_price"] / MAX_PRICE
    return data_sku_prices

def get_sku_sales(path="data/sku_sales.csv", sku_id=None, store_id=None, channel=None, start_date=None, end_date=None):
    data_sku_sales = pd.read_csv(path)
    if sku_id is not None:
        data_sku_sales = data_sku_sales[data_sku_sales["sku_id"] == sku_id]
    if store_id is not None:
        data_sku_sales = data_sku_sales[data_sku_sales["store_id"] == store_id]
    if channel is not None:
        data_sku_sales = data_sku_sales[data_sku_sales["channel"] == channel]
    # 将order_time转化为date
    data_sku_sales["order_time"] = pd.to_datetime(data_sku_sales["order_time"]).dt.date
    data_sku_sales.rename(columns={"order_time": "date"}, inplace=True)
    data_sku_sales = data_sku_sales.groupby(["date", "sku_id", "store_id", "channel"], as_index=False)["quantity"].sum().reindex()
    if start_date is not None:
        data_sku_sales = data_sku_sales[data_sku_sales["date"] >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        data_sku_sales = data_sku_sales[data_sku_sales["date"] <= pd.to_datetime(end_date).date()]
    return data_sku_sales

def get_pre_sku_sales(path="data/sku_sales.csv", sku_id=None, store_id=None, channel=None, start_date=None, end_date=None):
    data_sku_sales = get_sku_sales(path, sku_id, store_id, channel, start_date, end_date)
    return data_sku_sales

def get_sku_prom(path="data/sku_prom.csv", sku_id=None, store_id=None, start_date=None, end_date=None, channel=None):
    data_sku_prom = pd.read_csv(path)
    data_sku_prom["date"] = pd.to_datetime(data_sku_prom["date"]).dt.date
    if sku_id is not None:
        data_sku_prom = data_sku_prom[data_sku_prom["sku_id"] == sku_id]
    if store_id is not None:
        data_sku_prom = data_sku_prom[data_sku_prom["store_id"] == store_id]
    if start_date is not None:
        data_sku_prom = data_sku_prom[data_sku_prom["date"] >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        data_sku_prom = data_sku_prom[data_sku_prom["date"] <= pd.to_datetime(end_date).date()]
    if channel is not None:
        data_sku_prom = data_sku_prom[data_sku_prom["channel"] == channel]
    return data_sku_prom

def get_pre_sku_prom(path="data/sku_prom.csv", sku_id=None, store_id=None, start_date=None, end_date=None, channel=None):
    data_sku_prom = get_sku_prom(path, sku_id, store_id, start_date, end_date, channel)
    # 去除promotion_id
    data_sku_prom.drop(["promotion_id"], axis=1, inplace=True)
    # 转化curr_day和total_days为比例
    data_sku_prom["curr_day"] = data_sku_prom["curr_day"] / data_sku_prom["total_days"]
    data_sku_prom.drop(["total_days"], axis=1, inplace=True)
    # 将promotion_type转化为one-hot编码
    data_sku_prom = pd.get_dummies(data_sku_prom, columns=["promotion_type"])
    # 将threshold归一化
    data_sku_prom["threshold"] = data_sku_prom["threshold"] / data_sku_prom["threshold"].max()
    # 对于相同的sku_id, store_id, channel, date, 取平均，TODO: 这里可能有问题
    data_sku_prom = data_sku_prom.groupby(["sku_id", "store_id", "channel", "date"], as_index=False).mean().reindex()
    return data_sku_prom

def get_store_weather(path="data/store_weather.csv", store_id=None, start_date=None, end_date=None):
    data_store_weather = pd.read_csv(path)
    data_store_weather["date"] = pd.to_datetime(data_store_weather["date"]).dt.date
    if store_id is not None:
        data_store_weather = data_store_weather[data_store_weather["store_id"] == store_id]
    if start_date is not None:
        data_store_weather = data_store_weather[data_store_weather["date"] >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        data_store_weather = data_store_weather[data_store_weather["date"] <= pd.to_datetime(end_date).date()]
    return data_store_weather

def get_pre_store_weather(path="data/store_weather.csv", store_id=None, start_date=None, end_date=None):
    data_store_weather = get_store_weather(path, store_id, start_date, end_date)
    # 将weather_type转化为one-hot编码
    data_store_weather = pd.get_dummies(data_store_weather, columns=["weather_type"])
    return data_store_weather

def nmf(X, latent_features, max_iter=100, error_limit=1e-6, learning_rate=0.001, l2_penalty=0.001):
    """
    使用带L2正则化的NMF对有缺失值的矩阵X进行分解和填充。

    参数:
        X (np.ndarray): 原始数据矩阵，缺失值为np.nan
        latent_features (int): 隐含特征的数量
        max_iter (int): 最大迭代次数
        error_limit (float): 重构误差的阈值，达到则停止
        learning_rate (float): 梯度下降的学习率
        l2_penalty (float): L2正则化的权重

    返回:
        np.ndarray: 填充后的矩阵
    """

    # 初始化W和H为随机值
    W = np.abs(np.random.randn(X.shape[0], latent_features))
    H = np.abs(np.random.randn(latent_features, X.shape[1]))

    # 只在非缺失值上进行拟合
    mask = np.isfinite(X)
    X_safe = np.where(mask, X, 0)

    for epoch in range(max_iter):
        WH = np.dot(W, H)

        # 计算只针对非缺失值的重构误差
        cost = np.sum((mask * (X_safe - WH)) ** 2) + l2_penalty * (np.sum(W ** 2) + np.sum(H ** 2))

        if cost < error_limit:
            break

        if epoch % 10 == 0:
            print(f'Iteration {epoch}: cost={cost}')

        # 计算梯度
        grad_W = -2 * np.dot(mask * (X_safe - WH), H.T) + 2 * l2_penalty * W
        grad_H = -2 * np.dot(W.T, mask * (X_safe - WH)) + 2 * l2_penalty * H

        # 更新矩阵
        W -= learning_rate * grad_W
        H -= learning_rate * grad_H

        # 保持非负性
        W = np.maximum(W, 0)
        H = np.maximum(H, 0)

    return np.dot(W, H)

def get_pre_all_data(path="data", sku_id=None, store_id=None, channel=None, start_date=None, end_date=None):
    data_pre_sku_info = get_pre_sku_info(os.path.join(path, "sku_info.csv"), sku_id=sku_id)
    data_pre_sku_prices = get_pre_sku_prices(os.path.join(path, "sku_price_and_status.csv"), sku_id=sku_id, store_id=store_id, start_date=start_date, end_date=end_date)
    data_pre_sku_sales = get_pre_sku_sales(os.path.join(path, "sku_sales.csv"), sku_id=sku_id, store_id=store_id, channel=channel, start_date=start_date, end_date=end_date)
    data_pre_sku_prom = get_pre_sku_prom(os.path.join(path, "sku_prom.csv"), sku_id=sku_id, store_id=store_id, start_date=start_date, end_date=end_date, channel=channel)
    data_pre_store_weather = get_pre_store_weather(os.path.join(path, "store_weather.csv"), store_id=store_id, start_date=start_date, end_date=end_date)
    
    # 生成满足所有情况的数据
    if sku_id is None:
        sku_list = pd.DataFrame({"sku_id": np.arange(1, SKU_NUM + 1)})
    else:
        sku_list = pd.DataFrame({"sku_id": [sku_id]})
    if store_id is None:
        store_list = pd.DataFrame({"store_id": np.arange(1, STORE_NUM + 1)})
    else:
        store_list = pd.DataFrame({"store_id": [store_id]})
    if channel is None:
        channel_list = pd.DataFrame({"channel": np.arange(1, CHANNEL_NUM + 1)})
    else:
        channel_list = pd.DataFrame({"channel": [channel]})
    if start_date is None and end_date is None:
        date_list = pd.DataFrame({"date": pd.date_range(data_pre_sku_sales["date"].min(), data_pre_sku_sales["date"].max())})
    elif start_date is None:
        date_list = pd.DataFrame({"date": pd.date_range(data_pre_sku_sales["date"].min(), pd.to_datetime(end_date).date())})
    elif end_date is None:
        date_list = pd.DataFrame({"date": pd.date_range(pd.to_datetime(start_date).date(), data_pre_sku_sales["date"].max())})
    else:
        date_list = pd.DataFrame({"date": pd.date_range(pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date())})
    
    # 笛卡尔积
    data_pre_all = pd.MultiIndex.from_product([sku_list["sku_id"].to_list(), store_list["store_id"].to_list(), channel_list["channel"].to_list(), date_list["date"].to_list()], names=["sku_id", "store_id", "channel", "date"])
    data_pre_all = pd.DataFrame(data_pre_all.to_frame(), columns=["sku_id", "store_id", "channel", "date"]).reset_index(drop=True)
    data_pre_all["date"] = pd.to_datetime(data_pre_all["date"]).dt.date

    # 将其他数据与data_pre_all合并
    data_pre_all = pd.merge(data_pre_all, data_pre_sku_info, on=["sku_id"], how="left")
    if data_pre_all.isnull().any().any():
        raise Exception("sku_info有缺失值")
    
    data_pre_all = pd.merge(data_pre_all, data_pre_sku_prices, on=["sku_id", "store_id", "date"], how="left")
    # data_pre_all.fillna(0, inplace=True)
    
    data_pre_all = pd.merge(data_pre_all, data_pre_sku_sales, on=["sku_id", "store_id", "channel", "date"], how="left")
    # data_pre_all['quantity'] = nmf(data_pre_all['quantity'].values.reshape((1, -1)), latent_features=5, max_iter=10000, learning_rate=0.01, error_limit=1e-6, l2_penalty=0).reshape((-1, 1))
    
    data_pre_all = pd.merge(data_pre_all, data_pre_sku_prom, on=["sku_id", "store_id", "channel", "date"], how="left")
    # data_pre_all.fillna(0, inplace=True)
    
    data_pre_all = pd.merge(data_pre_all, data_pre_store_weather, on=["store_id", "date"], how="left")
    # data_pre_all.fillna(0, inplace=True)

    return data_pre_all
    

if __name__ == "__main__":
    data_pre_all = get_pre_all_data("../../data", start_date="2023-08-29", end_date="2023-08-30")
    print(data_pre_all)