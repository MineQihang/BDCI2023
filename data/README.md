# 解决方案

本方案使用[Kaggle M5竞赛](https://www.kaggle.com/competitions/m5-forecasting-accuracy)中的[Top1解决方案](https://github.com/Mcompetitions/M5-methods/tree/master/Code of Winning Methods/A1)作为baseline进行算法设计。

我们使用“预测、备货、分配”三步走策略来解决该问题，下面我们将从这三方面来分析我们的方案。

## 1 销量预测

### 1.1 特征工程

#### 1.1.1 价格特征

首先我们需要为单个`(store_id, sku_id)`填充从开始日期到最大日期之间的所有日期，从而方便计算后续的价格变化率等时序特征。对于缺失的值（价格、状态等）直接向前填充。

填充结束后，我们计算以下统计量并作为特征：

- 原始价格最大值、最小值、平均值、标准差、中位数、偏度
- 原始价格分位数（25%、75%、50%、5%、95%、10%、90%）
- 原始价格归一化（当前值除最大值）
- 商品原始价格数量（商品有多少种原始价格）
- 相同原始价格的商品数量
- 相对前一天的原始价格变化率
- 相对全年的价格变化率
- 相对全月的价格变化率

除此之外，我们还进行了分类编码：对相同的键，对原始价格进行求均值和标准差，这样可以得出不同类别类内的统计特征。我们设计了以下键：

```python
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
```

#### 1.1.2 促销特征

对于相同的`(store_id, sku_id, date, channel)`来说，促销不是单一的。从实际业务来说，商品参与多种促销活动是非常正常的现象。因此，如何处理促销以使其对其，从而能方便训练是非常重要的。

我们通过探索性数据分析（EDA）发现，在相同`(store_id, sku_id, date, channel)`下，促销最多有3个，因此我们将单个个体所有促销进行展开，展开为3大列，每一大列包括`promotion_id`、`curr_day`、`total_days`、`promotion_type`、`threshold`、`discount_off`六小列。对于促销小于3个的，展开后会出现`nan`值，我们直接填充0就可以了（因为0并未对应任何促销）。

具体来说，我们可以使用以下代码进行处理：

```python
data_sku_prom["row_num"] = data_sku_prom.groupby(["store_id", "sku_id", "date", "channel"]).cumcount()
data_sku_prom = data_sku_prom.set_index(["store_id", "sku_id", "date", "channel", "row_num"]).unstack(fill_value=0, level=-1)
data_sku_prom.columns = [f'{col}_{row_num}' for col, row_num in data_sku_prom.columns]
data_sku_prom.reset_index(inplace=True)
```

#### 1.1.3 销量特征

将所有的订单使用`(store_id, sku_id, date, channel)`对`quantity`求和来统计销量。

因为我们不使用递归预测，因此对于一0行来说，我们先向前推两周，然后计算前$x$天的销量均值、标准差、均值变化。然后对`nan`值填充为0。

具体来说，我们可以使用以下代码进行处理：

```python
PREDICT_WINDOW = 14
for day in [1,3,5,7,14,21,30,60,90]:
    print("processing day: ", day)
    data_merged[f'sales_{day}_pw_mean'] = data_merged.groupby(['store_id', 'sku_id', 'channel'])['quantity'].transform(lambda x: x.shift(PREDICT_WINDOW).rolling(day).mean())
    data_merged[f'sales_{day}_pw_std'] = data_merged.groupby(['store_id', 'sku_id', 'channel'])['quantity'].transform(lambda x: x.shift(PREDICT_WINDOW).rolling(day).std())
    data_merged[f'sales_{day}_pw_mean_change'] = data_merged.groupby(['store_id', 'sku_id', 'channel'])['quantity'].transform(lambda x: x.shift(PREDICT_WINDOW).rolling(day).mean().pct_change())
data_merged.fillna(0, inplace=True)
```

#### 1.1.4 日期特征

我们使用日期计算出了以下特征，从而能捕获时序特征：

- 年份、月份、月内第几天、周内第几天、年内第几天、季度
- 是否月初、是否月末、是否季初、是否季末、是否年初、是否年末、是否闰年

### 1.2 模型训练与集成

我们使用[LightGBM](https://lightgbm.readthedocs.io/)进行对数回归，参数设置参考M5竞赛Top1方案。

我们将除了`date`和`quantity`之外的所有列作为特征列，使用`quantity`列作为标签列。

通过EDA，我们发现每个商品的开售时间有所不同，如下图（横轴代表商品开售时间距2023-09-01的天数，纵轴表示单品数量）。如果直接将数据一股脑喂入模型训练可能会导致模型无法很好的拟合（因为这会存在时间序列的不一致性、特征的不平衡、过拟合等各种问题）。

<img src="https://qihang-1306873228.cos.ap-chongqing.myqcloud.com/imgs/image-20231205120306801.png" alt="image-20231205120306801" style="zoom: 80%;" />

因此，我们希望使用不同时间段的数据来训练多个模型然后进行集成，这样就能综合捕获长短期特征，使得模型的精确度得到提升。

具体来说，我们将2023-09-01前30、90、365、所有天共四份数据分别使用同一模型结构训练出四个不同的模型。集成时直接对模型输出取均值即可。实验及线上评测表明，这样的方式可以有效提升预测精度。

### 1.3 模型预测

将2023-09-01及之后的数据送入得到的四个模型中进行预测，最后对相同单品在同一日期求均值得到最终预测结果。预测结果示例如下：

|      | store_id | sku_id |    date    | channel | quantity |
| :--: | :------: | :----: | :--------: | :-----: | :------: |
|  0   |    1     |   1    | 2023-09-01 |    1    |   2.0    |
|  1   |    1     |   1    | 2023-09-01 |    2    |   1.0    |
| ...  |   ...    |  ...   |    ...     |   ...   |   ...    |

## 2 备货

预测模型总是做不到100%准确，因此我们需要考虑到误差的影响，对商品进行备货。备货策略有很多，我们尝试了如下库存模型：
$$
\tilde y= \lceil \alpha \hat y + \beta \rceil
$$
其中，$$\tilde y$$表示备货量，$$\hat y$$表示预测量，$$\alpha(\alpha\geq1)$$为魔法系数，$$\beta(\beta\geq0)$$为安全库存。

我们可以通过网格搜索得出最优的$$\alpha$$和$$\beta$$。通过实验及线上评测发现，当$$\alpha=1.51,\beta=0$$时，可以比预测即备货的方式（即$$\alpha=1,\beta=0$$）的结果好，即利润更高。

分析可以发现，这样的方式对于所有商品都是用同样的魔法系数和安全库存，那么当$$\alpha>1$$时，如果预测结果很大，那么备货量就会相对更大，而实际上根本不需要准备如此多的额外库存，这么多的额外库存只会导致损耗增多。同时，我们发现模型对大销量的商品预测准确度显著高于小销量的商品。从业务角度来看，这也是非常正常的，因为对大销量的商品，需求相对稳定，预测准确率自然会相对较高。

综合以上两点，我们认为，小销量的商品应该有更多的额外库存，而大销量的商品则更少。因此，我们设计出如下模型：
$$
\tilde y = \lceil (e^{-\alpha \hat y} + \beta) \hat y \rceil
$$
其中，$$\tilde y$$表示备货量，$$\hat y$$表示预测量，$$\alpha(\alpha\geq0), \beta(\beta\geq0)$$为超参数。

这样就能使得结果满足上述要求。我们可以通过验证集来对参数使用梯度下降法进行最优化搜索，最优化目标为利润：
$$
\begin{align}
\max & S(\tilde y) \\
s.t. & \alpha \geq 0 \\
& \beta \geq 0
\end{align}
$$
其中S表示使用备货量计算利润的整个过程（由于过程非常复杂，可以考虑使用代理函数进行代替）。

但上述方法相对难实现且比较耗时，且不一定能收敛到最优解，在赛题时间紧张情况下，我们采用了启发式网格搜索的方式来确定参数，当$$\alpha=0.1,\beta=1.23$$的时候，利润可以达到一个不错的结果。

## 3 库存分配

前后场备货量确定后，我们需要调整备货方案以使其满足条件，并得到较好的利润及履约效率。建立模型如下：
$$
\begin{align}
\max & S(x_{ijt}^k, x_{ijt}^m) \\
\text{s.t.} & x_{ijt}^k \geq 0 \\
& x_{ijt}^m \geq 0 \\
& x_{ijt}^k + x_{ijt}^m = \tilde y_{ijt}^k + \tilde y_{ijt}^m \\
& x_{ijt}^k > 0 \space \text{if} \space x_{ijt}^k + x_{ijt}^m > 0 \\ 
& \frac{\sum_{i = 1}^{I} \mathbb{I}(x_{ijt}^m > 0) }{\sum_{i = 1}^{I} \mathbb{I}(x_{ijt}^k + x_{ijt}^m > 0)} \leq 0.2 \\
& \frac{\sum_{i = 1}^{I} x_{ijt}^m}{\sum_{i = 1}^{I} x_{ijt}^k + x_{ijt}^m} \leq 0.4
\end{align}
$$
其中所有的符号定义来自上文及题目，在此不再赘述。

同样的，由于模型的利润确实较难去计算，要使用运筹模型进行优化时间复杂度过高，恐超过时间限制。因此，我们采用贪心方法得到一个次优解。

我们直接对每个日期每个店铺下所有sku根据后场备货量进行降序排序，取前200个sku保留后场库存，而后800个sku将后场库存全部放到前场中。如果前200个sku的后场库存之和没有满足条件，那么就依次取前199、198、...、0个sku作为保留后场库存的sku，直至满足条件。

通过上述贪心算法，我们完成了库存的分配。至此，我们便可导出结果作为提交文件，进行测评。

