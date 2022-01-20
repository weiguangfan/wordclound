# 构建数据集：注意这里构建评分数据时，对于缺失的部分我们需要保留为None，如果设置为0那么会被当作评分值为0去对待
users = ["User1", "User2", "User3", "User4", "User5"]
items = ["Item A", "Item B", "Item C", "Item D", "Item E"]
# 用户评分记录数据集
datasets = [
    [5,3,4,4,None],
    [3,1,2,3,3],
    [4,3,4,3,5],
    [3,3,1,5,4],
    [1,5,5,2,1],
]
import pandas as pd
import numpy as np
from pprint import pprint

# 计算相似度：对于评分数据这里我们采用皮尔逊相关系数[-1,1]来计算，-1表示强负相关，+1表示强正相关
# pandas中corr方法可直接用于计算皮尔逊相关系数
df = pd.DataFrame(datasets,
                  columns=items,
                  index=users)
pprint(df)
# 直接计算皮尔逊相关系数
# 默认是按列进行计算，因此如果计算用户间的相似度，当前需要进行转置
user_similar = df.T.corr()
print("用户之间的两两相似度：")
print(user_similar.round(4))

item_similar = df.corr()
print("物品之间的两两相似度：")
print(item_similar.round(4))

# 注意：我们在预测评分时，往往是通过与其有正相关的用户或物品进行预测，如果不存在正相关的情况，那么将无法做出预测。
# 这一点尤其是在稀疏评分矩阵中尤为常见，因为稀疏评分矩阵中很难得出正相关系数。
# 评分预测：
# User-Based CF 评分预测：使用用户间的相似度进行预测
# 关于评分预测的方法也有比较多的方案，下面介绍一种效果比较好的方案，该方案考虑了用户本身的评分评分以及近邻用户的加权平均相似度打分来进行预测
# Item-Based CF 评分预测：使用物品间的相似度进行预测
# 这里利用物品相似度预测的计算同上，同样考虑了用户自身的平均打分因素，结合预测物品与相似物品的加权平均相似度打分进行来进行预测
# User-Based CF与Item-Based CF比较
# User-Based CF和Item-Based CF 严格意义上属于两种不同的推荐算法（预测评分结果存在差异）
# User-Based CF应用场景：用户少于物品，或者用户没用物品变化快的场景 （举例，信息流产品）
# Item-Based CF应用场景：用户多于物品，或者物品变化较慢的场景 （举例，电商应用）
# 不好确定那种更合适，可以两种算法都去实现，然后对推荐效果进行评估分析选出更优方案








