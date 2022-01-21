# Item-Based CF 预测电影评分
# 加载ratings.csv，转换为用户-电影评分矩阵并计算用户之间相似度
import os
import pandas as pd
import numpy as np

DATA_PATH = "./data/ml-latest-small/ratings.csv"

dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
# 加载数据，我们只用前三列数据，分别是用户ID，电影ID，已经用户对电影的对应评分
ratings = pd.read_csv(DATA_PATH, dtype=dtype, usecols=range(3))
print('加载数据：')
print(ratings.head(5))
# 透视表，将电影ID转换为列名称，转换成为一个User-Movie的评分矩阵
ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"], values="rating")
print('透视表：')
print(ratings_matrix.head(5))
# 计算物品之间相似度
item_similar = ratings_matrix.corr()
print('物品相似度：')
print(item_similar.head(5))

# 预测用户对物品的评分 （以用户1对电影1评分为例）
# 1. 找出iid物品的相似物品
similar_items = item_similar[1].drop([1]).dropna()
print('物品1的相似物品：')
print(similar_items.head(5))

# 相似物品筛选规则：正相关的物品
similar_items = similar_items.where(similar_items>0).dropna()
print('物品1的正相关相似物品：')
print(similar_items.head(5))
# 2. 从iid物品的近邻相似物品中筛选出uid用户评分过的物品
index = set(ratings_matrix.loc[1].dropna().index) & set(similar_items.index)
finally_similar_items = similar_items.loc[index]
print('相似物品中用户评分过的物品：')
print(finally_similar_items.head(5))
# 3. 结合iid物品与其相似物品的相似度和uid用户对其相似物品的评分，预测uid对iid的评分
numerator = 0    # 评分预测公式的分子部分的值
denominator = 0    # 评分预测公式的分母部分的值
for sim_iid, similarity in finally_similar_items.iteritems():
    # 获取1用户的评分数据
    sim_user_rated_movies = ratings_matrix.loc[1].dropna()
    # 1用户对相似物品物品的评分
    sim_item_rating_from_user = sim_user_rated_movies[sim_iid]
    # 计算分子的值
    numerator += similarity * sim_item_rating_from_user
    # 计算分母的值
    denominator += similarity

# 计算预测的评分值并返回
predict_rating = numerator/denominator
print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (1, 1, predict_rating))

# 封装成方法：预测任意用户对任意电影的评分
def predict(uid, iid, ratings_matrix, item_similar):
    '''
    预测给定用户对给定物品的评分值
    :param uid: 用户ID
    :param iid: 物品ID
    :param ratings_matrix: 用户-物品评分矩阵
    :param user_similar: 用户两两相似度矩阵
    :return: 预测的评分值
    '''
    print("开始预测用户<%d>对电影<%d>的评分..."%(uid, iid))
    # 1. 找出iid物品的相似物品
    similar_items = item_similar[iid].drop([iid]).dropna()
    # 相似物品筛选规则：正相关的物品
    similar_items = similar_items.where(similar_items>0).dropna()

    if similar_items.empty is True:
        raise Exception("物品<%d>没有相似的物品" % iid)

    # 2. 从iid物品的近邻相似物品中筛选出uid用户评分过的物品
    index = set(ratings_matrix.loc[uid].dropna().index) & set(similar_items.index)
    finally_similar_items = similar_items.loc[index]

    # 3. 结合iid物品与其相似物品的相似度和uid用户对其相似物品的评分，预测uid对iid的评分
    numerator = 0    # 评分预测公式的分子部分的值
    denominator = 0    # 评分预测公式的分母部分的值

    for sim_iid, similarity in finally_similar_items.iteritems():
        # 获取uid用户的评分数据
        sim_user_rated_movies = ratings_matrix.loc[uid].dropna()
        # uid用户对相似物品物品的评分
        sim_item_rating_from_user = sim_user_rated_movies[sim_iid]
        # 计算分子的值
        numerator += similarity * sim_item_rating_from_user
        # 计算分母的值
        denominator += similarity

    # 计算预测的评分值并返回
    predict_rating = numerator/denominator
    print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (uid, iid, predict_rating))
    return round(predict_rating, 2)

# 为某一用户预测所有电影评分
def predict_all(uid, ratings_matrix, item_similar):
    '''
    预测全部评分
    :param uid: 用户id
    :param ratings_matrix: 用户-物品打分矩阵
    :param item_similar: 物品两两间的相似度
    :return: 生成器，逐个返回预测评分
    '''
    # 准备要预测的物品的id列表
    item_ids = ratings_matrix.columns
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, item_similar)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating

# if __name__ == '__main__':
#     for i in predict_all(1, ratings_matrix, item_similar):
#         pass

# 根据评分为指定用户推荐topN个电影
def top_k_rs_result(k):
    results = predict_all(1, ratings_matrix, item_similar)
    return sorted(results, key=lambda x: x[2], reverse=True)[:k]
if __name__ == '__main__':
    from pprint import pprint
    result = top_k_rs_result(20)
    pprint(result)











