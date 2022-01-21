# Model-Based CF（基于模型的协同过滤）
# 利用机器学习算法预测用户对物品的喜好程度
# 当用户-物品的评分数据比较稀疏的时候更适合使用Model-Based CF
# Model-Based 协同过滤算法会利用用户-物品评分矩阵训练机器学习模型，
# 将用户和物品的id带入到模型中就会得到用户对物品的评分预测，
# Model-Based CF算法分类如下：
# 基于分类算法、回归算法、聚类算法
# 基于矩阵分解的算法
# 基于神经网络的算法
# 基于图模型的算法
# 我们重点学习以下两种应用较多的方案：
# 基于回归模型的协同过滤推荐
# 基于矩阵分解的协同过滤推荐


# 基于回归模型的协同过滤推荐
# Baseline：基准预测原理
# 如果我们将评分看作是一个连续的值而不是离散的值，
# 那么就可以借助线性回归思想来预测目标用户对某物品的评分。
# 其中一种实现策略被称为Baseline（基准预测）。
# Baseline设计思想基于以下的假设：
# 有些用户的评分普遍高于其他用户，有些用户的评分普遍低于其他用户。
# 比如：有些用户天生愿意给别人好评，心慈手软，比较好说话，而有的人就比较苛刻，总是评分不超过3分（5分满分）。
# 一些物品的评分普遍高于其他物品，一些物品的评分普遍低于其他物品。
# 比如：一些物品一被生产便决定了它的地位，有的比较受人们欢迎，有的则被人嫌弃。

# 这个用户或物品普遍高于或低于平均值的差值，我们称为偏置(bias)。

# Baseline目标：
# 找出每个用户普遍高于或低于他用户的偏置值bu
# 找出每件物品普遍高于或低于其他物品的偏置值bi
# 我们的目标也就转化为寻找最优的bu和bi

# 使用Baseline的算法思想预测评分的步骤如下：
# 计算所有电影的平均评分μ（即全局平均评分）
# 计算每个用户评分与平均评分μ的偏置值bu
# 计算每部电影所接受的评分与平均评分μ的偏置值bi

# 预测用户对电影的评分：
# r^ui=bui=μ+bu+bi

# 举例：通过Baseline来预测用户A对电影"阿甘正传"的评分
# 首先计算出整个评分数据集的平均评分μ是3.5分
# 用户A比较苛刻，普遍比平均评分低0.5分，即用户A的偏置值bu是-0.5；
# "阿甘正传"比较热门且备受好评，评分普遍比平均评分要高1.2分，"阿甘正传"的偏置是bi是+1.2
# 因此就可以预测出用户A对电影"阿甘正传"的评分为：3.5+(-0.5)+1.2，也就是4.2分。

# 对于所有电影的平均评分是直接能计算出的，因此问题在于要测出每个用户的评分偏置和每部电影的得分偏置。

# 数据加载
import pandas as pd
import numpy as np
dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
dataset = pd.read_csv("./data/ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))
print("加载数据：")
print(dataset.head(5))

# 数据初始化
# 我们最终计算的目的是为了找到所有用户评分偏置的bu 和所有物品的得分偏置bi
# 初始化一个dict bu，元素个数=用户数量，每一个元素为{userId : 用户评分偏置} 偏置初始值为0
# 初始化一个dict bi，元素个数=电影数量，每一个元素为{movieId : 物品得分偏置} 偏置初始值为0

# 用户评分数据  groupby分组  groupby('userId'): 根据用户id分组 agg（aggregation聚合）
users_ratings = dataset.groupby('userId').agg(list)
print('用户评分数据：')
print(users_ratings.head(5))
# 物品评分数据
items_ratings = dataset.groupby('movieId').agg(list)
print('物品评分数据：')
print(items_ratings.head(5))
# 计算全局平均分
global_mean = dataset['rating'].mean()
print('全局平均分：')
print(global_mean)
# 初始化bu和bi
bu = dict(zip(users_ratings.index, np.zeros(len(users_ratings))))
print('bu：')
print(bu)
bi = dict(zip(items_ratings.index, np.zeros(len(items_ratings))))
print('bi：')
print(bi)

# 更新bu和bi
# 初始的bu 和 bi 均为0 ，每一次迭代都会更新一次bu和bi
# 根据梯度下降的原理，只要学习率选择的合适，每更新一次bu、bi，预测的损失就会降低一些，直至找到损失函数的极值点

# number_epochs：迭代次数 alpha：学习率  reg：正则化系数
# number_epochs = 30
# alpha = 0.1
# reg = 1
# for i in range(number_epochs):
#     print("iter%d" % i)
#     for uid, iid, real_rating in dataset.itertuples(index=False):
#         error = real_rating - (global_mean + bu[uid] + bi[iid])
#         bu[uid] += alpha * (error - reg * bu[uid])
#         bi[iid] += alpha * (error - reg * bi[iid])

# 预测评分
# 利用更新完毕的bu和bi 进行评分预测
def predict(uid, iid):
    predict_rating = global_mean + bu[uid] + bi[iid]
    return predict_rating

# 整体封装（参考）
class BaselineCFBySGD(object):
    def __init__(self, number_epochs, alpha, reg, columns=["uid", "iid", "rating"]):
        # 梯度下降最高迭代次数
        self.number_epochs = number_epochs
        # 学习率
        self.alpha = alpha
        # 正则参数
        self.reg = reg
        # 数据集中user-item-rating字段的名称
        self.columns = columns

    def fit(self, dataset):
        '''
        dataset: uid, iid, rating
        '''
        self.dataset = dataset
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg(list)
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg(list)
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 调用sgd方法训练模型参数
        self.bu, self.bi = self.sgd()

    def sgd(self):
        '''
        利用随机梯度下降，优化bu，bi的值
        return: bu, bi
        '''
        # 初始化bu、bi的值，全部设为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

        for i in range(self.number_epochs):
            print("iter%d" % i)
            for uid, iid, real_rating in self.dataset.itertuples(index=False):
                error = real_rating - (self.global_mean + bu[uid] + bi[iid])

                bu[uid] += self.alpha * (error - self.reg * bu[uid])
                bi[iid] += self.alpha * (error - self.reg * bi[iid])

        return bu, bi

    def predict(self, uid, iid):
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating


if __name__ == '__main__':
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    dataset = pd.read_csv("./data/ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))

    bcf = BaselineCFBySGD(20, 0.1, 0.1, ["userId", "movieId", "rating"])
    bcf.fit(dataset)

    while True:
        uid = int(input("uid: "))
        iid = int(input("iid: "))
        print(bcf.predict(uid, iid))


# 准确性指标评估
# 训练集、测试集划分
def data_split(data_path, x=0.8, random=False):
    '''
    切分数据集， 这里为了保证用户数量保持不变，将每个用户的评分数据按比例进行拆分
    data_path: 数据集路径
    x: 训练集的比例，如x=0.8，则0.2是测试集
    random: 是否随机切分，默认False
    return: 用户-物品评分矩阵
    '''
    print("开始切分数据集...")
    # 设置要加载的数据字段的类型
    dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
    # 加载数据，我们只用前三列数据，分别是用户ID，电影ID，已经用户对电影的对应评分
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))

    trainset_index = []
    # 为了保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby("userId").any().index:
        user_rating_data = ratings.where(ratings["userId"]==uid).dropna()
        if random:
            # 因为不可变类型不能被 shuffle方法作用，所以需要强行转换为列表
            index = list(user_rating_data.index)
            np.random.shuffle(index)    # 打乱列表
            _index = round(len(user_rating_data) * x)
            trainset_index += index[:_index]
        else:
            # 将每个用户的x比例的数据作为训练集，剩余的作为测试集
            index = list(user_rating_data.index)
            _index = round(len(user_rating_data) * x)
            trainset_index += index[:_index]

    trainset = ratings.loc[trainset_index]
    testset = ratings.drop(trainset_index)
    print("完成数据集切分...")
    return trainset, testset

# 创建accuary方法计算准确性指标
def accuray(predict_results, method="all"):
    '''
    准确性指标计算方法
    predict_results: 预测结果，类型为容器，每个元素是一个包含uid,iid,real_rating,pred_rating的序列
    method: 指标方法，类型为字符串，rmse或mae，否则返回两者rmse和mae
    '''

    def rmse(predict_results):
        '''
        rmse评估指标
        predict_results:
        return: rmse
        '''
        length = 0
        _rmse_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
        return round(np.sqrt(_rmse_sum / length), 4)

    def mae(predict_results):
        '''
        mae评估指标
        predict_results:
        return: mae
        '''
        length = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _mae_sum += abs(pred_rating - real_rating)
        return round(_mae_sum / length, 4)

    def rmse_mae(predict_results):
        '''
        rmse和mae评估指标
        param predict_results:
        return: rmse, mae
        '''
        length = 0
        _rmse_sum = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
            _mae_sum += abs(pred_rating - real_rating)
        return round(np.sqrt(_rmse_sum / length), 4), round(_mae_sum / length, 4)

    if method.lower() == "rmse":
        return rmse(predict_results)
    elif method.lower() == "mae":
        return mae(predict_results)
    else:
        return rmse_mae(predict_results)

# 添加test方法， 利用训练集评估训练结果
def test(self,testset):
    '''预测测试集数据'''
    for uid, iid, real_rating in testset.itertuples(index=False):
        try:
            pred_rating = self.predict(uid, iid)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, real_rating, pred_rating

# 整体封装（参考）
class BaselineCFBySGD2(object):
    def __init__(self, number_epochs, alpha, reg, columns=["uid", "iid", "rating"]):
        # 梯度下降最高迭代次数
        self.number_epochs = number_epochs
        # 学习率
        self.alpha = alpha
        # 正则参数
        self.reg = reg
        # 数据集中user-item-rating字段的名称
        self.columns = columns

    def fit(self, dataset):
        '''
        :param dataset: uid, iid, rating
        :return:
        '''
        self.dataset = dataset
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg(list)
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg(list)
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 调用sgd方法训练模型参数
        self.bu, self.bi = self.sgd()

    def sgd(self):
        '''
        利用随机梯度下降，优化bu，bi的值
        return: bu, bi
        '''
        # 初始化bu、bi的值，全部设为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

        for i in range(self.number_epochs):
            print("iter%d" % i)
            for uid, iid, real_rating in self.dataset.itertuples(index=False):
                error = real_rating - (self.global_mean + bu[uid] + bi[iid])

                bu[uid] += self.alpha * (error - self.reg * bu[uid])
                bi[iid] += self.alpha * (error - self.reg * bi[iid])

        return bu, bi

    def predict(self, uid, iid):
        '''评分预测'''
        if iid not in self.items_ratings.index:
            raise Exception("无法预测用户<{uid}>对电影<{iid}>的评分，因为训练集中缺失<{iid}>的数据".format(uid=uid, iid=iid))

        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating

    def test(self,testset):
        '''预测测试集数据'''
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating

if __name__ == '__main__':
    # 数据集划分
    trainset, testset = data_split("./data/ml-latest-small/ratings.csv", random=True)
    # 创建模型并训练
    bcf = BaselineCFBySGD2(20, 0.1, 0.1, ["userId", "movieId", "rating"])
    bcf.fit(trainset)
    # 模型预测
    pred_results = bcf.test(testset)
    # 准确性评估
    rmse, mae = accuray(pred_results)
    print("rmse: ", rmse, "mae: ", mae)












