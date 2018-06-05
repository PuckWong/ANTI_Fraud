import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from sklearn.metrics import log_loss
#pip install MLFeatureSelection
from MLFeatureSelection import FeatureSelection as FS
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

feature_columns = ['f%s' % i for i in range(1, 298)]
dtype = {}
for i in feature_columns:
    if i not in ['f5', 'f82', 'f83', 'f84', 'f85', 'f86']:
        dtype[i] = 'float16'
dtype.update({'f5':'float32', 'f82':'float32', 'f83':'float32', 
              'f84':'float32', 'f85':'float32', 'f86':'float32', 'id':'str', })

train = pd.read_csv('../input/atec-anti-fraud/atec_anti_fraud_train.csv', dtype=dtype)
train['label'] = train['label'].replace([-1], [1])
#test = pd.read_csv('../input/atec-anti-fraud/atec_anti_fraud_test_a.csv', dtype=dtype)

train.fillna(0, inplace=True)
#test.fillna(0, inplace=True)
subset = train.sample(500000, random_state=10)
subset.reset_index(drop=True, inplace=True)

def score(pred, real): #评分
    return log_loss(pred, real)

df = subset

def validation(X, Y, features, clf, lossfunction):
    totaltest = []
    kf = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X.ix[train_index,:][features], X.ix[test_index,:][features]
        y_train, y_test = Y[train_index], Y[test_index]
        #clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=50)
        clf.fit(X_train, y_train)
        totaltest.append(lossfunction(y_test, clf.predict(X_test)))
    return np.mean(totaltest)

def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

CrossMethod = {'+':add,
               '-':substract,
               '*':times,
               '/':divide,}

sf = FS.Select(Sequence = True, Random = False, Cross = False) #初始化选择器，选择你需要的流程
sf.ImportDF(df, label ='label') #导入数据集以及目标标签
#sf.ImportCrossMethod(CrossMethod)
sf.ImportLossFunction(score, direction = 'descend') #导入评价函数以及优化方向
sf.InitialNonTrainableFeatures(['id','date', 'label']) #初始化不能用的特征
sf.InitialFeatures(feature_columns) #初始化其实特征组合
sf.GenerateCol() #生成特征库 （具体该函数变量请参考根目录下的readme）
sf.SetSample(1, samplemode = 1) #初始化抽样比例和随机过程
sf.SetTimeLimit(240) #设置算法运行最长时间，以分钟为单位
sf.clf = lgb.LGBMClassifier(random_state=10, num_leaves =15, n_estimators=200, max_depth=5, learning_rate = 0.1, n_jobs=-1) #设定模型
sf.SetLogFile('record.log') #初始化日志文件
sf.run(validation) #输入检验函数并开始运行