import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import gc
import warnings
warnings.filterwarnings('ignore')

#feature_columns = ['f%s' % i for i in range(1, 298)]


train = pd.read_csv('../input/atec-anti-fraud/atec_anti_fraud_train.csv')
train['label'] = train['label'].replace([-1], [1])
test = pd.read_csv('../input/atec-anti-fraud/atec_anti_fraud_test_a.csv')
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


TR = train[train['date']<20171100]
VA = train[train['date']>20171100]
TR.reset_index(drop=True, inplace=True)

X = TR.drop(['id', 'label', 'date'], axis=1)
Y = TR['label']

X_test = test.drop(['id', 'date'], axis=1)
#Y_test = TS['label']
X_valid = VA.drop(['id', 'date', 'label'], axis=1)
Y_valid = VA['label']
submission = test[['id']]

#del TR, VA, test, train
#gc.collect()

def scorer(y, pred):
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    score = 0.4 * tpr[np.where(fpr>=0.001)[0][0]] + \
            0.3 * tpr[np.where(fpr>=0.005)[0][0]] + \
            0.3 * tpr[np.where(fpr>=0.01)[0][0]]
    print('-----------------------------result------------------------')
    print('fpr_0.001: {0} | fpr_0.005: {1} | fpr_0.01: {2}'.format(tpr[np.where(fpr>=0.001)[0][0]], 
                                   tpr[np.where(fpr>=0.005)[0][0]], 
                                   tpr[np.where(fpr>=0.01)[0][0]]))
    print('score : {}'.format(score))
    return score

NFOLDS = 5
kf = StratifiedKFold(n_splits=NFOLDS, random_state=10, shuffle=True)
def get_oof():
    oof_test = np.zeros((X_test.shape[0],))
    oof_test_skf = np.empty((NFOLDS, X_test.shape[0]))

    for i, (train_index, test_index) in enumerate(kf.split(TR, Y)):
        train_temp = TR.iloc[train_index]
        pos = train_temp[train_temp['label'] == 1]
        train_temp = pd.concat([train_temp, pos], ignore_index=True)
        x_tr = train_temp.drop(['id', 'date', 'label'], axis=1).values
        y_tr = train_temp['label'].values
        #x_tr = X.values[train_index]
        #y_tr = Y.values[train_index]
        x_te = X.values[test_index]
        y_te = Y.values[test_index]
        model = xgb.XGBClassifier(max_depth=7, 
                                  learning_rate=0.07,
                                  n_estimators=200, #928
                                  silent=True, 
                                  objective='binary:logistic', 
                                  booster='gbtree', 
                                  n_jobs=-1, 
                                  gamma=3.8111289765374132e-05, 
                                  min_child_weight=300, #22
                                  max_delta_step=4, 
                                  subsample=0.8, #0.65
                                  colsample_bytree=0.7,#0.5
                                  colsample_bylevel=0.8, 
                                  scale_pos_weight=1, 
                                  random_state=10, 
                                  eval_metric ='auc',
                                  tree_method='auto')

        model.fit(x_tr, y_tr, eval_set=[(x_te, y_te)], eval_metric='auc', early_stopping_rounds=50)
        oof_test_skf[i, :] = model.predict_proba(X_test.values)[:, 1]
        pred = model.predict_proba(X_valid)[:, 1]
        scorer(Y_valid, pred)
        oof_test_skf[i, :] = model.predict_proba(X_test)[:, 1]
        del pred, model
        gc.collect()
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_test.reshape(-1, 1)

oof_test = get_oof() 
submission['score'] = oof_test
submission.to_csv('xgb_201806071421.csv', index=False)   
