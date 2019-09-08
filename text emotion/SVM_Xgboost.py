import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
 

df = pd.read_csv('data/ChnSentiCorp_htl_ba_2000/2000_data.csv')
y = df.iloc[:, 1]#第一列是索引，第二列是标签
x = df.iloc[:, 2:]#第三列之后是400维的词向量
##根据图形取100维
warnings.filterwarnings('ignore')
x_pca = PCA(n_components = 100).fit_transform(x)
 
# SVM (RBF)
# using training data with 100 dimensions
 
clf = svm.SVC(C = 2, probability = True)
clf.fit(x_pca,y)

def show_roc(model, X, y):
    #Create ROC curve
    pred_probas = model.predict_proba(X)[:,1]#score
 
    fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
    roc_auc = metrics.auc(fpr,tpr)
    plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc = 'lower right')
    plt.show()

def modelfit(alg, dtrain_x, dtrain_y, useTrainCV=True, cv_flods=5, early_stopping_rounds=50):
    """
    :param alg: 初始模型
    :param dtrain_x:训练数据X
    :param dtrain_y:训练数据y（label）
    :param useTrainCV: 是否使用cv函数来确定最佳n_estimators
    :param cv_flods:交叉验证的cv数
    :param early_stopping_rounds:在该数迭代次数之前，eval_metric都没有提升的话则停止
    """
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain_x, dtrain_y)
        print(alg.get_params()['n_estimators'])
 
        cv_result = xgb.cv( xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'],
                        nfold=cv_flods, metrics = 'auc', early_stopping_rounds = early_stopping_rounds )
         
        print('useTrainCV\n',cv_result)
        print('Total estimators:',cv_result.shape[0])
        alg.set_params(n_estimators=cv_result.shape[0])
 
    # train data
    alg.fit(dtrain_x, dtrain_y, eval_metric='auc')
 
    #predict train data
    train_y_pre = alg.predict(dtrain_x)
    print ("\nModel Report")
    print ("Accuracy : %.4g" % accuracy_score( dtrain_y, train_y_pre) )
     
    return cv_result.shape[0]
     
 
#XGBoost调参
def xgboost_change_param(train_X, train_y):
 
    print('######Xgboost调参######')
    print('\n step1 确定学习速率和迭代次数n_estimators')
 
    xgb1 = XGBClassifier(learning_rate=0.1, booster='gbtree', n_estimators=1000,
                         max_depth=4, min_child_weight=1,
                         gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective='binary:logistic',scale_pos_weight=1, seed=10)
 
    #useTrainCV=True时，最佳n_estimators=23, learning_rate=0.1
    NEstimators = modelfit(xgb1, train_X, train_y, early_stopping_rounds=45)
 
    print('\n step2 调试参数min_child_weight以及max_depth')
    param_test1 = { 'max_depth' : range(3, 8, 1), 'min_child_weight' : range(1, 6, 2) }
 
    #GridSearchCV()中的estimator参数所使用的分类器
    #并且传入除需要确定最佳的参数之外的其他参数
    #每一个分类器都需要一个scoring参数，或者score方法
    gsearch1 = GridSearchCV( estimator=XGBClassifier( learning_rate=0.1,n_estimators=NEstimators,
                                                     gamma=0,subsample=0.8,colsample_bytree=0.8,
                                                     objective='binary:logistic',scale_pos_weight=1,seed=10 ),
                            param_grid=param_test1,scoring='roc_auc', cv=5)
 
    gsearch1.fit(train_X,train_y)
    #最佳max_depth = 4 min_child_weight=1
    print(gsearch1.best_params_, gsearch1.best_score_)
    MCW = gsearch1.best_params_['min_child_weight']
    MD = gsearch1.best_params_['max_depth']
     
    print('\n step3 gamma参数调优')
    param_test2 = { 'gamma': [i/10.0 for i in range(0,5)] }
     
    gsearch2 = GridSearchCV( estimator=XGBClassifier(learning_rate=0.1,n_estimators=NEstimators,
                                                    max_depth=MD, min_child_weight=MCW,
                                                    subsample=0.8,colsample_bytree=0.8,
                                                    objective='binary:logistic',scale_pos_weight=1,seed=10),
                            param_grid=param_test2,scoring='roc_auc',cv=5 )
 
    gsearch2.fit(train_X, train_y)
    #最佳 gamma = 0.0
    print(gsearch2.best_params_, gsearch2.best_score_)
    GA = gsearch2.best_params_['gamma']
     
    print('\n step4 调整subsample 和 colsample_bytrees参数')
    param_test3 = { 'subsample': [i/10.0 for i in range(6,10)],
                   'colsample_bytree': [i/10.0 for i in range(6,10)] }
 
    gsearch3 = GridSearchCV( estimator=XGBClassifier(learning_rate=0.1,n_estimators=NEstimators,
                                                     max_depth=MD,min_child_weight=MCW,gamma=GA,
                                                     objective='binary:logistic',scale_pos_weight=1,seed=10),
                            param_grid=param_test3,scoring='roc_auc',cv=5 )
 
    gsearch3.fit(train_X, train_y)
    #最佳'subsample': 0.8, 'colsample_bytree': 0.8
    print(gsearch3.best_params_, gsearch3.best_score_)
    SS = gsearch3.best_params_['subsample']
    CB = gsearch3.best_params_['colsample_bytree']
     
    print('\nstep5 正则化参数调优')
    param_test4= { 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100] }
    gsearch4= GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,n_estimators=NEstimators,
                                                   max_depth=MD,min_child_weight=MCW,gamma=GA,
                                                   subsample=SS,colsample_bytree=CB,
                                                   objective='binary:logistic',
                                                   scale_pos_weight=1,seed=10),
                           param_grid=param_test4,scoring='roc_auc',cv=5 )
    gsearch4.fit(train_X, train_y)
    #reg_alpha:1e-5
    print(gsearch4.best_params_, gsearch4.best_score_)
    RA = gsearch4.best_params_['reg_alpha']
     
    param_test5 ={ 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100] }
    gsearch5= GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,n_estimators=NEstimators,
                                                   max_depth=MD,min_child_weight=MCW,gamma=GA,
                                                   subsample=SS,colsample_bytree=CB,
                                                   objective='binary:logistic',reg_alpha=RA,
                                                   scale_pos_weight=1,seed=10),
                           param_grid=param_test5,scoring='roc_auc',cv=5)
 
    gsearch5.fit(train_X, train_y)
    #reg_lambda:1
    print(gsearch5.best_params_, gsearch5.best_score_)
    RL = gsearch5.best_params_['reg_lambda']
     
    return NEstimators, MD, MCW, GA, SS, CB, RA, RL
 
# XGBoost调参
X_train, X_test, y_train, y_test = train_test_split( x_pca, y, test_size=0.25, random_state=0)
X_train = np.array(X_train)
#返回最佳参数
NEstimators, MD, MCW, GA, SS, CB, RA, RL = xgboost_change_param(X_train, y_train)
 
#parameters at last
print( '\nNow we use the best parasms to fit and predict:\n' )
print( 'n_estimators = ', NEstimators)
print( 'max_depth = ', MD)
print( 'min_child_weight = ', MCW)
print( 'gamma = ', GA)
print( 'subsample = ', SS)
print( 'colsample_bytree = ', CB)
print( 'reg_alpha = ', RA)
print( 'reg_lambda = ', RL)
xgb1 = XGBClassifier(learning_rate=0.1,n_estimators=NEstimators,max_depth=MD,min_child_weight=MCW,
                     gamma=GA,subsample=SS,colsample_bytree=CB,objective='binary:logistic',reg_alpha=RA,reg_lambda=RL,
                     scale_pos_weight=1,seed=10)
xgb1.fit(X_train, y_train)
xgb_test_pred = xgb1.predict( np.array(X_test) )
print ("The xgboost model Accuracy : %.4g" % accuracy_score(y_pred=xgb_test_pred, y_true=y_test))
print('使用Xgboost进行分类的报告结果：\n')
print( "AUC值:",roc_auc_score( y_test, xgb_test_pred ) )
print( classification_report(y_test, xgb_test_pred) )
print('Show The Roc Curve：')
show_roc(xgb1, X_test, y_test)