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


df = pd.read_csv('data/ChnSentiCorp_htl_ba_2000/2000_data.csv')
y = df.iloc[:, 1]#第一列是索引，第二列是标签
x = df.iloc[:, 2:]#第三列之后是400维的词向量
##根据图形取100维
warnings.filterwarnings('ignore')
x_pca = PCA(n_components = 100).fit_transform(x)
 
# SVM (RBF)
# using training data with 100 dimensions
 
# clf = svm.SVC(C = 2, probability = True)
# #高于默认值，弱泛化
# clf.fit(x_pca,y)
 
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
 
X_train, X_test, y_train, y_test = train_test_split( x_pca, y, test_size=0.25, random_state=0)
 
param_grid = {'C': [0.01,0.1, 1, 10, 100, 1000,],'penalty': [ 'l1', 'l2']}
grid_search = GridSearchCV(LogisticRegression(),  param_grid, cv=10)
grid_search.fit( X_train,y_train )
print( grid_search.best_params_, grid_search.best_score_ )
  
#预测拆分的test
LR = LogisticRegression( C=grid_search.best_params_['C'], penalty=grid_search.best_params_['penalty'] )
LR.fit( X_train,y_train )
lr_y_predict = LR.predict(X_test)
print(accuracy_score(y_test, lr_y_predict))
print('使用LR进行分类的报告结果：\n')
print(classification_report(y_test, lr_y_predict))
print( "AUC值:",roc_auc_score( y_test, lr_y_predict ) )
print('Show The Roc Curve：')
show_roc(LR, X_test, y_test)