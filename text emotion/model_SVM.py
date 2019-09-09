import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

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

def save_model(X_train , y_train , model_path):
    clf = svm.SVC(C = 2, probability = True)
    clf.fit(X_train,y_train)
    joblib.dump(clf, model_path)

df = pd.read_csv('data/ChnSentiCorp_htl_ba_2000/2000_data.csv')
#第一列是索引，第二列是标签
y = df.iloc[:, 1]
#第三列之后是400维的词向量
x = df.iloc[:, 2:]

warnings.filterwarnings('ignore')
##根据图形取100维
x_pca = PCA(n_components = 100).fit_transform(x)

#分类测试集与数据集
X_train, X_test, y_train, y_test = train_test_split( x_pca, y, test_size=0.25, random_state=0) 

model_path="data/model/SVM_model.m"
#仅在第一次使用保存模型 
#save_model(X_train , y_train , model_path)

model = joblib.load(model_path)
y_predict = model.predict(X_test)

print(accuracy_score(y_test, y_predict))
print('使用SVM进行分类的报告结果：\n')
print(classification_report(y_test, y_predict))
print( "AUC值:",roc_auc_score( y_test, y_predict ) )
print('Show The Roc Curve：')
show_roc(model, X_test, y_test)