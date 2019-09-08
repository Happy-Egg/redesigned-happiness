import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
import warnings

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
 
print ('Test Accuracy: %.2f'% clf.score(x_pca,y))
#Create ROC curve
pred_probas = clf.predict_proba(x_pca)[:,1] #score
 
fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.show()