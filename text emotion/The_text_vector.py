import os
import sys
import logging
import gensim
import codecs
import numpy as np
import pandas as pd
 
def getWordVecs(wordList, model):
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
 
 
def buildVecs(filename, model):
    fileVecs = []
    with codecs.open(filename, 'r', encoding='gbk') as contents:
        for line in contents:
            logger.info('Start line: ' + line )
            #每一行去掉换行后分割
            wordList = line.strip().split(' ')
            #vecs为嵌套列表，每个列表元素为每个分词的词向量
            vecs = getWordVecs(wordList, model)
            if len(vecs) > 0:
                #用矩阵均值作为当前语句的特征词向量
                vecsArray = sum(np.array(vecs)) / len (vecs)
                fileVecs.append(vecsArray)
    return fileVecs
 
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
 
inp = 'data/ChnSentiCorp_htl_ba_2000/wiki.zh.text.vector'
model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
 
posInput = buildVecs('data/ChnSentiCorp_htl_ba_2000/2000_pos_cut.txt', model)
negInput = buildVecs('data/ChnSentiCorp_htl_ba_2000/2000_neg_cut.txt', model)

#总标签 
Y = np.concatenate( ( np.ones(len(posInput)), np.zeros(len(negInput)) ) )
#总分词向量
X = np.concatenate( (posInput , negInput) )

#数据形式转换
X = np.array(X)
df_x = pd.DataFrame(X)
df_y = pd.DataFrame(Y)
#标签与分词向量合并
data = pd.concat( [df_y, df_x], axis=1 )
data.to_csv('data/ChnSentiCorp_htl_ba_2000/2000_data.csv')