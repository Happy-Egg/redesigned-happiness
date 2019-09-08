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
            wordList = line.strip().split(' ')#每一行去掉换行后分割
            vecs = getWordVecs(wordList, model)#vecs为嵌套列表，每个列表元素为每个分词的词向量
            if len(vecs) > 0:
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
 
Y = np.concatenate( ( np.ones(len(posInput)), np.zeros(len(negInput)) ) )
 
#这里也可以用np.concatenate将posInput和negInput进行合并
X = posInput[:]
for neg in negInput:
    X.append(neg)
X = np.array(X)
df_x = pd.DataFrame(X)
df_y = pd.DataFrame(Y)
data = pd.concat( [df_y, df_x], axis=1 )
data.to_csv('data/ChnSentiCorp_htl_ba_2000/2000_data.csv')