import jieba
import os
import codecs
import re
 
def prepareData(sourceFile, targetFile):
    f =codecs.open(sourceFile, 'r', encoding='gbk')
    target = codecs.open(targetFile, 'w', encoding='gbk')
    print( 'open source file: '+ sourceFile )
    print( 'open target file: '+ targetFile )
     
    lineNum = 0
    for eachline in f:
        lineNum += 1
        print('---processing ', sourceFile, lineNum,' article---')
        eachline = clearTxt(eachline)
        #print( eachline )
        seg_line = sent2word(eachline)
        #print(seg_line)
        target.write(seg_line + '\n')
    print('---Well Done!!!---' * 4)
    f.close()
    target.close()
         
         
#文本清洗
def clearTxt(line):
    if line != '':
        line = line.strip()
        #去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]","",line)
         
        #去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
        return line
    else:
        return 'Empyt Line'
 
#文本切割，并去除停顿词
def sent2word(line):
     
    segList = jieba.cut(line, cut_all=False)
    segSentence = ''
    for word in segList:
        if word != '\t' and ( word not in stopwords ):
            segSentence += ( word + " " )
    return segSentence.strip()
     
inp = 'data/ChnSentiCorp_htl_ba_2000'
stopwords = [ w.strip() for w in codecs.open('data/stopWord.txt', 'r', encoding='utf-8') ]
 
folders = ['neg', 'pos']
for folder in folders:
    sourceFile = '2000_{}.txt'.format(folder)
    targetFile = '2000_{}_cut.txt'.format(folder)
    prepareData( os.path.join(inp, sourceFile), os.path.join(inp,targetFile) )