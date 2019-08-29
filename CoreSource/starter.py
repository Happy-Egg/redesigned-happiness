# coding=utf-8

from CoreSource import speechRec
from CoreSource import wordSeg

filePath = 'C:/Users/JyunmauChan/Music/课程设计测试音频/16k.wav'
fileFormat = 'wav'

textU8 = speechRec.do_tts(filePath, fileFormat)

segList = wordSeg.do_cut(textU8)
wordSeg.do_search(segList, '北京')
