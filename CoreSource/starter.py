# coding=utf-8

from CoreSource import speechRec

filePath = 'C:/Users/JyunmauChan/Music/课程设计测试音频/16k.wav'
fileFormat = 'wav'

textU8 = speechRec.do_tts(filePath, fileFormat)
print(textU8)
