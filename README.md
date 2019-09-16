# redesigned-happiness
A curriculum design project from NJUST, include speech-recognition, word-segmentation &amp; emotion-recognition.

## 功能和使用说明
- speechRec模块：调用百度API实现语音识别
- wordSeg模块：调用jieba分词实现分词检索

在项目根目录下的`starter.py`是主程序，默认调取`AudioFiles`目录下的一个音频文件，其录音内容是“北京科技馆”，默认音频类型为“wav”，默认搜索词为“北京”。

使用本项目，在命令行下切换至项目根目录，命令`python stater.py`将会以上述默认参数调用有关模块。可选参数列表及其功能如下：

- `-p`或`path`：要分析的音频文件路径
- `-f`或`format`：音频文件格式
- `-w`或`word`：要检索的词语

调用示例如下：`python starter.py -p AudioFiles/16k.wav -f wav -w 北京`

有关模块所能处理的音频和分词情况，参考模块说明。

## 功能模块说明

- [speechRec模块](https://github.com/Happy-Egg/redesigned-happiness/wiki/speechRec%E6%A8%A1%E5%9D%97)
- [wordSeg模块](https://github.com/Happy-Egg/redesigned-happiness/wiki/wordSeg%E6%A8%A1%E5%9D%97)

## 项目目录
```
├─ AudioFiles  
│  └─ 16k.wav  
├─ CoreSource  
│  ├─ speechRec.py  
│  └─ wordSeg.py  
├─ venv  
├─ starter.py  
```
