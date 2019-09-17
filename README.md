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
│  ├─ CASIA_database  // 语音情感数据集，用于训练和测试
│  ├─ predictsets     // 自己录的音频，用于预测
│  └─ 16k.wav         // 标准音频，用于识别和测试程序
├─ CoreSource  
│  ├─ SpeechBasedEmotionRec
│  |  ├─ Models                   // 训练好的模型
│  |  ├─ Features                 // 从音频中提取出来的特征
│  |  ├─ Config.py                // Opensmile的参数配置
│  |  ├─ Common_Model.py          // 模型的父类
│  |  ├─ DNN_Model.py             // lstm模型
│  |  ├─ Opensmile_Feature.py     // 利用Opensmile提取特征
│  |  ├─ Utils.py                 // 输出工具
│  |  ├─ machineLearning.py       // 模型训练和数据准备
│  |  ├─ speechFeature.py         // librosa提取特征
|  |  └─ speechFeatureNew.py      // 自己实现mfcc
│  ├─ text emotion                // 文字语音识别
│  ├─ spectrogram.py              // 提取语谱图并保存
│  ├─ speechRec.py                // 语音识别
│  └─ wordSeg.py                  // 分词检索
├─ venv                     
├─ starter.py         // 语音识别和分词检索启动
└─ ml_starter.py      // 情感识别启动
```
