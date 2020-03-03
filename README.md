# redesigned-happiness
A curriculum design project from NJUST, include speech-recognition, word-segmentation &amp; emotion-recognition.

## license
- [see LICENSE](https://github.com/Happy-Egg/redesigned-happiness/blob/master/LICENSE)

## 环境
- 使用python3.6作为底包
- 在项目根目录下打开终端使用命令`pip install -r requirements.txt`安装依赖

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
│  ├─ text emotion
│  |  ├─ data
│  |  |  ├─ ChnSentiCorp_htl_ba_2000
│  │  |  |  ├─ 2000_data.csv      // 句向量
│  │  |  |  ├─ 2000_neg_cut.txt   // 正向文本分词结果   
│  │  |  |  └─ 2000_pos_cut.txt   // 负向文本分词结果  
│  |  |  ├─ model
│  │  |  |  ├─ LR_model.m         // LR模型文件
│  │  |  |  └─ SVM_model.m        // SVM模型文件
│  |  |  └─ stopWord.txt          // 中文停用词
│  |  ├─ The_cut.py               // 去除字符、停用词
│  |  ├─ The_text_vector.py       // 抽取句子特征得到句向量
│  |  ├─ dimension_cut_test.py    // 对维度进行测试
│  |  ├─ merge.py                 // 将多条语句整合为一个文本
│  |  ├─ model_LR.py              // LR模型
│  |  └─ model_SVM.py             // SVM模型
├─ venv                     
├─ starter.py         // 语音识别和分词检索启动
├─ requirements.txt   // 环境依赖
└─ ml_starter.py      // 情感识别启动
```
