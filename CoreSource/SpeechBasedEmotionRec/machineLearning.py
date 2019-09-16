# coding=<encoding name>
from sklearn import svm
import joblib
import glob
from random import shuffle
from CoreSource.SpeechBasedEmotionRec import speechFeature as sf
import numpy as np
from keras.utils import np_utils
import os

from CoreSource.SpeechBasedEmotionRec.DNN_Model import LSTM_Model
from CoreSource.SpeechBasedEmotionRec.Utils import load_model, Radar, playAudio

import CoreSource.SpeechBasedEmotionRec.Opensmile_Feature as of
from CoreSource.SpeechBasedEmotionRec.Config import Config

EMOTION_LABEL = {'angry': '1', 'fear': '2', 'happy': '3', 'neutral': '4', 'sad': '5', 'surprise': '6'}
EMOTION_LABEL1 = {'1': 'angry', '2': 'fear', '3': 'happy', '4': 'neutral', '5': 'sad', '6': 'surprise'}


def all_file_path(path='../../AudioFiles/CASIA_database/', file_type='wav'):
    """获取文件夹下所有音频文件的路径(乱序)
        Args:
            path: 文件夹路径，默认是项目中的数据集
            file_type: 文件类型，默认是wav
        Returns:
            打乱后的文件列表
            example：
            [path1/file1.wav, path2/file2.wav]
    """
    read_files = glob.glob(path + '*/*/*' + file_type)
    shuffle(read_files)
    return read_files


def get_data_l(files, feature_model='all', split_ratio=0.9):
    """生成特征数据集并划分训练集
    Args:
        files: list,每个元素是一个文件路径
        feature_model: 特征提取的模式，'all'或'mfcc'
            all （默认）包含梅尔倒频谱系数、短时过零率和均方根能量
            mfcc 仅提取梅尔倒频谱系数
        split_ratio: 训练集占所有数据集文件的比例，默认0.9
    Returns:
        train_data: 训练集，二维矩阵的每一行代表一条语音的特征向量
        train_label: 训练集标签，与训练集向量顺序同，用1~6代表6类情感
        test_data: 测试集，二维矩阵的每一行代表一条语音的特征向量
        test_label: 测试集标签，与训练集向量顺序同，用1~6代表6类情感
    """
    data_feature = []
    data_labels = []
    for f in files:
        if feature_model == 'all':
            data_feature.append(sf.get_all_feature(f))
        elif feature_model == 'mfcc':
            data_feature.append(sf.get_mfcc(f))
        else:
            print('feature_model error !')
            return
        data_labels.append(int(EMOTION_LABEL[f.split('\\')[-2]]))
    data_feature = np.array(data_feature)
    data_labels = np.array(data_labels)
    split_num = int(len(files) * split_ratio)
    train_data = data_feature[:split_num, :]
    train_label = data_labels[:split_num]
    test_data = data_feature[split_num:, :]
    test_label = data_labels[split_num:]
    return train_data, train_label, test_data, test_label


def svm_train(train_data, train_label, test_data, test_label):
    """SVM训练模型
        Args:
            train_data: 训练集，二维矩阵的每一行代表一条语音的特征向量
            train_label: 训练集标签，与训练集向量顺序同，用1~6代表6类情感
            test_data: 测试集，二维矩阵的每一行代表一条语音的特征向量
            test_label: 测试集标签，与训练集向量顺序同，用1~6代表6类情感
        Returns:
            训练好的模型
    """
    print('>> ==================== Train Start =================== <<')
    clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=19, gamma=0.0001, probability=True)
    clf.fit(train_data, train_label)
    acc = clf.score(test_data, test_label)
    joblib.dump(clf, "svm_model.m")
    print('>>==================== Train Over ===================<<')
    print("测试集正确率为：" + "{:.2f}".format(acc * 100) + "%")
    return clf


def svm_predict(model: str, file_path: str):
    """lstm预测音频情感
        Args:
            model: 已加载或训练的模型
            file_path: 要预测的文件路径
        Returns：
            预测结果和置信概率
    """
    test_feature = []
    test_feature.append(sf.get_all_feature(file_path))
    clf = joblib.load(model)
    result = clf.predict(test_feature)
    result = np.argmax(result)

    result_prob = clf.predict_proba(test_feature)[0]
    print('Recogntion: ', Config.CLASS_LABELS[int(result)])
    print('Probability: ', result_prob)
    Radar(result_prob)


def lstm_train(save_model_name: str, if_load: bool = True):
    """lstm训练模型
        Args:
            save_model_name: 保存模型的文件名
            if_load: 是否加载已有特征（True / False）
        Returns：
            model: 训练好的模型
    """
    # 提取特征
    if (if_load == True):
        x_train, x_test, y_train, y_test = of.load_feature(feature_path=Config.TRAIN_FEATURE_PATH_OPENSMILE,
                                                           train=True)
    else:
        x_train, x_test, y_train, y_test = of.get_data_o(Config.DATA_PATH, Config.TRAIN_FEATURE_PATH_OPENSMILE,
                                                         train=True)
    # 创建模型
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_test)

    model = LSTM_Model(input_shape=x_train.shape[1], num_classes=len(Config.CLASS_LABELS))

    # 二维数组转三维（samples, time_steps, input_dim）
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    # 训练模型
    print('>> ==================== Train Start =================== <<')
    model.train(x_train, y_train, x_test, y_val, n_epochs=Config.epochs)
    model.evaluate(x_test, y_test)
    model.save_model(save_model_name)
    print('>> ==================== Train Over =================== <<')

    return model


def lstm_predict(model, file_path: str):
    """lstm预测音频情感
        Args:
            model: 已加载或训练的模型
            file_path: 要预测的文件路径
        Returns：
            预测结果和置信概率
    """
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path
    playAudio(file_path)

    # 一个玄学 bug 的暂时性解决方案
    of.get_data_o(file_path, Config.PREDICT_FEATURE_PATH_OPENSMILE, train=False)
    test_feature = of.load_feature(Config.PREDICT_FEATURE_PATH_OPENSMILE, train=False)

    # 二维数组转三维（samples, time_steps, input_dim）
    test_feature = np.reshape(test_feature, (test_feature.shape[0], 1, test_feature.shape[1]))

    result = model.predict(test_feature)
    result = np.argmax(result)

    result_prob = model.predict_proba(test_feature)[0]
    print('Recogntion: ', Config.CLASS_LABELS[int(result)])
    print('Probability: ', result_prob)
    Radar(result_prob)
