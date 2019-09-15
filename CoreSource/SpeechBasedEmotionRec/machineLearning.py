from sklearn import svm
import glob
from random import shuffle
from CoreSource.SpeechBasedEmotionRec import speechFeature as sf
import matplotlib.image as img
import numpy as np
import time

EMOTION_LABEL = {'angry': '1', 'fear': '2', 'happy': '3', 'neutral': '4', 'sad': '5', 'surprise': '6'}
EMOTION_LABEL1 = {'1': 'angry', '2': 'fear', '3': 'happy', '4': 'neutral', '5': 'sad', '6': 'surprise'}


def all_file_path(path='../../AudioFiles/CASIA_database/', file_type='wav'):
    """获取文件夹下所有音频文件的路径(乱序)"""
    read_files = glob.glob(path + '*/*/*' + file_type)
    shuffle(read_files)
    return read_files


def get_data(files, feature_model='all', split_ratio=0.9):
    """生成特征数据集"""
    data_feature = []
    data_labels = []
    mel_lens = []
    counter = 1
    for f in files:
        # # print('正在处理第 ' + "{:.0f}".format(counter) + ' 条...')
        # counter += 1
        if feature_model == 'all':
            data_feature.append(sf.get_all_feature(f))
        elif feature_model == 'mfcc':
            mel_length, mfcc_feature = sf.get_mfcc(f)
            # data_feature.append(mfcc_feature)
        else:
            print('feature_model error !')
            return
        data_labels.append(int(EMOTION_LABEL[f.split('\\')[-2]]))
        # if mel_length == 16:
        #     mel_lens.append(counter)
    # print(mel_lens)
    data_feature = np.array(data_feature)
    data_labels = np.array(data_labels)
    split_num = int(len(files) * split_ratio)
    train_data = data_feature[:split_num, :]
    train_label = data_labels[:split_num]
    test_data = data_feature[split_num:, :]
    test_label = data_labels[split_num:]
    return train_data, train_label, test_data, test_label


def get_predict_data(file):
    data_feature = []
    data_feature.append(sf.get_all_feature(file))
    return data_feature


def do_train(train_data, train_label, test_data, test_label):
    """SVM训练模型"""
    clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=19, gamma=0.0001)
    clf.fit(train_data, train_label)
    acc = clf.score(test_data, test_label)
    print('>>==================== Train Over ===================<<')
    print("测试集正确率为：" + "{:.2f}".format(acc * 100) + "%")
