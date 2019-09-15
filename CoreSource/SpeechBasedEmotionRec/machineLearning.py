from sklearn import svm
import glob
from random import shuffle
from CoreSource.SpeechBasedEmotionRec import speechFeature as sf
import numpy as np

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


def get_data(files, feature_model='all', split_ratio=0.9):
    """生成特征数据集并划分训练集
    Args:
        files: list,每个元素是一个文件路径
        feature_model: 特征提取的模式，'all'或'mfcc'
            all （默认）包含梅尔倒频谱系数、短时过零率和均方根能量
            mfcc 仅提取梅尔倒频谱系数
        split_ratio: 训练集占所有数据集文件的比例，默认0.9
    Returns:
        返回四个矩阵，按序分别是
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


def do_train(train_data, train_label, test_data, test_label):
    """SVM训练模型
        Args:
            train_data: 训练集，二维矩阵的每一行代表一条语音的特征向量
            train_label: 训练集标签，与训练集向量顺序同，用1~6代表6类情感
            test_data: 测试集，二维矩阵的每一行代表一条语音的特征向量
            test_label: 测试集标签，与训练集向量顺序同，用1~6代表6类情感
    """
    clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=19, gamma=0.0001)
    clf.fit(train_data, train_label)
    acc = clf.score(test_data, test_label)
    print('>>==================== Train Over ===================<<')
    print("测试集正确率为：" + "{:.2f}".format(acc * 100) + "%")
