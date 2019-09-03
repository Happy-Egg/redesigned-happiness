import librosa
import numpy as np


def get_mfcc(path):
    """获取梅尔倒频谱系数"""
    y, sr = librosa.load(path)
    mfcc_feature = librosa.feature.mfcc(y, sr, n_mfcc=16)
    mfcc_feature = mfcc_feature.T.flatten()
    return mfcc_feature


def get_zcr(path):
    """获取平均短时过零率"""
    y, sr = librosa.load(path)
    zcr_feature = librosa.feature.zero_crossing_rate(y)
    zcr_feature = zcr_feature.flatten()
    zcr_feature = np.array([np.mean(zcr_feature)])
    return zcr_feature


def get_energy(path):
    """获取均方根能量"""
    y, sr = librosa.load(path)
    energy_feature = librosa.feature.zero_crossing_rate(y)
    energy_feature = energy_feature.flatten()
    energy_feature = np.array([np.mean(energy_feature)])
    return energy_feature


def get_all_feature(path):
    """获取所有语音特征"""
    y, sr = librosa.load(path)
    mfcc_feature = librosa.feature.mfcc(y, sr, n_mfcc=16)
    mfcc_feature = mfcc_feature.T.flatten()
    zcr_feature = librosa.feature.zero_crossing_rate(y)
    zcr_feature = zcr_feature.flatten()
    zcr_feature = np.array([np.mean(zcr_feature)])
    energy_feature = librosa.feature.zero_crossing_rate(y)
    energy_feature = energy_feature.flatten()
    energy_feature = np.array([np.mean(energy_feature)])
    return np.concatenate((mfcc_feature, zcr_feature, energy_feature))

