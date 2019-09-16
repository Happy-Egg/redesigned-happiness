# coding=<encoding name>
import librosa
import numpy as np
import CoreSource.SpeechBasedEmotionRec.spechFeatureNew as sfn


def get_mfcc(path):
    """获取梅尔倒频谱系数
        Args:
            path: 语音文件路径
        Returns:
            一维向量，是mfcc矩阵翻转后展开成一维的结果
    """
    win = 256
    inc = 80
    wavedata, nframes, framerate = sfn.read(path)
    FrameK = sfn.point_check(wavedata, win, inc)
    mel_length, S, mel_bank, P, logP, mfcc_feature = sfn.mfcc(FrameK, framerate, win)
    return mfcc_feature.T.flatten()


def get_zcr(path):
    """获取平均短时过零率
        Args:
            path: 语音文件路径
        Returns:
            数字，短时过零率
    """
    y, sr = librosa.load(path)
    zcr_feature = librosa.feature.zero_crossing_rate(y)
    zcr_feature = zcr_feature.flatten()
    zcr_feature = np.array([np.mean(zcr_feature)])
    return zcr_feature


def get_energy(path):
    """获取均方根能量
        Args:
            path: 语音文件路径
        Returns:
            数字，均方根能量
    """
    y, sr = librosa.load(path)
    energy_feature = librosa.feature.zero_crossing_rate(y)
    energy_feature = energy_feature.flatten()
    energy_feature = np.array([np.mean(energy_feature)])
    return energy_feature


def get_all_feature(path):
    """获取所有语音特征
        Args:
            path: 语音文件路径
        Returns:
            一维向量，mfcc矩阵翻转后展开成一维取前20个特征值
            在其后接上短时过零率和均方根能量
    """
    y, sr = librosa.load(path)
    mfcc_feature = librosa.feature.mfcc(y, sr, n_mfcc=16)
    mfcc_feature = mfcc_feature.T.flatten()[:20]
    zcr_feature = librosa.feature.zero_crossing_rate(y)
    zcr_feature = zcr_feature.flatten()
    zcr_feature = np.array([np.mean(zcr_feature)])
    energy_feature = librosa.feature.zero_crossing_rate(y)
    energy_feature = energy_feature.flatten()
    energy_feature = np.array([np.mean(energy_feature)])
    return np.concatenate((mfcc_feature, zcr_feature, energy_feature))
