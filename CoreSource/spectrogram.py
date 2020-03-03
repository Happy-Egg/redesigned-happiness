# coding=<encoding name>
# 导入相应的包
import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def all_file_path(path='../../AudioFiles/CASIA_database/', file_type='wav'):
    """获取文件夹下所有音频文件的路径"""
    read_files = glob.glob(path + '*/*/*' + file_type)
    return read_files


if __name__ == '__main__':

    files = all_file_path()

    for file in files[:3]:
        f = wave.open(file, 'rb')  # 调用wave模块中的open函数，打开语音文件。
        params = f.getparams()  # 得到语音参数
        nchannels, sampwidth, framerate, nframes = params[:4]
        # nchannels:音频通道数，sampwidth:每个音频样本的字节数，framerate:采样率，nframes:音频采样点数
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        wavaData = np.fromstring(strData, dtype=np.int16)  # 得到的数据是字符串，将字符串转为int型
        wavaData = wavaData * 1.0 / max(abs(wavaData))  # wave幅值归一化
        wavaData = np.reshape(wavaData, [nframes, nchannels]).T  # .T 表示转置
        f.close()
        file_name = file.split('.')[-2].split('/')[-1]
        # 绘制语谱图
        plt.figure()
        plt.specgram(wavaData[0], Fs=framerate, scale_by_freq=True, sides='default')  # 绘制频谱
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.xlabel('Time(s)')
        # plt.ylabel('Frequency')
        # plt.title("Spectrogram_" + file_name)
        plt.savefig('../../ImageFiles/'+file_name + '.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.show()
