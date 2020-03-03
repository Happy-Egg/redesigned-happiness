# coding=<encoding name>
import getopt
import sys
from CoreSource.SpeechBasedEmotionRec import machineLearning as ml
from CoreSource.SpeechBasedEmotionRec import Utils as ul

def main():
    # file_path = 'AudioFiles/16k.wav'
    # file_format = 'wav'
    # word = '北京'
    #
    # try:
    #     options, args = getopt.getopt(argv, "-p:-f:-w:", ["path=", "format=", "word="])
    # except getopt.GetoptError:
    #     sys.exit()
    #
    # for option, value in options:
    #     if option in ("-p", "--path"):
    #         file_path = value
    #     if option in ("-f", "--format"):
    #         file_format = value
    #     if option in ("-w", "--word"):
    #         word = value

    files = ml.all_file_path('AudioFiles/CASIA_database/')
    train_data, train_label, test_data, test_label = ml.get_data_l(files)
    ml.svm_train(train_data, train_label, test_data, test_label)
    ml.svm_predict("svm_model.m", "AudioFiles/predictsets/record.wav")

    model = ul.load_model(load_model_name='LSTM_OPENSMILE', model_name='lstm')
    ml.lstm_predict(model, file_path="../../AudioFiles/predictsets/record.wav")


if __name__ == "__main__":
    main()
