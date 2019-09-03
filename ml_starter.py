from CoreSource.SpeechBasedEmotionRec import machineLearning as ml


def main():
    files = ml.all_file_path()
    train_data, train_label, test_data, test_label = ml.get_data(files)
    ml.do_train(train_data, train_label, test_data, test_label)


if __name__ == "__main__":
    main()
