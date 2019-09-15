import os

m4a_path = "C://Users//JyunmauChan//Documents//GitHub//Speech-Emotion-Recognition//Datasets//predictsets//"

m4a_file = os.listdir(m4a_path)

for i, m4a in enumerate(m4a_file):
    print(m4a)
    print(i)
    os.system("ffmpeg -i " + m4a_path + m4a + " " + m4a_path + str(i) + ".wav")
