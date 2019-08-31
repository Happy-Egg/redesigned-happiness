# coding=utf-8

import getopt
import sys

from CoreSource import speechRec
from CoreSource import wordSeg


def main(argv):

    file_path = 'AudioFiles/16k.wav'
    file_format = 'wav'
    word = '北京'

    try:
        options, args = getopt.getopt(argv, "-p:-f:-w:", ["path=", "format=", "word="])
    except getopt.GetoptError:
        sys.exit()

    for option, value in options:
        if option in ("-p", "--path"):
            file_path = value
        if option in ("-f", "--format"):
            file_format = value
        if option in ("-w", "--word"):
            word = value

    text_u8 = speechRec.do_tts(file_path, file_format)
    seg_list = wordSeg.do_cut(text_u8)
    wordSeg.do_search(seg_list, word)


if __name__ == '__main__':
    main(sys.argv[1:])
