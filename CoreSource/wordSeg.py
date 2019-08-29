# encoding=utf-8
import jieba


def do_cut(sentence):
    seg_list = jieba.cut(sentence)
    return seg_list


def do_search(seg_list, word):
    for w in seg_list:
        if w == word:
            print('Got a match in word ' + word)
            return
        else:
            continue
    print('Do not have a match in word ' + word)
