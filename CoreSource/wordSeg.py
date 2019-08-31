# encoding=utf-8
import jieba


def do_cut(sentence):
    seg_list = jieba.cut_for_search(sentence)
    return seg_list


def do_search(seg_list, word):
    for w in seg_list:
        if w == word:
            print('\n'+'>> ===================== Search Result ===================== <<')
            print('Got a match in word \'' + word + '\'')
            return
        else:
            continue
    print('\n'+'>> ===================== Search Result ===================== <<')
    print('Do not have a match in word \'' + word + '\'')
