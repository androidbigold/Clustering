import json
import codecs
import os
import re
import shutil
import jieba

rstr = r"[\/\\\:\*\?\"\<\>\|]"


def dividetotxt(filepath, savepath):
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath)
    data = codecs.open(filepath, "r", encoding="utf-8")
    load_dict = data.read().replace('\n', '')
    dict_list = load_dict.split('$')
    for i in range(len(dict_list)-1):
        new_dict = json.loads(dict_list[i])
        new_title = re.sub(rstr, "", new_dict['title'][0])
        news = open(savepath + "\\" + new_title + ".txt", "w", encoding="utf-8")
        line = json.dumps(new_dict, ensure_ascii=False, indent=0)
        news.write(line)


def read_from_file(file_name):
    with open(file_name, "r") as fp:
        words = fp.read()
    return words


def stop_words(stop_word_file):
    words = read_from_file(stop_word_file)
    result = jieba.cut(words)
    new_words = []
    for r in result:
        new_words.append(r)
    return set(new_words)


def del_stop_words(words, stop_words_set):
    #   words是已经切词但是没有去除停用词的文档。
    #   返回的会是去除停用词后的文档
    result = jieba.cut(words)
    new_words = []
    for r in result:
        if r not in stop_words_set:
            new_words.append(r)
    return new_words
