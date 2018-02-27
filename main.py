import os
from stopwords import *
from tfidf import get_all_vector
from kmeans import *
import shutil

filepath = "F:\\PycharmProjects\\Crawl\\data.json"
savepath = "F:\\PycharmProjects\\Clustering\\data"
newspath = "F:\\PycharmProjects\\Clustering\\news"
historypath = "F:\\PycharmProjects\\Clustering\\history"

dividetotxt(filepath, savepath)
stop_words_set = stop_words("F:\\PycharmProjects\\Clustering\\stopwords.txt")
dataset = get_all_vector(savepath, historypath, stop_words_set)
result = kmeans(dataset[1], 8)
if os.path.exists(newspath):
    shutil.rmtree(newspath)
os.makedirs(newspath)
resultpaths = []
for i in range(result[1].shape[0]):
    temp = dataset[0][i].rfind("\\") + 1
    sort = int(result[1].tolist()[i][0])
    resultpath = newspath + "\\" + str(sort)
    if resultpath not in resultpaths:
        resultpaths.append(resultpath)
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    shutil.copyfile(dataset[0][i], resultpath + "\\" + dataset[0][i][temp:len(dataset[0][i])])

resultpaths.sort()
hisnames = os.listdir(historypath)
flag = 0
for path in resultpaths:
    for filename in os.listdir(path):
        for name in hisnames:
            if name in filename:
                os.rename(path, newspath + "\\favourite")
                flag = 1
                break
        if flag == 1:
            break
    if flag == 0:
        shutil.rmtree(path)
    flag = 0
