import os
def read_stopword(fpath):
    # 读取中文停用词表
    with open(fpath, 'r', encoding='utf-8') as file:
        stopword = file.readlines()
    return [word.replace('\n', '') for word in stopword]

#加载多个停用词表
path = 'E:\WeChatPublicNumber\python\词云图\stopwords'
# 前两个停用词表是网上下载的，第三个是自己设置的
name_list = ['中文停用词.txt', '哈工大停用词.txt', 'stopword.txt']

stop_word = []
for fname in name_list:
    stop_word += read_stopword(os.path.join(path, fname))
stop_word = set(stop_word)