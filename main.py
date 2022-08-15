import pandas as pd
import numpy as np
#将path1设置为test.csv path2 设置为train.csv path3 设只为stopwords.txt
#将64行地址分别设置为 neg.txt 和pos.txt
#首次运行时将 63 到68 行解注释 运行时间较长越四分钟
#后续运行时可以将其加上注释 ，提升速度

# 处理数据集
import sklearn.linear_model
path2 = r"test.csv"
df2 = pd.read_csv(path2).astype(str)
path1 = r"train.csv"
df1 = pd.read_csv(path1) .astype(str) # 读取数据// 防止结巴报错 float' object has no attribute 'decode'
df1.drop(['News Url', 'Image Url'], axis='columns', inplace=True)  # 删除不要的特征
df2.drop(['News Url', 'Image Url'], axis='columns', inplace=True)  # 删除不要的特征
path3=r"stopword.txt"#停助词表
def stopwordslist():#去停助词的函数
      stopwords = [line.strip() for line in open(path3,encoding='UTF-8').readlines()]
      return stopwords
stopwords=stopwordslist()
#特征化
report_content=pd.concat([df1['Report Content'],df2['Report Content']],axis=0)#只对评论
title_set=pd.concat([df1['Title'], df2['Title']], axis=0)#只对title数据
source_set=pd.concat([df1["Ofiicial Account Name"],df2["Ofiicial Account Name"]],axis=0)#只对来源

#分词
import jieba
train_data=list()#将title 分词
for each in title_set:
    train_data.append(' '.join(jieba.cut(each)))
train_data2=source_set
train_data2=train_data2.fillna('查无')
print("来源空的",train_data2.isnull().sum())#将来源分词
train_data2=list()
for each in source_set:
    train_data2.append(' '.join(jieba.cut(each)))
print(len(train_data))
train_data3=list()
for each in report_content:#将评论分词
    train_data3.append(' '.join(jieba.cut(each)))
#去停助词
all_data=list()
for i in range(len(train_data)):
    l1=train_data[i].split()
    l2=train_data2[i].split()
    l3=train_data3[i].split()
    l1.extend(l2)
    l1.extend(l3)
    l4=list()
    for j in l1:
        if j not in stopwords:
            l4.append(j)
    all_data.append(' '.join(l4))
#输出一个结果
print(all_data[0])

#进行情感分析
from snownlp import SnowNLP
from snownlp import sentiment
 #数据预处理
#主要程序太慢了存到一个文件里
l1=np.array(report_content)
l2=np.array([])
sentiment.train(r"neg.txt", r"pos.txt")
for i in range(l1.size):
    l1[i]=SnowNLP(l1[i]).sentiments
np.save('datafeeling1',l1)
m1=np.load("datafeeling1.npy",allow_pickle=True)
m2=m1
for i in range(m1.size):
    if m1[i]>=0.5:
        m2[i]='1'
    else:
        m2[i]='0'
y_predict2=np.array(m2[10587:])


#   tfidf
from sklearn.feature_extraction.text import TfidfVectorizer#l
vectorizer = TfidfVectorizer(max_features=30000)
features=vectorizer.fit_transform(all_data)  ##得到tf—idf权重矩阵\
from sklearn.linear_model import RidgeClassifier
RidgeClassifier(alpha=1.0,class_weight=None,copy_X=True,fit_intercept=True,max_iter=None,normalize=False,random_state= None ,solver="auto",tol=0.0001)
clf=RidgeClassifier()
clf.fit(features[:df1.shape[0]],df1['label'])    #稀疏矩阵，目标值
y_predict=clf.predict(features[10587:])


#结果评价
target_names = ['class 0', 'class 1']
from sklearn.metrics import classification_report
print("基于tfidf")
print(classification_report(df2['label'],y_predict,target_names=target_names))#计算描述指标
from sklearn.metrics import roc_auc_score
print("基于情感分析")
print(classification_report(df2['label'],y_predict2))
print("roc_auc_score")
print(roc_auc_score(df2['label'],y_predict))
print(roc_auc_score(df2['label'],m2[10587:]))




