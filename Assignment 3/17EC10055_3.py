import re
import os
import json
import pandas as pd
import numpy as np
import glob
import pickle
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer as TfidfV
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics.pairwise import cosine_similarity

def identity_tokenizer(text):
    return text

def cosine(a, b):
    return 1-cosine_similarity(a,b)

def process_it(file_list,size,map1,q):
    lemmat = WordNetLemmatizer()
    stop = stopwords.words('english')
    tokenizer = RegexpTokenizer(r"\w+")
    X=list()
    y=list()
    for i in range(0,len(file_list['class1'])):
        DF = dict()
        tmp_ftr = np.zeros(size)
        lt = file_list['class1'][i]
        with open(lt,errors='ignore') as f:
            data = f.read()
            stp = (data).replace('\n',' ').lower().strip()
            new_words = tokenizer.tokenize(stp)
            for token in new_words:
                if token not in stop:
                    trim = lemmat.lemmatize(token)
                    if trim not in DF.keys():
                        DF[trim]=1
                    else:
                        DF[trim]+=1
        for key in DF.keys():
            if key in map1.keys():
                tmp_ftr[map1[key]]=DF[key]
        X.append(tmp_ftr)
        y.append('class1')
    for i in range(0,len(file_list['class2'])):
        DF = dict()
        tmp_ftr = np.zeros(size)
        with open(file_list['class2'][i],errors='ignore') as f:
            data = f.read()
            stp = (data).replace('\n',' ').lower().strip()
            new_words = tokenizer.tokenize(stp)
            for token in new_words:
                if token not in stop:
                    trim = lemmat.lemmatize(token)
                    if trim not in DF.keys():
                        DF[trim]=1
                    else:
                        DF[trim]+=1
        for key in DF.keys():
            if key in map1.keys():
                tmp_ftr[map1[key]]=DF[key]
        X.append(tmp_ftr)
        y.append('class2')
    return np.array(X),np.array(y)

def TFIDF_processor(H,map1):
    X_proc=list()
    for i in range(0,len(H)):
        temp=list()
        for j in np.nonzero(H[i])[0]:
            for cnt in range(0,H[i][j]):
                k = map1[j]
                temp.append(k)
        X_proc.append(temp)
    return np.array(X_proc)

lemmat = WordNetLemmatizer()
stop = stopwords.words('english')
output_file = sys.argv[2]
tokenizer = RegexpTokenizer(r"\w+")
path_data_directory = sys.argv[1]
if path_data_directory[-1]=="/":
	path_data_directory=path_data_directory[:-1]
train_dataset=dict()
vocabulary = set()
test_dataset=dict()
train_dataset['class1']=[path_data_directory+"/"+"class1"+"/train/"+f for f in os.listdir(path_data_directory+"/"+"class1"+"/train")]
train_dataset['class2']=[path_data_directory+"/"+"class2"+"/train/"+f for f in os.listdir(path_data_directory+"/"+"class2"+"/train")]
train_dataset['class1'].sort(key = lambda x:x.split("/")[-1])
train_dataset['class2'].sort(key = lambda x:x.split("/")[-1])
test_dataset['class1']=[path_data_directory+"/"+"class1"+"/test/"+f for f in os.listdir(path_data_directory+"/"+"class1"+"/test")]
test_dataset['class2']=[path_data_directory+"/"+"class2"+"/test/"+f for f in os.listdir(path_data_directory+"/"+"class2"+"/test")]
test_dataset['class1'].sort(key = lambda x:x.split("/")[-1])
test_dataset['class2'].sort(key = lambda x:x.split("/")[-1])

for i in range(0,len(train_dataset['class1'])):
    lt = train_dataset['class1'][i]
    with open(lt,errors='ignore') as f:
        data = f.read()
        stp = (data).replace('\n',' ').lower().strip()
        new_words = tokenizer.tokenize(stp)
        for token in new_words:
            if token not in stop:
                token = lemmat.lemmatize(token)
                vocabulary.add(token)
for i in range(0,len(train_dataset['class2'])):
    lt = train_dataset['class2'][i]
    with open(lt) as f:
        data = f.read()
        stp = (data).replace('\n',' ').lower().strip()
        new_words = tokenizer.tokenize(stp)
        for token in new_words:
            if token not in stop:
                token = lemmat.lemmatize(token)
                vocabulary.add(token)

map_2_idx=dict()
map_frm_idx=dict()
for idx,term in enumerate(vocabulary):
    map_frm_idx[idx]=term
    map_2_idx[term]=idx

X_train,y_train = process_it(train_dataset,len(vocabulary),map_2_idx,0)
X_test,y_test = process_it(test_dataset,len(vocabulary),map_2_idx,1)

X_train_tfidf_tmp = TFIDF_processor(X_train.astype(int),map_frm_idx)
X_test_tfidf_tmp = TFIDF_processor(X_test.astype(int),map_frm_idx)
print("TfidfVectorizer starts")
tf_idf = TfidfV(tokenizer=identity_tokenizer, stop_words='english',lowercase=False)    
X_train_tfidf=tf_idf.fit_transform(X_train_tfidf_tmp)
X_test_tfidf =tf_idf.transform(X_test_tfidf_tmp)

Knn_k=list()
print("Please wait for KNeighborsClassifier to complete")
for i in range(0,3):
    if i==0:
        Knn_k.append(1)
    if i==1:
        Knn_k.append(10)
    if i==2:
        Knn_k.append(50)    
answer = open(output_file,'w',encoding ="utf-8")
answer.write("NumFeature")
knn_results=list()
for i in range(0,3):
    print("KNC for k =",Knn_k[i])
    no = Knn_k[i]
    answer.write(f" {no}")
    clf=KNC(no,metric=cosine)
    clf.fit(X_train_tfidf,y_train)
    y_pred_knn=clf.predict(X_test_tfidf)
    h = f1_score(y_test,y_pred_knn,average='macro')
    knn_results.append(h)
answer.write("\n")
answer.write("KNeighborsClassifier")
for vals in knn_results:
    answer.write(f" {round(vals,6)}")
answer.close()