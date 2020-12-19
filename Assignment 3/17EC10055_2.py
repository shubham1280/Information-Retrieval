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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

def identity_tokenizer(text):
    return text

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

class Rochhio:
    def __init__(self,b):
        self.b=b
        self.class1_mean=list()
        self.class2_mean=list()
        self.feature_len=int(0)
    def fit(self,X_train,y_train):
        self.feature_len = X_train.shape[1]
        self.class1_mean=np.zeros(self.feature_len)
        self.class2_mean=np.zeros(self.feature_len)
        cnt1=int(0)
        cnt2=int(0)
        classes = ['class1','class2']
        to_add = int(1)
        for i in range(0,X_train.shape[0]):
            if y_train[i] == classes[0]:
                self.class1_mean+=X_train[i]
                cnt1 = cnt1 + to_add
            elif y_train[i] == classes[1]:
                self.class2_mean+=X_train[i]
                cnt2 = cnt2 + to_add
        self.class1_mean = np.divide(self.class1_mean,cnt1)
        self.class2_mean = np.divide(self.class2_mean,cnt2)

    def predict(self,X_test):
        y_pred=list()
        length = X_test.shape[0]
        classes = ['class1','class2']
        for i in range(0,length):
            t1 = np.linalg.norm(X_test[i]-self.class1_mean)
            t2 = np.linalg.norm(X_test[i]-self.class2_mean)
            if (t1+self.b)<t2:
                y_pred.append(classes[0])
            else:
                y_pred.append(classes[1])
        return y_pred

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
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english',lowercase=False)    
X_train_tfidf=tfidf.fit_transform(X_train_tfidf_tmp)
X_test_tfidf = tfidf.transform(X_test_tfidf_tmp)
print("Please wait for Rocchio Classifier to complete")
answer = open(output_file,'w',encoding ="utf-8")
answer.write("Value of b: ")
b = 0
answer.write(f" {b}")
rc = Rochhio(b)
rc.fit(X_train_tfidf,y_train)
y_pred_rochio=rc.predict(X_test_tfidf)
f1_val = f1_score(y_test,y_pred_rochio,average='macro')
answer.write("\n")
answer.write("Rocchio f1 score: ")
answer.write(f" {round(f1_val,5)}")
answer.close()
print("Completed")