import re
import os
import json
import pandas as pd
import numpy as np
import glob
import pickle
import sys
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import BernoulliNB,MultinomialNB

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

f1_MultNB=list()
f1_BernNB=list()
K = list()
y=1
for i in range(0,5):
    K.append(y)
    y = y*10

print("Please wait for Naive Bayes to complete......")

answer = open(output_file,'w',encoding ="utf-8")
answer.write("NumFeature")
for k in K:
    
    best_ftr=SelectKBest(mutual_info_classif, k = k)
    answer.write(f" {k}")
    X_train_trans= best_ftr.fit_transform(X_train, y_train)
    X_test_trans = best_ftr.transform(X_test)

    #FIT MULTINOMIAL NB
    mult_nb = MultinomialNB()
    mult_nb.fit(X_train_trans, y_train)
    y_pred_mnb = mult_nb.predict(X_test_trans)
    score1 = f1_score(y_test,y_pred_mnb,average='macro')
    #FIT_BERNAULLI
    bern_nb = BernoulliNB()
    bern_nb.fit(X_train_trans, y_train)
    y_pred_bern = bern_nb.predict(X_test_trans)
    score2 = f1_score(y_test,y_pred_bern,average='macro')
    f1_MultNB.append(score1)
    f1_BernNB.append(score2)
    print(f"At NB for k = {k}")

answer.write("\n")
answer.write("MultinomialNaiveBayes")
for i in range(0,len(f1_MultNB)):
    jk = f1_MultNB[i]
    answer.write(f" {round(jk,5)}")
answer.write("\n")
answer.write("BernoulliNaiveBayes")
for l in range(0,len(f1_BernNB)):
    jk = f1_BernNB[l]
    answer.write(f" {round(jk,5)}")
answer.close()
print("Complete")