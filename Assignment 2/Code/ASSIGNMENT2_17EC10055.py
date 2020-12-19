from bs4 import BeautifulSoup
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

def file_number(string):
    return int(string.split('/')[3].split('.')[0])

pth = os.path.join('../Dataset/Dataset/*.html')
file_list = glob.glob(pth)
file_list.sort(key=file_number)
lemmat = WordNetLemmatizer()
stop = stopwords.words('english')
tokenizer = RegexpTokenizer(r"\w+")
size = len(file_list)

def cosinesim(a,b):
    dot = np.dot(a, b)
    if(dot==0):
        return 0
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos

def vectorizequery(K,query,position_terms,Inverted_Positional_Dictionary):
    query = query.lower().strip()
    q_words = tokenizer.tokenize(query)
    y = np.zeros(K)
    for q in q_words:
        if q not in stop:
            q = lemmat.lemmatize(q)
            if q in Inverted_Positional_Dictionary.keys():
                y[position_terms[q]] = np.log10(size/len(Inverted_Positional_Dictionary[q]))
    return y

def get_champion_local_docs(K,query,position_terms,ChampionListLocal):
    query = query.lower().strip()
    q_words = tokenizer.tokenize(query)
    y = list()
    for q in q_words:
        if q not in stop:
            q = lemmat.lemmatize(q)
            if q in ChampionListLocal.keys():
                y.extend(ChampionListLocal[q])
    return y

def get_champion_global_docs(K,query,position_terms,ChampionListGlobal):
    query = query.lower().strip()
    q_words = tokenizer.tokenize(query)
    y = list()
    for q in q_words:
        if q not in stop:
            q = lemmat.lemmatize(q)
            if q in ChampionListGlobal.keys():
                y.extend(ChampionListGlobal[q])
    return y

def vectorizedoc(K,doc,position_terms,Inverted_Positional_Dictionary,Document_term_dictionary):
    dit = Document_term_dictionary[doc]
    y = np.zeros(K)
    for key,value in dit.items():
        y[position_terms[key]] = value*(np.log10(size/len(Inverted_Positional_Dictionary[key])))
    return y

def assign_followers(p,L):
    Leader_followed = dict()
    Leader_foll_dict = dict()
    for i in p.keys():
        maxsim = 0
        for j in L:
            if j in p.keys():
                if(cosinesim(p[i],p[j])>maxsim):
                    maxsim = cosinesim(p[i],p[j])
                    Leader_followed[i] = int(j)
    for i in L:
        if i in p.keys():
            Leader_foll_dict[i] = list()
    for i in p.keys():
        Leader_foll_dict[Leader_followed[i]].append(i)
    return Leader_foll_dict

############## Building InvertedPositionalIndex ##################
#print("Building InvertedPositionalIndex")
InvertedPositionalIndex = dict()
Inverted_Positional_Dictionary = dict()
Document_term_dictionary = dict()
for i in range(0,len(file_list)):
    lt = file_list[i]
    with open(lt) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'html.parser')
        stp = (soup.text).replace('\n',' ').lower().strip()
        new_words = tokenizer.tokenize(stp)
        DF = {}
        for token in new_words:
            if token in DF.keys():
                DF[token] += 1
            else:
                DF[token] = 1
    for key,value in DF.items():
        if key not in Inverted_Positional_Dictionary.keys():
            Inverted_Positional_Dictionary[key] = list()
        Inverted_Positional_Dictionary[key].append((file_number(lt),np.log10(1+value)))
    for key,value in DF.items():
        DF[key] = np.log10(1+value)
    Document_term_dictionary[file_number(lt)] = DF
for key,value in Inverted_Positional_Dictionary.items():
    InvertedPositionalIndex[(key,np.log10(size/len(value)))] = value

############## Building ChampionListLocal ##################
#print("Building ChampionListLocal")
ChampionListLocal = dict()
for key,value in Inverted_Positional_Dictionary.items():
    value = sorted(value, key = lambda x: x[1],reverse=True)
    Inverted_Positional_Dictionary[key] = value
for key,value in Inverted_Positional_Dictionary.items():
    ChampionListLocal[key] = list()
    for i in range(0,50):
        if(i<len(value)):
            ChampionListLocal[key].append(value[i][0])

############## Building ChampionListGlobal ##################
#print("Building ChampionListGlobal")
ChampionListGlobal = dict()
with open('../Dataset/StaticQualityScore.pkl','rb') as f:
    Stat_scores = pickle.load(f)
for key,value in Inverted_Positional_Dictionary.items():
    value = sorted(value, key = lambda x: (x[1]+Stat_scores[x[0]]),reverse=True)
    Inverted_Positional_Dictionary[key] = value
for key,value in Inverted_Positional_Dictionary.items():
    ChampionListGlobal[key] = list()
    for i in range(0,50):
        if(i<len(value)):
            ChampionListGlobal[key].append(value[i][0])

############## Answering Free Text Queries ######################
position_terms = dict()
terms = list(Inverted_Positional_Dictionary.keys())
K = len(Inverted_Positional_Dictionary)
for i in range(0,len(terms)):
    position_terms[terms[i]] = i
p = dict() ###### Creating Document Vectors ######
for i in range(0,size):
    p[file_number(file_list[i])] = list()
for i in range(0,size):
    p[file_number(file_list[i])] = vectorizedoc(K,file_number(file_list[i]),position_terms,Inverted_Positional_Dictionary,Document_term_dictionary)

with open('../Dataset/Leaders.pkl','rb') as f:  ##### Assign Leader #######
    Leader_list = pickle.load(f)
Leader_follow_dict = assign_followers(p,Leader_list)

#print("Reading Queries to get answers")
with open(sys.argv[1]) as f:
    contents = f.read()
contents = contents.split("\n")
with open("RESULTS2_17EC10055"+'.txt', 'w') as f:
    for content in contents:
        #print(content)
        f.write(content)
        f.write("\n")
        relevance1 = dict()
        q = vectorizequery(K,content,position_terms,Inverted_Positional_Dictionary)
        for i in p.keys():
            relevance1[i] = cosinesim(p[i],q)
        res1 = sorted(relevance1.items(), key=lambda item: item[1],reverse=True)
        res1 = res1[:10]
        for i in range(0,len(res1)):
            if(i+1<len(res1)):
                #print("<"+str(res1[i][0])+","+str(res1[i][1])+">",end=",")
                f.write("<"+str(res1[i][0])+","+str(res1[i][1])+">,")
            else:
                #print("<"+str(res1[i][0])+","+str(res1[i][1])+">")
                f.write("<"+str(res1[i][0])+","+str(res1[i][1])+">")
        f.write("\n")

        relevance2 = dict()
        list_of_local_docs = get_champion_local_docs(K,content,position_terms,ChampionListLocal)
        if(len(list_of_local_docs)==0):
            #print("No relevant documents found for the given query")
            f.write("No relevant documents found for the given query")
            f.write("\n")
        else:
            for i in list_of_local_docs:
                relevance2[i] = cosinesim(p[i],q)
            res2 = sorted(relevance2.items(), key=lambda item: item[1],reverse=True)
            if(len(res2)>10):
                res2 = res2[:10]
            for i in range(0,len(res2)):
                if(i+1<len(res2)):
                    #print("<"+str(res2[i][0])+","+str(res2[i][1])+">",end=",")
                    f.write("<"+str(res2[i][0])+","+str(res2[i][1])+">,")
                else:
                    #print("<"+str(res2[i][0])+","+str(res2[i][1])+">")
                    f.write("<"+str(res2[i][0])+","+str(res2[i][1])+">")
            f.write("\n")

        relevance3 = dict()
        list_of_global_docs = get_champion_global_docs(K,content,position_terms,ChampionListGlobal)
        if(len(list_of_global_docs)==0):
            #print("No relevant documents found for the given query")
            f.write("No relevant documents found for the given query")
            f.write("\n")
        else:
            for i in list_of_global_docs:
                relevance3[i] = cosinesim(p[i],q)
            res3 = sorted(relevance3.items(), key=lambda item: item[1],reverse=True)
            if(len(res3)>10):
                res3 = res3[:10]
            for i in range(0,len(res3)):
                if(i+1<len(res3)):
                    #print("<"+str(res3[i][0])+","+str(res3[i][1])+">",end=",")
                    f.write("<"+str(res3[i][0])+","+str(res3[i][1])+">,")
                else:
                    #print("<"+str(res3[i][0])+","+str(res3[i][1])+">")
                    f.write("<"+str(res3[i][0])+","+str(res3[i][1])+">")
            f.write("\n")
        
        relevance4 = dict()
        best_leader = Leader_list[0]
        maximum_sim = 0
        for j in Leader_list:
            if j in p.keys():
                if(cosinesim(p[j],q)>maximum_sim):
                    maximum_sim = cosinesim(p[j],q)
                    best_leader = j
        for i in Leader_follow_dict[best_leader]:
            relevance4[i] = cosinesim(p[i],q)
        res4 = sorted(relevance4.items(), key=lambda item: item[1],reverse=True)
        if(len(res4)>10):
            res4 = res4[:10]
        for i in range(0,len(res4)):
            if(i+1<len(res4)):
                #print("<"+str(res4[i][0])+","+str(res4[i][1])+">",end=",")
                f.write("<"+str(res4[i][0])+","+str(res4[i][1])+">,")
            else:
                #print("<"+str(res4[i][0])+","+str(res4[i][1])+">")
                f.write("<"+str(res4[i][0])+","+str(res4[i][1])+">")
        f.write("\n")
        f.write("\n")