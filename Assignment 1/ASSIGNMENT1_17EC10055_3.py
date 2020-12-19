import os, json
import collections
import pandas as pd
import numpy as np
import glob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

def file_number(string):
    return int(string.split('/')[2][:-4])

lemmat = WordNetLemmatizer()
stop = stopwords.words('english')
tokenizer = RegexpTokenizer(r"\w+")
txt_pattern = os.path.join('./ECTText/*.txt')
file_list = glob.glob(txt_pattern)
file_list.sort(key=file_number)
invert_id = dict()
for i in range(0,len(file_list)):
    with open(file_list[i]) as f:
        string = f.read().replace('\n',' ').lower().strip()
        new_words = tokenizer.tokenize(string)
        cnt=0
        for word in new_words:
            if word not in stop:
                word = lemmat.lemmatize(word)
                if word not in invert_id.keys():
                    invert_id[word] = [(int(i),cnt)]
                else:
                    invert_id[word].append((int(i),cnt))
                cnt = cnt+1
    print("Completed for file ",file_list[i])
invert_id = collections.OrderedDict(sorted(invert_id.items()))
print("No of keys in inverted index = ",len(invert_id))
print("Construction of inverted index complete.")
try:
    json.dump(invert_id,open("Inverted_Positional_Index.json", "w"))
except:
    print("failed")