import requests
import time
from bs4 import BeautifulSoup
import re
import os
import json
import pandas as pd
import numpy as np
import glob

def file_number(string):
    return int(string.split('/')[2].split('-')[0])

html_pattern = os.path.join('./ECT/*.html')
file_list = glob.glob(html_pattern)
file_list.sort(key=file_number)
ECTNestedDict = {}
failed_new = []

if(os.path.isdir("./ECTNestedDict") == False):
    os.mkdir("./ECTNestedDict")
if(os.path.isdir("./ECTText") == False):
    os.mkdir("./ECTText")

regEx = r'(?:\d{1,2}[-/th|st|nd|rd\s.])?(?:(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|August|Sep|September|Oct|October|Nov|November|Dec|December)[\s,.]*)?(?:(?:\d{1,2})[-/th|st|nd|rd\s,.]*)?(?:\d{2,4})'

for i in range(0,len(file_list)):
    file = file_list[i]
    print(i,":",file)
    with open(file) as f:
        data = f.read()
        dict1 = {}
        soup = BeautifulSoup(data, 'html.parser')
        alldata = soup.find_all('p')
        result = re.findall(regEx, alldata[0].text)
        flg=0
        for r in result:
            if(r.find(',')!=-1):
                dict1['Date'] = r
                flg=1
        j = 1
        if flg==0:
            for j in range(1,len(alldata)):
                if alldata[j].text.lower().find('participants')!=-1 or alldata[j].text.lower().find('representatives')!=-1 or alldata[j].text.lower().find('executives')!=-1 or alldata[j].text.lower().find('analysts')!=-1:
                    break
            result = re.findall(regEx, alldata[j-1].text)
            for r in result:
                if(r.find(',')!=-1):
                    dict1['Date'] = r
                    flg=1
                elif r.find('65')!=-1:
                    dict1['Date'] = r
                    flg=1
            if flg==0:
                print(file)
        listf = []
        flg=0
        while(j<len(alldata)):
            if alldata[j].find('strong') is not None and len(alldata[j].text.strip())>0:
                if alldata[j].text.lower().find('participants')==-1 and alldata[j].text.lower().find('representatives')==-1 and alldata[j].text.lower().find('executives')==-1 and alldata[j].text.lower().find('analysts')==-1:
                    break
                elif alldata[j].text.lower().find('participants')!=-1 or alldata[j].text.lower().find('representatives')!=-1 or alldata[j].text.lower().find('executives')!=-1 or alldata[j].text.lower().find('analysts')!=-1:
                    r = j+1
                    while r<len(alldata):
                        if alldata[r].text.find('–')==-1 and alldata[r].text.find('-')==-1 and len(alldata[r].text.strip())>0:
                             break
                        if len(alldata[r].text.strip())>0:
                            listf.append(alldata[r].text)
                        r = r+1
                    j = r
            else:
                j = j+1
        dict1['Participants'] = listf
        if(len(listf)==0):
            print(i)
        dict4 = {}
        while(j<len(alldata)):
            if(alldata[j].text.strip().lower().startswith('question')):
                break
            else:
                if alldata[j].find('strong') is not None:
                    key = alldata[j].text.strip()
                    if key not in dict4:
                        dict4[key] = list()
                    r = j+1
                    if r<len(alldata):
                        while (alldata[r].find('strong') is None)==True:
                            if(alldata[r].text.strip().lower().startswith('question')):
                                break
                            dict4[key].append(alldata[r].text)
                            r = r+1
                            if r>=len(alldata):
                                break
                        j = r
                    else:
                        break
        dict1['Presentation'] = dict4
        j = j+1
        count = 1
        dict5 = {}
        while(j<len(alldata)):
            if alldata[j].find('strong') is not None:
                if(len(alldata[j].text.strip())>0):
                    dict6 = {}
                    dict6['Speaker'] = alldata[j].text.strip()
                    dict6['Remarks'] = list()
                    r = j+1
                    if(r<len(alldata)):
                        while alldata[r].find('strong') is None:
                            dict6['Remarks'].append(alldata[r].text)
                            r = r+1
                            if r>=len(alldata):
                                break
                        j = r
                        if (len(dict6['Remarks'])>0):
                            dict5[count] = dict6
                            count = count+1
                    else:
                        break
                else:
                    j = j+1
            else:
                j = j+1
        dict1['Questionnaire'] = dict5
        try:
            json.dump(dict1,open("./ECTNestedDict/"+str(i)+".json", "w"))
        except:
            failed_new.append(file)
        with open("./ECTText/"+str(i)+'.txt', 'w') as f:
            for key,value in dict1.items():
                f.write("%s" % key+" "+ "%s\n" % value)
    ECTNestedDict[file] = dict1