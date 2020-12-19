import requests
import time
from bs4 import BeautifulSoup
import re
import os

def grab_page(url,filename):
    print("attempting to grab page: " + url)
    page = requests.get(url,headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'})
    page_html = page.text
    soup = BeautifulSoup(page_html, 'html.parser')
    html_str = soup.find_all('div',class_="sa-art article-width")[0]
    if(os.path.isdir("./ECT") == False):
        os.mkdir("./ECT")
    html_file= open("./ECT/"+str(filename)+".html","w")
    html_file.write(str(html_str))
    html_file.close()

def process_list_page(i):
    origin_page = "https://seekingalpha.com/earnings/earnings-call-transcripts" + "/" + str(i)
    print("getting page " + origin_page)
    page = requests.get(origin_page,headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'})
    page_html = page.text
    soup = BeautifulSoup(page_html, 'html.parser')
    data = soup.find_all('a',class_ = 'dashboard-article-link')
    for i in range(0,len(data)):
        url_ending = data[i].get('href')
        url = "https://seekingalpha.com" + url_ending
        filename = str(i) + "-" + re.sub("/","-",url_ending[1:])
        grab_page(url,filename)

for i in range(1,400):
    process_list_page(i)