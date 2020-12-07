import random
from urllib import request
from urllib.request import urlretrieve
import os
import sys
import threading

path = ".\originData"
if not os.path.exists(path):
    os.mkdir(path)
num = 10000
random.seed(114514)
with open("fall11_urls.txt", "rb") as f:
    urls = f.readlines()
random.shuffle(urls)

def get_image(url, index):
    url = url.decode("utf-8").split("\t")[1]
    if os.path.exists(path + '/' + '%s.jpg' % index):
        print("skip %s" % index)
        return
    try:
        request.urlretrieve(url, path + '/' + '%s.jpg' % index)
    except:
        print("error %s" % index)
    else:
        print("success %s" % index)


def main(index):
    print("thread %s" % index)
    start = index * 100000
    for i in range(start, start + 100000):
        get_image(urls[i], i)

t = []

for i in range(100):
    t.append(threading.Thread(target=main, args=(i,)))
    t[i].start()

for i in range(100):
    t[i].join()
