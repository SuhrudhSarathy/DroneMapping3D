from imutils import paths
import argparse
import requests
import cv2
import os
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
help="path to file containing image urls")
ap.add_argument("-o", "--output", required=True,
help="path to output dir for images")
args = vars(ap.parse_args())

#grab the list of urls and initialize total number of downloads
rows = open(args["urls"]).read().strip().split("\n")
total = 0

for url in rows:
    
    try:
        filename = '{}.jpg'.format(str(total))
        filepath = '{}{}'.format(args["output"], filename)
        urllib.request.urlretrieve(url, filepath)
        print('{} saved'.format(filename))
        total += 1
        
    except:
        print("skipping")

print("completed all urls in the file")