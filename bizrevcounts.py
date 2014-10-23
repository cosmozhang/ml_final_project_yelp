#2014 Yelp Data Challenge 
#Cosmo Zhang & Praveen @Purdue
# Filename:snetwork.py
# -*- coding: utf-8 -*-

import cPickle as cpcl
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

f = open('../data/yelp_academic_dataset_business.json', 'r')
g = open('./biz_review_counts_AZ.csv', 'w')
data = f.readlines()

lnklsdic={}

for line in data:
    #print line
    linedata = json.loads(line)#.replace("\\n", ""))
    if linedata["state"] == "AZ":
        lnklsdic[linedata["business_id"]]=linedata["review_count"]
    
for k in lnklsdic:
    #print k
    g.write(k+ ": " + str(lnklsdic[k]) + '\n')

g.close()
f.close()

h = open('./biz_review_counts_AZ.data', 'wb')
cpcl.dump(lnklsdic, h)
h.close()
