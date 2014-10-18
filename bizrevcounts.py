#2014 Yelp Data Challenge 
#Cosmo Zhang & Praveen @Purdue
#!/usr/bin/python3
# Filename:snetwork.py
# -*- coding: utf-8 -*-

import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

f = open('../data/yelp_academic_dataset_business.json', 'r')
g = open('./biz_review_counts.csv', 'w')
data = f.readlines()

lnklsdic={}

for line in data:
    #print line
    linedata = json.loads(line)#.replace("\\n", ""))
    if linedata["city"] == "Summerlin":
        lnklsdic[linedata["business_id"]]=linedata["review_count"]
    
for k in lnklsdic:
    #print k
    g.write(k+ ": " + str(lnklsdic[k]) + '\n')

g.close()
f.close()
