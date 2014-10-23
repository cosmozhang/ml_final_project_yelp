#2014 Yelp Data Challenge 
#Cosmo Zhang & Praveen @Purdue
# Filename:snetwork.py
# -*- coding: utf-8 -*-

import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

f = open('../data/yelp_academic_dataset_review.json', 'r')
g = open('./user_bz_network.csv', 'w')
data = f.readlines()

lnklsdic={}

for line in data:
    #print line
    linedata = json.loads(line)#.replace("\\n", ""))
    if linedata["user_id"] not in lnklsdic:
        lnklsdic[linedata["user_id"]] = [linedata["business_id"]]
    lnklsdic[linedata["user_id"]].append(linedata["business_id"])
    
    
for k in lnklsdic:
    g.write(k+ ": " +', '.join(lnklsdic[k])+ '\n')

g.close()
f.close()
