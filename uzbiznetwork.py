#2014 Yelp Data Challenge 
#Cosmo Zhang & Praveen @Purdue
#!/usr/bin/python3
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
    data = json.loads(line)#.replace("\\n", ""))
    if data["user_id"] not in lnklsdic:
        lnklsdic[data["user_id"]] = [data["business_id"]]
    lnklsdic[data["user_id"]].append(data["business_id"])
    
    
for k in lnklsdic:
    g.write(k+ ": " +', '.join(lnklsdic[k])+ '\n')

g.close()
f.close()
