#2014 Yelp Data Challenge 
#Cosmo Zhang & Praveen @Purdue
# Filename:snetwork.py
# -*- coding: utf-8 -*-

import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

f = open('../data/yelp_academic_dataset_user.json', 'r')
g = open('./user_review_counts.csv', 'w')
data = f.readlines()

lnklsdic={}

for line in data:
    #print line
    linedata = json.loads(line)#.replace("\\n", ""))
    lnklsdic[linedata["user_id"]]=linedata["review_count"]
    
for k in lnklsdic:
    #print k
    g.write(k+ ": " + str(lnklsdic[k]) + '\n')

g.close()
f.close()
