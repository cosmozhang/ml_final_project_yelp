#2014 Yelp Data Challenge 
#Cosmo Zhang & Praveen @Purdue
# Filename:snetwork.py
# -*- coding: utf-8 -*-

import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

f = open('../data/yelp_academic_dataset_user.json', 'r')
g = open('./user_user_network.csv', 'w')
data = f.readlines()

lnklsdic={}

for line in data:
    #print line
    linedata = json.loads(line)#.replace("\\n", ""))
    lnklsdic[linedata["user_id"]]=linedata["friends"]
    
for k in lnklsdic:
    g.write(k+ ": " +', '.join(lnklsdic[k])+ '\n')

g.close()
f.close()
