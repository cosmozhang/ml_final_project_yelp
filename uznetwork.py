#2014 Yelp Data Challenge 
#Cosmo Zhang & Praveen @Purdue
#!/usr/bin/python3
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
    data = json.loads(line)#.replace("\\n", ""))
    lnklsdic[data["user_id"]]=data["friends"]
    
for k in lnklsdic:
    g.write(k+ ": " +', '.join(lnklsdic[k])+ '\n')

g.close()
f.close()
