#2014 Yelp Data Challenge 
#Cosmo Zhang & Praveen @Purdue
#!/usr/bin/python3
# Filename:avgcityrv.py
# -*- coding: utf-8 -*-

import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

f = open('../data/yelp_academic_dataset_business.json', 'r')
g = open('./avg_review_city.csv', 'w')
data = f.readlines()

citydic={}

for line in data:
    #print line
    linedata = json.loads(line)#.replace("\\n", ""))
    if linedata["city"] not in citydic:
        citydic[linedata["city"]] = {'bizn': 1, 'rvct': int(linedata["review_count"])}
    else:
        citydic[linedata["city"]]['bizn'] += 1
        citydic[linedata["city"]]['rvct'] += int(linedata["review_count"])
    
#sortdic = sorted(citydic, key = lambda x: float(x[1]['rvct']*1.0/x[1]['bizn']), reverse = True)

#print sortdic

maxv = float('-Inf')
for k in citydic:
    #print k
    g.write(k+ ", " + str(citydic[k]['rvct']) +', ' + str(citydic[k]['bizn']) + ', ' + str(citydic[k]['rvct']*1.0/citydic[k]['bizn']) + '\n')
    if citydic[k]['rvct']*1.0/citydic[k]['bizn'] >= maxv:
        maxv = citydic[k]['rvct']*1.0/citydic[k]['bizn']
        maxcity = k
print maxcity, maxv

g.close()
f.close()
