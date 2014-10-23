## Cosmo Zhang @ Purdue 10/1
## cs578 final project on yelp
## Filename:combine.py
## -*- coding: utf-8 -*-

import cPickle as cpcl
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def uzbiznetwork(rvcndic, thre):
    #print rvcndic
    f = open('../data/yelp_academic_dataset_review.json', 'r')
    g = open('./user_bz_network_' + str(thre) +'.csv', 'w')
    data = f.readlines()
    lnklsdic={}
    for line in data:
    #print line
        linedata = json.loads(line)#.replace("\\n", ""))
        if (linedata["business_id"] in rvcndic) and (rvcndic[linedata["business_id"]] >= thre):
            if linedata["user_id"] not in lnklsdic:
                lnklsdic[linedata["user_id"]] = [linedata["business_id"]]
            else:
                lnklsdic[linedata["user_id"]].append(linedata["business_id"])
    for k in lnklsdic:
        g.write(k+ ": " +', '.join(lnklsdic[k])+ '\n')

    g.close()
    f.close()
    h = open('./user_bz_network_' + str(thre) +'.data', 'wb')
    cpcl.dump(lnklsdic, h)
    h.close()
    return lnklsdic

def uznetwork(adic, thre):
    f = open('../data/yelp_academic_dataset_user.json', 'r')
    g = open('./user_user_network_' + str(thre) +'.csv', 'w')
    data = f.readlines()

    lnklsdic={}

    for line in data:
    #print line
        linedata = json.loads(line)#.replace("\\n", ""))
        if linedata["user_id"] in adic:
            lnklsdic[linedata["user_id"]] = []
            for friend in linedata["friends"]:
                if friend in adic:
                    lnklsdic[linedata["user_id"]].append(friend)
    for k in lnklsdic:
        g.write(k+ ": " +', '.join(lnklsdic[k])+ '\n')

    f.close()
    g.close()
    h = open('./user_user_network_' + str(thre) +'.data', 'wb')
    cpcl.dump(lnklsdic, h)
    h.close()
    print '**********\nusernetwork constructed\n*********'

def main():

    if len(sys.argv) != 2:
        print 'Usage: %s, [reviewcounts threshold]' % sys.argv[0]
        sys.exit('Please provide enough parameters')

    thre = int(sys.argv[1]) #threshold as a parameter
    #f = open('./biz_review_counts_AZ.csv', 'r')
    #data = f.readlines()
    h = open('./biz_review_counts_AZ.data', 'rb')
    rvcntdic = cpcl.load(h)
    h.close()
    '''
    for line in data: #resume the dictionary of biz counts
        linedata = line.split(':')
        rvcntdic[linedata[0]] = int(linedata[1])
    '''
    uzbzn_dic = uzbiznetwork(rvcntdic, thre)
    uznetwork(uzbzn_dic, thre)

    
    
if __name__ == '__main__':
    main()
