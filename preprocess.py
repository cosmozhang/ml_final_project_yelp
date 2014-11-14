## Cosmo Zhang @ Purdue 10/2014
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
    g = open('./data/user_bz_network_' + str(thre) +'.csv', 'w')
    data = f.readlines()
    lnklsdic={}
    h = open('./data/dataset_' + str(thre) +'.csv', 'w')
    dataset = []
    for line in data:
    #print line
        linedata = json.loads(line)#.replace("\\n", ""))
        if (linedata["business_id"] in rvcndic) and (rvcndic[linedata["business_id"]] >= thre) and (linedata["stars"] != 3):
            if linedata["user_id"] not in lnklsdic:
                lnklsdic[linedata["user_id"]] = [linedata["business_id"]]
            else:
                lnklsdic[linedata["user_id"]].append(linedata["business_id"])
            if linedata["stars"] > 3: pn = '+'
            elif linedata["stars"] < 3: pn = '-'
            eg = [str(linedata["user_id"]), str(linedata["business_id"]), pn, str(linedata["text"]).replace('\n', '').replace('\r', '').replace('\t', '')]
            dataset.append(eg)

    for k in lnklsdic:
        g.write(k+ ": " +', '.join(lnklsdic[k])+ '\n')

    for k in dataset:
        #print k
        h.write(k[0]+ ": " +', '.join(k[1:])+ '\n')


    f.close()
    g.close()
    h.close()

    g = open('./data/dataset_' + str(thre) +'.data', 'wb')
    cpcl.dump(dataset, g)
    g.close()

    h = open('./data/user_bz_network_' + str(thre) +'.data', 'wb')
    cpcl.dump(lnklsdic, h)
    h.close()
    print '**********\ndataset and user-biz network constructed\n*********'
    return lnklsdic

def uznetwork(adic, thre):
    f = open('../data/yelp_academic_dataset_user.json', 'r')
    g = open('./data/user_user_network_' + str(thre) +'.csv', 'w')
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
    h = open('./data/user_user_network_' + str(thre) +'.data', 'wb')
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
