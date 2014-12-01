## Cosmo Zhang @ Purdue 11/2014
## cs578 final project on yelp
## Filename:mltpreprocess.py
## -*- coding: utf-8 -*-

import gc
import os
import codecs
import cPickle as cpcl
import sys
import numpy
reload(sys)
import nltk
from nltk.parse.malt import MaltParser
from pprint import pprint
sys.setdefaultencoding("utf-8")

def sdfprocess(rvdata, partidx):
    os.environ["MALT_PARSER"] = "/home/cosmo/Dropbox/Purdue/nlp/maltparser-1.8"
    parser=MaltParser(mco='engmalt.poly-1.7', working_dir='/home/cosmo/Dropbox/Purdue/nlp/maltparser-1.8', additional_java_args=['-Xmx5000m'])
    sdfdata=[]
    cnn = 1
    # demo()
    print parser.raw_parse("I am a student.")
    for eg in rvdata:
        if cnn%100 == 0: print "%f%% of document %d finished" % (cnn*100*1.0/len(rvdata), partidx+1)
        cmt = eg[3].decode('utf-8') #3 is the idx of comment
        sentences = nltk.sent_tokenize(cmt)
        sdfparsed = [parser.raw_parse(sentence) for sentence in sentences]
        sdfdata.append(eg[:3]+[sdfparsed])
        # print cnn
        print sdfparsed
        # print sdfdata
        cnn += 1        
        if cnn > 5: break
        
    return sdfdata


def main():
    if len(sys.argv) != 3:
        print 'Usage: %s [datafile path] [part number]' % sys.argv[0]
        sys.exit("please provide data file path")
    
    datafilename = sys.argv[1]
    # f = codecs.open(datafilename, 'rb', encoding="utf-8")
    f=open(datafilename, 'rb')
    revdata = cpcl.load(f)
    f.close()
    partlen = (len(revdata)-len(revdata)%9)/9
    # for i in range(10):
    i = int(sys.argv[2])-1
    if len(revdata[i*partlen:]) >= partlen: partdata = revdata[i*partlen:(i+1)*partlen]
    elif len(revdata[i*partlen:]) < partlen: partdata = revdata[i*partlen:]
    # print i*partlen, (i+1)*partlen, partlen, len(revdata)
    sdfdata = sdfprocess(partdata, i)
    # sdfdata = []
    # g = file('sdfdata.data', 'wb')
    g = file('maltdata'+str(i+1)+'.data', 'wb')
    cpcl.dump(sdfdata, g)
    g.close()
    del sdfdata
    gc.collect()
    print "Stanford Parser Process Done on part %d!" % (i+1)
    # print revdata[0] rev[0][3] is the review, and rev[0][2] is the polarity

if __name__ == "__main__":
    main()
