## Cosmo Zhang @ Purdue 11/2014
## cs578 final project on yelp
## Filename:sdfpreprocess.py
## -*- coding: utf-8 -*-

import os
import codecs
import cPickle as cpcl
import sys
import numpy
reload(sys)
import nltk
from nltk.parse.stanford import StanfordParser
from pprint import pprint
sys.setdefaultencoding("utf-8")

def sdfprocess(rvdata):
    parser=StanfordParser(path_to_jar='/home/cosmo/Dropbox/Purdue/nlp/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar', path_to_models_jar='/home/cosmo/Dropbox/Purdue/nlp/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1-models.jar', model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz', java_options='-mx5000m')
    sdfdata=[]
    cnn = 1
    for eg in rvdata:
        if cnn%100 == 0: print "%f%% of documents finished" % (cnn*100*1.0/len(rvdata))
        cmt = eg[3].decode('utf-8') #3 is the idx of comment
        sentences = nltk.sent_tokenize(cmt)
        sdfparsed = parser.raw_parse_sents(sentences)
        sdfdata.append(eg[:3]+[sdfparsed])
        # print cnn
        # print sdfparsed
        # print sdfdata
        cnn += 1        
        # if cnn > 5: break
    return sdfdata


def main():
    if len(sys.argv) != 2:
        print 'Usage: %s [datafile path]' % sys.argv[0]
        sys.exit("please provide data file path")
    
    datafilename = sys.argv[1]
    # f = codecs.open(datafilename, 'rb', encoding="utf-8")
    f=open(datafilename, 'rb')
    revdata = cpcl.load(f)
    f.close()
    sdfdata = sdfprocess(revdata)
    # g = file('sdfdata.data', 'wb')
    g = file('../data/sdfdata.data', 'wb')
    cpcl.dump(sdfdata, g)
    g.close()
    print "Stanford Parser Process Done!"
    # print revdata[0] rev[0][3] is the review, and rev[0][2] is the polarity

if __name__ == "__main__":
    main()
