## Cosmo Zhang @ Purdue 11/2014
## cs578 final project on yelp
## Filename:sdfpreprocess.py
## -*- coding: utf-8 -*-

import gc
import os
import codecs
import cPickle as cpcl
import sys
import numpy
import time
reload(sys)
import nltk
from nltk.parse.stanford import StanfordParser
from pprint import pprint
from progressbar import *
sys.setdefaultencoding("utf-8")

def sdfprocess(rvdata):
    parser=StanfordParser(path_to_jar='/home/cosmo/Dropbox/Purdue/nlp/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar', path_to_models_jar='/home/cosmo/Dropbox/Purdue/nlp/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1-models.jar', model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz', java_options='-mx15000m')
    sdfdata=[]
    cnn = 0
    widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=len(rvdata)).start()
    for eg in rvdata:
        # if cnn%100 == 0: print "%f%% of document %d finished" % (cnn*100*1.0/len(rvdata), partidx+1)
        cmt = eg[3].decode('utf-8') #3 is the idx of comment
        sentences = nltk.sent_tokenize(cmt)
        parsedls = []
        for snt in sentences:
            sntparsed = parser.raw_parse(snt)
            parsedls.append(sntparsed)
        sdfdata.append(eg[:3]+[parsedls])
        # print cnn
        # print sdfparsed
        # print sdfdata
        # if cnn > 5: break
        pbar.update(cnn + 1)
        cnn += 1        
    pbar.finish()
    return sdfdata


def main():
    if len(sys.argv) != 3:
        print 'Usage: %s [datafile path] [output path]' % sys.argv[0]
        sys.exit("please provide data file path")
    
    datafilename = sys.argv[1]
    output = sys.argv[2]
    # f = codecs.open(datafilename, 'rb', encoding="utf-8")
    f = open(datafilename, 'rb')
    revdata = cpcl.load(f)
    f.close()
    # for i in range(10):
    # i = int(sys.argv[2])-1
    # if len(revdata[i*partlen:]) >= partlen: partdata = revdata[i*partlen:(i+1)*partlen]
    # elif len(revdata[i*partlen:]) < partlen: partdata = revdata[i*partlen:]
    # print i*partlen, (i+1)*partlen, partlen, len(revdata)
    sdfdata = sdfprocess(revdata)
    # sdfdata = []
    # g = file('sdfdata.data', 'wb')
    g = file(output, 'wb')
    cpcl.dump(sdfdata, g)
    g.close()
    del sdfdata
    gc.collect()
    print "Stanford Parser Process Done!"
    # revdata[0] rev[0][3] is the review, and rev[0][2] is the polarity

if __name__ == "__main__":
    main()
