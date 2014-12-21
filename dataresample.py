## Cosmo Zhang @ Purdue 10/2014
## cs578 final project on yelp
## Filename:dataresample.py
## -*- coding: utf-8 -*-

import cPickle as cpcl
import json
import sys
from random import *
from math import *
reload(sys)
sys.setdefaultencoding('utf-8')

def resample(filename):
    f = open(filename, 'rb')
    data = cpcl.load(f)
    f.close()

    posdata = []
    negdata = []
    
    for eg in data:
        if eg[2] == '+':
            posdata.append(eg)
        elif eg[2] == '-':
            negdata.append(eg)
            
    posdataresam = sample(posdata, 30000)
    negdataresam = sample(negdata, 10000)
    
    datatosave = posdataresam + negdataresam

    shuffle(datatosave)
    
    newfilename = filename.replace('.data', '')
    g = open(newfilename + '_resample.data', 'wb')
    cpcl.dump(datatosave, g)
    g.close()

    print '**********\nresampled constructed\n*********'

def main():

    if len(sys.argv) != 2:
        print 'Usage: %s [file]' % sys.argv[0]
        sys.exit('Please provide enough parameters')
        
    filename = sys.argv[1]
    resample(filename)

    
if __name__ == '__main__':
    main()
