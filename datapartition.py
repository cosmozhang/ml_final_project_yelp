## Cosmo Zhang @ Purdue 10/2014
## cs578 final project on yelp
## Filename:datapartition.py
## -*- coding: utf-8 -*-

import cPickle as cpcl
import json
import sys
from random import *
from math import *
reload(sys)
sys.setdefaultencoding('utf-8')

def smellset(filename):
    f = open(filename, 'rb')
    data = cpcl.load(f)
    f.close()
    
    train_data = sample(data[0:int(ceil(len(data)*.6))], int(ceil(len(data)*.06)))
    validation_data = sample(data[int(ceil(len(data)*.6)):int(ceil(len(data)*.8))], int(ceil(len(data)*.02)))
    test_data = sample(data[int(ceil(len(data)*.8)):], int(ceil(len(data)*.02)))
    
    newfilename = filename.replace('.data', '')
    g = open(newfilename + '_train_small.data', 'wb')
    cpcl.dump(train_data, g)
    g.close()

    g = open(newfilename + '_val_small.data', 'wb')
    cpcl.dump(validation_data, g)
    g.close()

    g = open(newfilename + '_test_small.data', 'wb')
    cpcl.dump(test_data, g)
    g.close()

    
    print '**********\nsmallset constructed\n*********'

def normalset(filename):
    f = open(filename, 'rb')
    data = cpcl.load(f)
    f.close()
    
    train_data = data[0:int(ceil(len(data)*.6))]
    validation_data = data[int(ceil(len(data)*.6)):int(ceil(len(data)*.8))]
    test_data = data[int(ceil(len(data)*.8)):]
    
    newfilename = filename.replace('.data', '')
    g = open(newfilename + '_train.data', 'wb')
    cpcl.dump(train_data, g)
    g.close()

    g = open(newfilename + '_val.data', 'wb')
    cpcl.dump(validation_data, g)
    g.close()

    g = open(newfilename + '_test.data', 'wb')
    cpcl.dump(test_data, g)
    g.close()
    
    
    print '**********\nnormalset constructed\n*********'

def main():

    if len(sys.argv) != 2:
        print 'Usage: %s [file]' % sys.argv[0]
        sys.exit('Please provide enough parameters')
        
    filename = sys.argv[1]
    smellset(filename)
    normalset(filename)

    
if __name__ == '__main__':
    main()
