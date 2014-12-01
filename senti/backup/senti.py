## Cosmo Zhang @ Purdue 11/2014
## cs578 final project on yelp
## Filename:senti.py
## -*- coding: utf-8 -*-

import os
import redshift.parser
from redshift.sentence import Input
import cPickle as cpcl
import sys
import numpy
reload(sys)
from textblob import TextBlob
sys.setdefaultencoding('utf-8')

def postag(rvdata):
    piece = rvdata[0][3]
    test = TextBlob(piece)
    print test
    # print [str(x[0] + '/' + x[1]) for x in test.sentences[0].pos_tags]
    tagstc = ' '.join([str(x[0]) + '/' + str(x[1]) for x in test.sentences[-1].pos_tags])
    parser = redshift.parser.Parser('models')
    # print tagstc
    print test.sentences[0] + ''
    testsent = Input.from_pos(tagstc)#['A', 'list', 'of', 'tokens', 'is', 'required', '.'])
    # testsent = Input.from_untagged(test.sentences[0].words)
    parser.parse(testsent)
    print testsent.to_conll()#.split('\n')
    # sentence = Input.from_untagged(
    # parser.parse(test)
    '''
    parser = redshift.parser
    sentences = redshift.io_parse.read_pos(tagstc)
    parser.add_parses(sentences)
    sentences.write_parses(sys.stdout)
    '''
    






















def main():
    if len(sys.argv) != 2:
        print 'Usage: %s [datafile path]' % sys.argv[0]
        sys.exit("please provide data file path")
    
    datafilename = sys.argv[1]
    f=open(datafilename, 'rb')
    revdata = cpcl.load(f)
    f.close()
    tagged_data = postag(revdata)
    # print revdata[0] rev[0][3] is the review, and rev[0][2] is the polarity


if __name__ == "__main__":
    main()
