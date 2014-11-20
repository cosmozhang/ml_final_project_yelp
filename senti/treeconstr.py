## Cosmo Zhang @ Purdue 11/2014
## cs578 final project on yelp
## Filename:treeconstr.py
## -*- coding: utf-8 -*-

import os
import codecs
import cPickle as cpcl
import sys
import numpy
reload(sys)
import nltk
sys.setdefaultencoding("utf-8")

def traverse(t):
    try:
        t.node
    except AttributeError:
        print t
    
    else:
        print '(', t.node,
        for child in t:
            traverse(child)
        print ')',


def construct_tree(rvdata):
    # tparse = nltk.tree.Tree
    for eg in rvdata:
        print type(eg[3][0])
        eg[3][0].draw()
        # traverse(eg[3][0])
        
        # try:
        '''
        text = nltk.sent_tokenize(cmt)
        text = [nltk.word_tokenize(snt) for snt in text]
        text = [nltk.pos_tag(snt) for snt in text]
        '''






















def main():
    if len(sys.argv) != 2:
        print 'Usage: %s [datafile path]' % sys.argv[0]
        sys.exit("please provide data file path")
    
    datafilename = sys.argv[1]
    # f = codecs.open(datafilename, 'rb', encoding="utf-8")
    f=open(datafilename, 'rb')
    revdata = cpcl.load(f)
    f.close()
    treeddata = construct_tree(revdata)
    # print revdata[0] rev[0][3] is the review, and rev[0][2] is the polarity


if __name__ == "__main__":
    main()
