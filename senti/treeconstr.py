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


class node(object):
    def __init__(self, idx, word, parent = None, children = [], parentfactor = None, childfactor = None):
        self.children = children
        self.parent = parent
        self.idx = idx
        self.word = word
        self.parentfactor = parentfactor
        self.childfactor = childfactor

def reformat(cmt):
    wtp  = [tuple(x.rsplit('/')) for x in eg[0][0].split()] #word/tag
    # print wtp, '\n'
    wtp.insert(0, tuple([u'ROOT', u'S'])) #add the root 'word'
    # print wtp, '\n'

    #dependency relation reformat
    step1 = [x.strip(')').split('(',1) for x in eg[2]]
    step2 = [[x[0]] + x[1].split(', ') for x in step1]
    step3 = [[x[0]] + x[1].rsplit('-', 1) + x[2].rsplit('-', 1) for x in step2]
    depend = [[x[0]] + [tuple([x[1], int(x[2])])] + [tuple([x[3], int(x[4])])] for x in step3]

    return wtp, depend



def construct_tree(rvdata):
    
    wtp, depend = reformat(eg[3])
    
    # print eg[2], '\n'
    
    # treesent = nltk.Tree('S', rvdata[0][3][1])
    # print treesent.label()
    
    # traverse(treesent)
    # treesent.draw()























def main():
    if len(sys.argv) != 2:
        print 'Usage: %s [datafile path]' % sys.argv[0]
        sys.exit("please provide data file path")
    
    datafilename = sys.argv[1]
    # f = codecs.open(datafilename, 'rb', encoding="utf-8")
    f=open(datafilename, 'rb')
    parsedrev = cpcl.load(f)
    f.close()
    treeddata = construct_tree(parsedrev)
    # print revdata[0] rev[0][3] is the review, and rev[0][2] is the polarity


if __name__ == "__main__":
    main()
