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
from random import uniform
from poldicload import loadbinpoldic #my own module
from txttoformat import reformat
from math import exp
from nltk.stem import SnowballStemmer
sys.setdefaultencoding("utf-8")


#global


class NodeFactor(object):
    def __init__(self, parent, word_q, word_r, word_baseform, postag):
        self.parent = parent
        self.word_q = word_q
        self.word_r = word_r
        self.word_baseform = word_baseform
        self.postag = postag
        self.posv = None
        self.negv = None
        self.downvec0 = [None, None]
        self.downvec1 = [None, None]
        
    def __str__(self):
        return self.word_baseform

    def calnfv(self, paradicN):
        #initialization of parameters if it is not in the parameter dictionary
        #parameters for featureset a
        if '+' not in paradicN[0]:
            paradicN[0]['+'] = uniform(-1.0, 1.0)
        if '-' not in paradicN[0]:
            paradicN[0]['-'] = uniform(-1.0, 1.0)
        #parameters for featureset b
        if '+' + self.word_q not in paradicN[1]:
            paradicN[1]['+' + self.word_q] = uniform(-1.0, 1.0)
        if '-' + self.word_q not in paradicN[1]:
            paradicN[1]['-' + self.word_q] = uniform(-1.0, 1.0)
        #parameters for featureset tag
        if '+' + self.postag not in paradicN[2]:
            paradicN[2]['+' + self.postag] = uniform(-1.0, 1.0)
        if '-' + self.postag not in paradicN[2]:
            paradicN[2]['-' + self.postag] = uniform(-1.0, 1.0)
        #parameters for featureset word
        if '+' + self.word_baseform not in paradicN[3]:
            paradicN[3]['+' + self.word_baseform] = uniform(-1.0, 1.0)
        if '-' + self.word_baseform not in paradicN[3]:
            paradicN[3]['-' + self.word_baseform] = uniform(-1.0, 1.0)

        #calculate probabilities of both posi and neg
        self.posv = exp(paradicN[0]['+'] + paradicN[1]['+' + self.word_q] + paradicN[2]['+' + self.postag] + paradicN[3]['+' + self.word_baseform])
        self.negv = exp(paradicN[0]['-'] + paradicN[1]['-' + self.word_q] + paradicN[2]['-' + self.postag] + paradicN[3]['-' + self.word_baseform])
        # self.pos_p = sum_pos/(sum_pos + sum_neg)
        # self.neg_p = sum_neg/(sum_pos + sum_neg)
        return (self.posv, self.negv)

class EdgeFactor(object):
    def __init__(self, parent, child, parent_word_q, parent_word_r, child_word_baseform, parent_word_baseform):
        self.parent = parent
        self.child = child
        self.parent_word_q = parent_word_q
        self.child_word_baseform = child_word_baseform
        self.parent_word_baseform = parent_word_baseform
        self.parent_word_r = parent_word_r
        self.pos_posv = None
        self.pos_negv = None
        self.neg_negv = None
        self.neg_posv = None
        self.upvec = [None, None]
        self.downvec0 = [None, None]
        self.downvec1 = [None, None]

    def __str__(self):
        return self.parent_word_baseform + '~' + self.child_word_baseform

    def calefv(self, paradicE):
        #initialization of parameters if it is not in the parameter dictionary
        #parameters for featureset A
        if '++' not in paradicE[0]:
            paradicE[0]['++'] = uniform(0.9, 1.1)
        if '--' not in paradicE[0]:
            paradicE[0]['--'] = uniform(0.9, 1.1)
        if '+-' not in paradicE[0]:
            paradicE[0]['+-'] = uniform(-1.0, 1.0)
        if '-+' not in paradicE[0]:
            paradicE[0]['-+'] = uniform(-1.0, 1.0)    
        '''
        #parameters for featureset B
        if '+' + word_q not in paradicE[1]:
            paradicE[1]['+' + self.word_q] = uniform(-1.0, 1.0)
        if '-' + word_q not in paradicE[1]:
            paradicE[1]['-' + self.word_q] = uniform(-1.0, 1.0)
        #parameters for featureset tag
        if '+' + self.postag not in paradicE[2]:
            paradicE[2]['+' + self.postag] = uniform(-1.0, 1.0)
        if '-' + self.postag not in paradicE[2]:
            paradicE[2]['-' + self.postag] = uniform(-1.0, 1.0)
        '''
        #parameters for parent_featureset word
        if '++' + self.parent_word_baseform not in paradicE[3]:
            paradicE[3]['++' + self.parent_word_baseform] = uniform(-1.0, 1.0)
        if '--' + self.parent_word_baseform not in paradicE[3]:
            paradicE[3]['--' + self.parent_word_baseform] = uniform(-1.0, 1.0)
        if '+-' + self.parent_word_baseform not in paradicE[3]:
            paradicE[3]['+-' + self.parent_word_baseform] = uniform(-1.0, 1.0)
        if '-+' + self.parent_word_baseform not in paradicE[3]:
            paradicE[3]['-+' + self.parent_word_baseform] = uniform(-1.0, 1.0)

        #parameters for child_featureset word
        if '++' + self.child_word_baseform not in paradicE[4]:
            paradicE[4]['++' + self.child_word_baseform] = uniform(-1.0, 1.0)
        if '--' + self.child_word_baseform not in paradicE[4]:
            paradicE[4]['--' + self.child_word_baseform] = uniform(-1.0, 1.0)
        if '+-' + self.child_word_baseform not in paradicE[4]:
            paradicE[4]['+-' + self.child_word_baseform] = uniform(-1.0, 1.0)
        if '-+' + self.child_word_baseform not in paradicE[4]:
            paradicE[4]['-+' + self.child_word_baseform] = uniform(-1.0, 1.0)

        #calculate probabilities of both posi and neg
        self.pos_posv = exp(paradicE[0]['++'] + paradicE[3]['++' + self.parent_word_baseform] + paradicE[4]['++' + self.child_word_baseform])
        self.neg_negv = exp(paradicE[0]['--'] + paradicE[3]['--' + self.parent_word_baseform] + paradicE[4]['--' + self.child_word_baseform])
        self.pos_negv = exp(paradicE[0]['+-'] + paradicE[3]['+-' + self.parent_word_baseform] + paradicE[4]['+-' + self.child_word_baseform])
        self.neg_posv = exp(paradicE[0]['-+'] + paradicE[3]['-+' + self.parent_word_baseform] + paradicE[4]['-+' + self.child_word_baseform])
        # self.pos_p = sum_pos/(sum_pos + sum_neg)
        # self.neg_p = sum_neg/(sum_pos + sum_neg)
        return (self.pos_posv, self.pos_negv, self.neg_posv, self.neg_negv)

class Node(object):
    def __init__(self, word, tag):
        self.children = []
        self.parents = []
        # self.toparantrelation = None
        # self.idx = idx
        self.pol = None
        self.word = word
        self.parents_edgefactor = []
        self.children_edgefactor = []
        self.nodefactor = None
        self.tag = tag 
        self.toplgod = float("-inf")
        self.toparentrelation = {}

    def __str__(self):
        return self.word
    
    def addparent(self, obj, relation):
        self.parents.append(obj)
        self.toparentrelation[obj] = relation
        # self.toparantrelation = relation

    def addchild(self, obj):
        self.children.append(obj)
    
    def addchildfactor(self, obj):
        self.children_edgefactor.append(obj)

    def addparentfactor(self, obj):
        self.parents_edgefactor.append(obj)

    def addnodefactor(self, obj):
        self.nodefactor = obj

def simplebfs(nd):
    if nd.children != []:
        depthls = []
        for child in nd.children:
            if child.toplgod < nd.toplgod + 1:
                child.toplgod = nd.toplgod +1
            depthls.append(simplebfs(child))
        return max(depthls)
    else:
        return nd.toplgod
        
def word_q_check(wd, pold):
    if wd in pold[0]:
        return 'q+'
    elif wd in pold[1]:
        return 'q-'
    else:
        return 'q0'

def construct_tree(eg, pold):
    snowball_stemmer = SnowballStemmer("english")
    wtp, depend = reformat(eg[3])
    # print wtp
    print depend
    nodels = []
    for item in wtp:
        if item[0] != 'ROOT':
            newnode = Node(item[0], item[1])
        else:
            newnode = Node('rootnode', item[1])
        word_q = word_q_check(item[0], pold)
        word_r = 'r' #will change to reverse later after got the dictionary
        word_baseform = snowball_stemmer.stem(item[0])
        if item[0] != 'ROOT': newnode.nodefactor = NodeFactor(newnode, word_q, word_r, word_baseform, item[1]) #nodefactor initialize and attach them to the hidden varibles
        nodels.append(newnode) #nodels[0] is the root
    
    #tree construction part!
    for each in depend:
        if each[0] != 'conj_and':
            parentidx = each[1][1]
            childidx = each[2][1]
            relation = each[0]
            # print "parent:", parentidx, nodels[parentidx]
            # print "child:", childidx, nodels[childidx]
            nodels[parentidx].addchild(nodels[childidx]) #add a child to a node
            nodels[childidx].addparent(nodels[parentidx], relation) #add a parent to a node
            
            #add a edgefactor to two hidden variables with a relationship
            parent_word_q = word_q_check(nodels[parentidx].word, pold)
            parent_word_r = 'r' #need to change later
            child_word_baseform = snowball_stemmer.stem(nodels[childidx].word)
            parent_word_baseform = snowball_stemmer.stem(nodels[parentidx].word)
            edgefactor = EdgeFactor(nodels[parentidx], nodels[childidx], parent_word_q, parent_word_r, child_word_baseform, parent_word_baseform)
            nodels[parentidx].addchildfactor(edgefactor)
            nodels[childidx].addparentfactor(edgefactor)
                # print nodels[childidx], nodels[childidx].parents_edgefactor[0]            
                # print nodels[childidx].parents
    #BFS for topological order
    nodels[0].toplgod = 0
    treedepth = simplebfs(nodels[0])
    for nd in nodels: #remove non-informative words
        if nd.toplgod == float('-inf'):
            nodels.pop(nodels.index(nd))
    # print treedepth
    nodels = sorted(nodels, key=lambda nd: nd.toplgod, reverse=True) #sort the nodelost based on nodes' topological order
    return nodels, treedepth


def estimate(ndls, paradicN, paradicE, rtlabel):
    #forword stage, all messages passed to the root
    for nd in ndls[:-1]:
        upvec = [1.0, 1.0]
        #if nd.nodefactor != None:
        upvec[0], upvec[1] = nd.nodefactor.calnfv(paradicN) # a vector of two values from the node factor
        # print nodevec
        if nd.children_edgefactor != []:
            for childedf in nd.children_edgefactor:
                upvec[0] *= childedf.upvec[0] #positive polarity
                upvec[1] *= childedf.upvec[1] #negative polarity
        # print nd, nodevec
        for parentedf in nd.parents_edgefactor:
            factorvec = parentedf.calefv(paradicE)
            #marginalize current node
            parentedf.upvec[0] = factorvec[0]*upvec[0] + factorvec[1]*upvec[1] 
            parentedf.upvec[1] = factorvec[2]*upvec[0] + factorvec[3]*upvec[1]
    #backward stage, all messages passed back
    #two types of message of s0: the root label is given; and the root label is marginalized
    #first deal with the root
    rootinfo = ndls[-1].children_edgefactor[0].calefv(paradicE)
    #type1
    if rtlabel == '+':
        ndls[-1].children_edgefactor[0].downvec0 = [rootinfo[0], rootinfo[2]]
    elif rtlabel == '-':
        ndls[-1].children_edgefactor[0].downvec0 = [rootinfo[3], rootinfo[1]]
    #type2
    ndls[-1].children_edgefactor[0].downvec1 = [rootinfo[0]+rootinfo[3], rootinfo[1]+rootinfo[2]]
    # print ndls[-1].children_edgefactor[0], ndls[-1].children_edgefactor[0].downvec0
    for nd in reversed(ndls[:-1]):
        downvec0 = [1.0, 1.0]
        downvec1 = [1.0, 1.0]
        downvec0[0], downvec0[1] = nd.nodefactor.calnfv(paradicN) #type1
        downvec1[0], downvec1[1] = nd.nodefactor.calnfv(paradicN) #type2
        for parentedf in nd.parents_edgefactor:
            # print parentedf, parentedf.downvec0
            #parent, nodefactor product
            downvec0[0] *= parentedf.downvec0[0] #pos
            downvec0[1] *= parentedf.downvec0[1] #neg
            downvec1[0] *= parentedf.downvec1[0] #pos
            downvec1[1] *= parentedf.downvec1[1] #neg

        prod = [1.0, 1.0]
        if nd.children_edgefactor != []:
            for tempchildedf in nd.children_edgefactor: #product of all up messages
                    prod[0] *= tempchildedf.upvec[0]
                    prod[1] *= tempchildedf.upvec[1]

            #backward message
            for childedf in nd.children_edgefactor:                
                # factorvec = childedf.calefv(paradicE) #wait till need probability 
                #type1
                childedf.downvec0[0] = downvec0[0]*prod[0]/childedf.upvec[0] #+
                childedf.downvec0[1] = downvec0[1]*prod[1]/childedf.upvec[1] #-
                #type2
                childedf.downvec1[0] = downvec1[0]*prod[0]/childedf.upvec[0] #+
                childedf.downvec1[1] = downvec1[1]*prod[1]/childedf.upvec[1] #-
        #nodefactor's marginal
        nd.nodefactor.downvec0[0] = downvec0[0] * prod[0]
        nd.nodefactor.downvec0[1] = downvec0[1] * prod[1]

        nd.nodefactor.downvec1[0] = downvec1[0] * prod[0]
        nd.nodefactor.downvec1[1] = downvec1[1] * prod[1]
    #backforward propgation done

    #parameter estimate 
    

def inference(ndls, paradicN, paradicE):
    #belief propagation algo
    for nd in ndls[:-1]:
        nodevec = [1.0, 1.0]
        #if nd.nodefactor != None:
        nodevec[0], nodevec[1] = nd.nodefactor.calnfv(paradicN) # a vector of two values from the node factor
        # print nodevec
        if nd.children_edgefactor != []:
            for childedf in nd.children_edgefactor:
                nodevec[0] *= childedf.upvec[0] #positive polarity
                nodevec[1] *= childedf.upvec[1] #negative polarity
        # print nd, nodevec
        for parentedf in nd.parents_edgefactor:
            factorvec = parentedf.calefv(paradicE)
            parentedf.upvec[0] = factorvec[0]*nodevec[0] + factorvec[1]*nodevec[1] #marginalize current node
            parentedf.upvec[1] = factorvec[2]*nodevec[0] + factorvec[3]*nodevec[1]
    
    #get the polarity of the root
    rootpos = ndls[-1].children_edgefactor[0].upvec[0]
    rootneg = ndls[-1].children_edgefactor[0].upvec[1]
    if rootpos >= rootneg:
        sntpolarity = '+'
    elif rootpos < rootneg:
        sntpolarity = '-'

    return sntpolarity

def testfunc(parsedrev, poldic):
    #rev[0][3] is the review, and rev[0][2] is the polarity
    eg = parsedrev[0]
    nodels, treedepth = construct_tree(eg, poldic)
    '''
    ## small print test
    for nd in nodels:
        if nd.children_edgefactor != []:
            print nd, nd.toplgod, nd.children_edgefactor[0]
    '''
    paradicN = [{}, {}, {}, {}]
    paradicE = [{}, {}, {}, {}, {}]
    print 'sentence sentiment is %s' % inference(nodels, paradicN, paradicE)
    estimate(nodels, paradicN, paradicE, '+')
    
    










def main():
    if len(sys.argv) != 2:
        print 'Usage: %s [datafile path]' % sys.argv[0]
        sys.exit("please provide data file path")
    
    datafilename = sys.argv[1]
    # f = codecs.open(datafilename, 'rb', encoding="utf-8")
    f=open(datafilename, 'rb')
    parsedrev = cpcl.load(f)
    f.close()
    poldic = loadbinpoldic() #poldic[0] is positive, poldic[1] is negative
    testfunc(parsedrev, poldic)
                

if __name__ == "__main__":
    main()
