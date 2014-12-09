## Cosmo Zhang @ Purdue 11/2014
## cs578 final project on yelp
## Filename:dlcrf.py
## -*- coding: utf-8 -*-

import os
import codecs
import cPickle as cpcl
import sys
import numpy
import time
from scipy.optimize import fmin_bfgs
reload(sys)
import nltk
from random import uniform
from poldicload import loadbinpoldic, loadrevdic #my own module
from txttoformat import reformat
from math import exp, log
from progressbar import *
from nltk.stem import SnowballStemmer
sys.setdefaultencoding("utf-8")


#global


class NodeFactor(object):
    def __init__(self, parent, word_q, word_r, word_baseform, postag):
        self.type = 'nodefactor'
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
        #parameters for featureset s
        if '+' not in paradicN[0]:
            paradicN[0]['+'] = uniform(-1.0, 1.0)
        if '-' not in paradicN[0]:
            paradicN[0]['-'] = uniform(-1.0, 1.0)
        #parameters for featureset q
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
        #parameters for featureset q & r
        if '+' + self.word_q + self.word_r not in paradicN[4]:
            paradicN[4]['+' + self.word_q + self.word_r] = uniform(-1.0, 1.0)
        if '-' + self.word_q + self.word_r not in paradicN[3]:
            paradicN[4]['-' + self.word_q + self.word_r] = uniform(-1.0, 1.0)

        #calculate probabilities of both posi and neg
        self.posv = exp(paradicN[0]['+'] + paradicN[1]['+' + self.word_q] + paradicN[2]['+' + self.postag] + paradicN[3]['+' + self.word_baseform] + paradicN[4]['+' + self.word_q + self.word_r])
        self.negv = exp(paradicN[0]['-'] + paradicN[1]['-' + self.word_q] + paradicN[2]['-' + self.postag] + paradicN[3]['-' + self.word_baseform] + paradicN[4]['-' + self.word_q + self.word_r])
        # self.pos_p = sum_pos/(sum_pos + sum_neg)
        # self.neg_p = sum_neg/(sum_pos + sum_neg)
        return (self.posv, self.negv)

class EdgeFactor(object):
    def __init__(self, parent, child, parent_word_q, parent_word_r, child_word_baseform, parent_word_baseform):
        self.type = 'edgefactor'
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
        self.upvec = [None, None] #parent-child: ++, +-, -+, --
        self.downvec0 = [None, None] #parent-child: ++, +-, -+, --
        self.downvec1 = [None, None] #parent-child: ++, +-, -+, --

    def __str__(self):
        return self.parent_word_baseform + '~' + self.child_word_baseform

    def calefv(self, paradicE):
        #initialization of parameters if it is not in the parameter dictionary
        #parameters for featureset s
        if '++' not in paradicE[0]:
            paradicE[0]['++'] = uniform(0.9, 1.1)
        if '--' not in paradicE[0]:
            paradicE[0]['--'] = uniform(0.9, 1.1)
        if '+-' not in paradicE[0]:
            paradicE[0]['+-'] = uniform(-1.0, 1.0)
        if '-+' not in paradicE[0]:
            paradicE[0]['-+'] = uniform(-1.0, 1.0)
   
        #parameters for featureset r
        if '++' + self.parent_word_r not in paradicE[1]:
            paradicE[1]['++' + self.parent_word_r] = uniform(-1.0, 1.0)
        if '--' + self.parent_word_r not in paradicE[1]:
            paradicE[1]['--' + self.parent_word_r] = uniform(-1.0, 1.0)
        if '+-' + self.parent_word_r not in paradicE[1]:
            paradicE[1]['+-' + self.parent_word_r] = uniform(-1.0, 1.0)
        if '-+' + self.parent_word_r not in paradicE[1]:
            paradicE[1]['-+' + self.parent_word_r] = uniform(-1.0, 1.0)

        #parameters for featureset r & q
        if '++' + self.parent_word_r + self.parent_word_q not in paradicE[2]:
            paradicE[2]['++' + self.parent_word_r + self.parent_word_q] = uniform(-1.0, 1.0)
        if '--' + self.parent_word_r + self.parent_word_q not in paradicE[2]:
            paradicE[2]['--' + self.parent_word_r + self.parent_word_q] = uniform(-1.0, 1.0)
        if '+-' + self.parent_word_r + self.parent_word_q not in paradicE[2]:
            paradicE[2]['+-' + self.parent_word_r + self.parent_word_q] = uniform(-1.0, 1.0)
        if '-+' + self.parent_word_r + self.parent_word_q not in paradicE[2]:
            paradicE[2]['-+' + self.parent_word_r + self.parent_word_q] = uniform(-1.0, 1.0)

        #parameters for parent_featureset parent word
        if '++' + self.parent_word_baseform not in paradicE[3]:
            paradicE[3]['++' + self.parent_word_baseform] = uniform(-1.0, 1.0)
        if '--' + self.parent_word_baseform not in paradicE[3]:
            paradicE[3]['--' + self.parent_word_baseform] = uniform(-1.0, 1.0)
        if '+-' + self.parent_word_baseform not in paradicE[3]:
            paradicE[3]['+-' + self.parent_word_baseform] = uniform(-1.0, 1.0)
        if '-+' + self.parent_word_baseform not in paradicE[3]:
            paradicE[3]['-+' + self.parent_word_baseform] = uniform(-1.0, 1.0)

        #parameters for child_featureset child word
        if '++' + self.child_word_baseform not in paradicE[4]:
            paradicE[4]['++' + self.child_word_baseform] = uniform(-1.0, 1.0)
        if '--' + self.child_word_baseform not in paradicE[4]:
            paradicE[4]['--' + self.child_word_baseform] = uniform(-1.0, 1.0)
        if '+-' + self.child_word_baseform not in paradicE[4]:
            paradicE[4]['+-' + self.child_word_baseform] = uniform(-1.0, 1.0)
        if '-+' + self.child_word_baseform not in paradicE[4]:
            paradicE[4]['-+' + self.child_word_baseform] = uniform(-1.0, 1.0)

        #calculate probabilities of both posi and neg
        self.pos_posv = exp(paradicE[0]['++'] + paradicE[1]['++' + self.parent_word_r] + paradicE[2]['++' + self.parent_word_r + self.parent_word_q] + paradicE[3]['++' + self.parent_word_baseform] + paradicE[4]['++' + self.child_word_baseform])
        self.neg_negv = exp(paradicE[0]['--'] + paradicE[1]['--' + self.parent_word_r] + paradicE[2]['--' + self.parent_word_r + self.parent_word_q] + paradicE[3]['--' + self.parent_word_baseform] + paradicE[4]['--' + self.child_word_baseform])
        self.pos_negv = exp(paradicE[0]['+-'] + paradicE[1]['+-' + self.parent_word_r] + paradicE[2]['+-' + self.parent_word_r + self.parent_word_q] + paradicE[3]['+-' + self.parent_word_baseform] + paradicE[4]['+-' + self.child_word_baseform])
        self.neg_posv = exp(paradicE[0]['-+'] + paradicE[1]['-+' + self.parent_word_r] + paradicE[2]['-+' + self.parent_word_r + self.parent_word_q] + paradicE[3]['-+' + self.parent_word_baseform] + paradicE[4]['-+' + self.child_word_baseform])
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

#nodefactor objective function that we minimize by BFGS
def ndobjf(x, sum_other, sumop, a, b, c, d):
    try:
        return - log((a + b*exp(sumop-x-sum_other))/(c + d*exp(sumop-x-sum_other))) + 0.5*x**2
    except:
        print a, b, c, d
        print 'nd', (a + b*exp(sumop-x-sum_other)), (c + d*exp(sumop-x-sum_other))
        # break
        
#nodefactor deravative that we sent to BFGS
def ndobjf_der(x, sum_other, sumop, a, b, c, d):
    return - a/(a + b*exp(sumop - x -sum_other)) + c/(c + d*exp(sumop - x -sum_other)) + x

#edgefactor objective function that we minimize by BFGS
def edobjf(x, sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h):
    try:
        return - log((a + b*exp(sumop1-x-sum_other) + c*exp(sumop2-x-sum_other) + d*exp(sumop2-x-sum_other))/(e + f*exp(sumop1-x-sum_other) + g*exp(sumop2-x-sum_other) + h*exp(sumop3-x-sum_other))) + 0.5*x**2
    except:
        print a, b, c, d, e, f, g, h
        print 'ed', (a + b*exp(sumop1-x-sum_other) + c*exp(sumop2-x-sum_other) + d*exp(sumop2-x-sum_other)), (e + f*exp(sumop1-x-sum_other) + g*exp(sumop2-x-sum_other) + h*exp(sumop3-x-sum_other))
        # break

#edgefactor deravative that we sent to BFGS
def edobjf_der(x, sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h):
    return - a/(a + b*exp(sumop1-x-sum_other) + c*exp(sumop2-x-sum_other) + d*exp(sumop2-x-sum_other)) + e/(e +  f*exp(sumop1-x-sum_other) + g*exp(sumop2-x-sum_other) + h*exp(sumop3-x-sum_other)) + x

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

def word_r_check(wd, revd):
    if wd in revd:
        return 'r+'
    else:
        return 'r-'

def construct_tree(eg, stemmer, pold, revd):
    wtp, depend = reformat(eg[3])
    # print wtp
    # print depend
    nodels = []
    ndfactorls = []
    for item in wtp:
        if item[0] != 'ROOT':
            newnode = Node(item[0], item[1])
        else:
            newnode = Node('rootnode', item[1])
        word_q = word_q_check(item[0], pold)
        # print word_q
        word_r = word_r_check(item[0], revd) 
        word_baseform = snowball_stemmer.stem(item[0])
        if item[0] != 'ROOT': 
            #nodefactor initialize and attach them to the hidden varibles
            nodefactor = NodeFactor(newnode, word_q, word_r, word_baseform, item[1])         
            newnode.nodefactor = nodefactor
            ndfactorls.append(nodefactor)
        nodels.append(newnode) #nodels[0] is the root
    
    #tree construction part!
    edfactorls = []
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
            parent_word_r = word_r_check(nodels[parentidx].word, revd) 
            child_word_baseform = snowball_stemmer.stem(nodels[childidx].word)
            parent_word_baseform = snowball_stemmer.stem(nodels[parentidx].word)
            edgefactor = EdgeFactor(nodels[parentidx], nodels[childidx], parent_word_q, parent_word_r, child_word_baseform, parent_word_baseform)
            nodels[parentidx].addchildfactor(edgefactor)
            nodels[childidx].addparentfactor(edgefactor)
            edfactorls.append(edgefactor)
            # print nodels[childidx], nodels[childidx].parents_edgefactor[0]            
            # print nodels[childidx].parents
    #BFS for topological order
    nodels[0].toplgod = 0
    treedepth = simplebfs(nodels[0])
    for nd in nodels: #remove non-informative words
        if nd.toplgod == float('-inf'):
            nodels.pop(nodels.index(nd))
            ndfactorls.pop(ndfactorls.index(nd.nodefactor))
    # print treedepth
    nodels = sorted(nodels, key=lambda nd: nd.toplgod, reverse=True) #sort the nodelost based on nodes' topological order
    return nodels, ndfactorls, edfactorls, treedepth


def estimate(ndls, ndfactorls, edfactorls, paradicN, paradicE, rtlabel):

    #forword stage, all messages passed to the root
    #! information stored on the edgefactor as upvec is from node to factor
    for nd in ndls[:-1]:
        prodvec = [1.0, 1.0] #factor to node information
        #if nd.nodefactor != None:
        nfvec = nd.nodefactor.calnfv(paradicN) # a vector of two values from the node to factor
        # print nfvec
        if nd.children_edgefactor != []:
            for childedf in nd.children_edgefactor:
                efvec = childedf.calefv(paradicE) #this contains four values (parent, child: ++, +-, -+, --)
                prodvec[0] *= (childedf.upvec[0]*efvec[0] + childedf.upvec[1]*efvec[1]) #positive polarity: ++ & +-
                prodvec[1] *= (childedf.upvec[0]*efvec[2] + childedf.upvec[1]*efvec[3]) #negative polarity: -+ & --
        # print nd, nodevec
        for parentedf in nd.parents_edgefactor:            
            #marginalized later, each edge factor keeps the information of two values 
            parentedf.upvec[0] = nfvec[0]*prodvec[0] #+
            parentedf.upvec[1] = nfvec[1]*prodvec[1] #-
    #'''
    #backward stage, all messages passed back
    #! information stored on the edgefactor as down vec is from node to factor
    #two types of message of s0: the root label is given; and the root label is marginalized
    #first deal with the root
    #type1
    if rtlabel == '+':
        ndls[-1].children_edgefactor[0].downvec0 = [1.0, 0.0]
    elif rtlabel == '-':
        ndls[-1].children_edgefactor[0].downvec0 = [0.0, 1.0]
    #type2
    ndls[-1].children_edgefactor[0].downvec1 = [1.0, 1.0]

    for nd in reversed(ndls[:-1]):
        prodvec0 = [1.0, 1.0] #parent-child: +, -
        prodvec1 = [1.0, 1.0] #parent-child: +, -  
        nfvec = nd.nodefactor.calnfv(paradicN) #node factor, a vector of two
        
        for parentedf in nd.parents_edgefactor: #node's parent factor
            # print parentedf, parentedf.downvec0
            #parent, nodefactor product
            efvec = parentedf.calefv(paradicE)
            #marginalize here
            prodvec0[0] *= (parentedf.downvec0[0]*efvec[0] + parentedf.downvec0[1]*efvec[2]) #pos: ++ & -+
            prodvec0[1] *= (parentedf.downvec0[1]*efvec[3] + parentedf.downvec0[0]*efvec[1]) #neg: -- & +-
            prodvec1[0] *= (parentedf.downvec1[0]*efvec[0] + parentedf.downvec1[1]*efvec[2]) #pos: ++ & -+
            prodvec1[1] *= (parentedf.downvec1[1]*efvec[3] + parentedf.downvec1[0]*efvec[1]) #neg: -- & +-

        if nd.children_edgefactor != []:
            for childedf in nd.children_edgefactor: #product of all up messages
                efvec = childedf.calefv(paradicE)
                #marginalize
                prodvec0[0] *= (childedf.upvec[0]*efvec[0] + childedf.upvec[1]*efvec[2]) #pos: ++ & -+
                prodvec0[1] *= (childedf.upvec[1]*efvec[3] + childedf.upvec[0]*efvec[1]) #neg: -- & +-
                prodvec1[0] *= (childedf.upvec[0]*efvec[0] + childedf.upvec[1]*efvec[2]) #pos: ++ & -+
                prodvec1[1] *= (childedf.upvec[1]*efvec[3] + childedf.upvec[0]*efvec[1]) #neg: -- & +-

            #backward message
            for childedf in nd.children_edgefactor:
                efvec = childedf.calefv(paradicE)
                # factorvec = childedf.calefv(paradicE) #wait till need probability 
                #type1
                childedf.downvec0[0] = nfvec[0]*prodvec0[0]/(childedf.upvec[0]*efvec[0] + childedf.upvec[1]*efvec[2]) #+
                childedf.downvec0[1] = nfvec[1]*prodvec0[1]/(childedf.upvec[1]*efvec[3] + childedf.upvec[0]*efvec[1]) #-

                #type2
                childedf.downvec1[0] = nfvec[0]*prodvec1[0]/(childedf.upvec[0]*efvec[0] + childedf.upvec[1]*efvec[2]) #+
                childedf.downvec1[1] = nfvec[1]*prodvec1[1]/(childedf.upvec[1]*efvec[3] + childedf.upvec[0]*efvec[1]) #-

        #nodefactor's marginal
        nd.nodefactor.downvec0[0] = prodvec0[0]
        nd.nodefactor.downvec0[1] = prodvec0[1]

        nd.nodefactor.downvec1[0] = prodvec1[0]
        nd.nodefactor.downvec1[1] = prodvec1[1]
    #backforward propgation done
    #'''

    #use bfgs to learn parameters
    #first the node factors
    for ndfactor in ndfactorls:
        
        # for '+':
        sumop = paradicN[0]['-'] + paradicN[1]['-' + ndfactor.word_q] + paradicN[2]['-' + ndfactor.postag] + paradicN[3]['-' + ndfactor.word_baseform]
        a,b,c,d = ndfactor.downvec0[0], ndfactor.downvec0[1], ndfactor.downvec1[0], ndfactor.downvec1[1]

        #parameters for featureset s
        # for nodefeature '+':
        sum_other = paradicN[1]['+' + ndfactor.word_q] + paradicN[2]['+' + ndfactor.postag] + paradicN[3]['+' + ndfactor.word_baseform]
        paradicN[0]['+'] = fmin_bfgs(f=ndobjf, fprime=ndobjf_der, x0=paradicN[0]['+'], args=(sum_other, sumop, a, b, c, d), disp = False)


        #parameters for featureset q
        # for nodefeature '+' + ndfactor.word_q:
        sum_other = paradicN[0]['+'] + paradicN[2]['+' + ndfactor.postag] + paradicN[3]['+' + ndfactor.word_baseform]
        paradicN[1]['+' + ndfactor.word_q] = fmin_bfgs(f=ndobjf, fprime=ndobjf_der, x0=paradicN[1]['+' + ndfactor.word_q], args=(sum_other, sumop, a, b, c, d), disp = False)

        #parameters for featureset tag
        # for nodefeature '+' + ndfactor.postag:
        sum_other = paradicN[0]['+'] + paradicN[1]['+' + ndfactor.word_q] + paradicN[3]['+' + ndfactor.word_baseform]
        paradicN[2]['+' + ndfactor.postag] = fmin_bfgs(f=ndobjf, fprime=ndobjf_der, x0=paradicN[2]['+' + ndfactor.postag], args=(sum_other, sumop, a, b, c, d), disp = False)

        #parameters for featureset word
        # for nodefeature '+' + ndfactor.word_baseform:
        sum_other = paradicN[0]['+'] + paradicN[1]['+' + ndfactor.word_q] + paradicN[2]['+' + ndfactor.postag]
        paradicN[3]['+' + ndfactor.word_baseform] = fmin_bfgs(f=ndobjf, fprime=ndobjf_der, x0=paradicN[3]['+' + ndfactor.word_baseform], args=(sum_other, sumop, a, b, c, d), disp = False)
        
        #for '-':
        sumop = paradicN[0]['+'] + paradicN[1]['+' + ndfactor.word_q] + paradicN[2]['+' + ndfactor.postag] + paradicN[3]['+' + ndfactor.word_baseform]
        a,b,c,d = ndfactor.downvec0[1], ndfactor.downvec0[0], ndfactor.downvec1[1], ndfactor.downvec1[0]

        #parameters for featureset s
        # for nodefeature '-':
        sum_other = paradicN[1]['-' + ndfactor.word_q] + paradicN[2]['-' + ndfactor.postag] + paradicN[3]['-' + ndfactor.word_baseform]
        paradicN[0]['-'] = fmin_bfgs(f=ndobjf, fprime=ndobjf_der, x0=paradicN[0]['-'], args=(sum_other, sumop, a, b, c, d), disp = False)
        
        #parameters for featureset q
        # for nodefeature '-' + ndfactor.word_q:
        sum_other = paradicN[0]['-'] + paradicN[2]['-' + ndfactor.postag] + paradicN[3]['-' + ndfactor.word_baseform]
        paradicN[1]['-' + ndfactor.word_q] = fmin_bfgs(f=ndobjf, fprime=ndobjf_der, x0=paradicN[1]['-' + ndfactor.word_q], args=(sum_other, sumop, a, b, c, d), disp = False)
        
        #parameters for featureset tag
        # for nodefeature '-'+ ndfactor.postag:
        sum_other = paradicN[0]['-'] + paradicN[1]['-' + ndfactor.word_q] + paradicN[3]['-' + ndfactor.word_baseform]
        paradicN[2]['-' + ndfactor.postag] = fmin_bfgs(f=ndobjf, fprime=ndobjf_der, x0=paradicN[2]['-' + ndfactor.postag], args=(sum_other, sumop, a, b, c, d), disp = False)

        #parameters for featureset word
        # for nodefeature '-'+ ndfactor.word_baseform:
        sum_other = paradicN[0]['-'] + paradicN[1]['-' + ndfactor.word_q] + paradicN[2]['-' + ndfactor.postag]
        paradicN[3]['-' + ndfactor.word_baseform] = fmin_bfgs(f=ndobjf, fprime=ndobjf_der, x0=paradicN[3]['-' + ndfactor.word_baseform], args=(sum_other, sumop, a, b, c, d), disp = False)


    #second the edge factors
    for edfactor in edfactorls:
        
        #for '++':
        sumop1 = paradicE[0]['--'] + paradicE[1]['--' + edfactor.parent_word_r] + paradicE[2]['--' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['--' + edfactor.parent_word_baseform] + paradicE[4]['--' + edfactor.child_word_baseform]
        sumop2 = paradicE[0]['+-'] + paradicE[1]['+-' + edfactor.parent_word_r] + paradicE[2]['+-' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['+-' + edfactor.parent_word_baseform] + paradicE[4]['+-' + edfactor.child_word_baseform]
        sumop3 = paradicE[0]['-+'] + paradicE[1]['-+' + edfactor.parent_word_r] + paradicE[2]['-+' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['-+' + edfactor.parent_word_baseform] + paradicE[4]['-+' + edfactor.child_word_baseform]
        a, b, c, d = edfactor.downvec0[0]*edfactor.upvec[0], edfactor.downvec0[1]*edfactor.upvec[1], edfactor.downvec0[0]*edfactor.upvec[1], edfactor.downvec0[1]*edfactor.upvec[0]
        e, f, g, h = edfactor.downvec1[0]*edfactor.upvec[0], edfactor.downvec1[1]*edfactor.upvec[1], edfactor.downvec1[0]*edfactor.upvec[1], edfactor.downvec1[1]*edfactor.upvec[0]

        #parameters for featureset s
        #for edgefeature '++'
        sum_other = paradicE[1]['++' + edfactor.parent_word_r] + paradicE[2]['++' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['++' + edfactor.parent_word_baseform] + paradicE[4]['++' + edfactor.child_word_baseform]
        paradicE[0]['++'] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[0]['++'], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)
        
        #parameters for featureset parent_r
        #for edgefeature '++' & parent_r
        sum_other = paradicE[0]['++'] + paradicE[2]['++' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['++' + edfactor.parent_word_baseform] + paradicE[4]['++' + edfactor.child_word_baseform]
        paradicE[1]['++' + edfactor.parent_word_r] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[1]['++' + edfactor.parent_word_r], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_r & and parent_q
        #for edgefeature '++' & parent_r & parent_q
        sum_other = paradicE[0]['++'] + paradicE[1]['++' + edfactor.parent_word_r] + paradicE[3]['++' + edfactor.parent_word_baseform] + paradicE[4]['++' + edfactor.child_word_baseform]
        paradicE[2]['++' + edfactor.parent_word_r + edfactor.parent_word_q] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[2]['++' + edfactor.parent_word_r + edfactor.parent_word_q], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_word_baseform
        #for edgefeature '++' & parent_word_baseform
        sum_other = paradicE[0]['++'] + paradicE[1]['++' + edfactor.parent_word_r] + paradicE[2]['++' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[4]['++' + edfactor.child_word_baseform]
        paradicE[3]['++' + edfactor.parent_word_baseform] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[3]['++' + edfactor.parent_word_baseform], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_r
        #for edgefeature '++' & parent_r
        sum_other = paradicE[0]['++'] + paradicE[1]['++' + edfactor.parent_word_r] + paradicE[2]['++' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['++' + edfactor.parent_word_baseform]
        paradicE[4]['++' + edfactor.child_word_baseform] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[4]['++' + edfactor.child_word_baseform], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #for '--':
        sumop1 = paradicE[0]['++'] + paradicE[1]['++' + edfactor.parent_word_r] + paradicE[2]['++' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['++' + edfactor.parent_word_baseform] + paradicE[4]['++' + edfactor.child_word_baseform]
        sumop2 = paradicE[0]['+-'] + paradicE[1]['+-' + edfactor.parent_word_r] + paradicE[2]['+-' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['+-' + edfactor.parent_word_baseform] + paradicE[4]['+-' + edfactor.child_word_baseform]
        sumop3 = paradicE[0]['-+'] + paradicE[1]['-+' + edfactor.parent_word_r] + paradicE[2]['-+' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['-+' + edfactor.parent_word_baseform] + paradicE[4]['-+' + edfactor.child_word_baseform]
        a, b, c, d = edfactor.downvec0[1]*edfactor.upvec[1], edfactor.downvec0[0]*edfactor.upvec[0], edfactor.downvec0[0]*edfactor.upvec[1], edfactor.downvec0[1]*edfactor.upvec[0]
        e, f, g, h = edfactor.downvec1[1]*edfactor.upvec[1], edfactor.downvec1[0]*edfactor.upvec[0], edfactor.downvec1[0]*edfactor.upvec[1], edfactor.downvec1[1]*edfactor.upvec[0]
        
        #parameters for featureset s
        #for edgefeature '--'
        sum_other = paradicE[1]['--' + edfactor.parent_word_r] + paradicE[2]['--' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['--' + edfactor.parent_word_baseform] + paradicE[4]['--' + edfactor.child_word_baseform]
        paradicE[0]['--'] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[0]['--'], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_r
        #for edgefeature '--' & parent_r
        sum_other = paradicE[0]['--'] + paradicE[2]['--' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['--' + edfactor.parent_word_baseform] + paradicE[4]['--' + edfactor.child_word_baseform]
        paradicE[1]['--' + edfactor.parent_word_r] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[1]['--' + edfactor.parent_word_r], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_r & and parent_q
        #for edgefeature '--' & parent_r & parent_q
        sum_other = paradicE[0]['--'] + paradicE[1]['--' + edfactor.parent_word_r] + paradicE[3]['--' + edfactor.parent_word_baseform] + paradicE[4]['--' + edfactor.child_word_baseform]
        paradicE[2]['--' + edfactor.parent_word_r + edfactor.parent_word_q] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[2]['--' + edfactor.parent_word_r + edfactor.parent_word_q], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_word_baseform
        #for edgefeature '--' & parent_word_baseform
        sum_other = paradicE[0]['--'] + paradicE[1]['--' + edfactor.parent_word_r] + paradicE[2]['--' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[4]['--' + edfactor.child_word_baseform]
        paradicE[3]['--' + edfactor.parent_word_baseform] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[3]['--' + edfactor.parent_word_baseform], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_r
        #for edgefeature '--' & parent_r
        sum_other = paradicE[0]['--'] + paradicE[1]['--' + edfactor.parent_word_r] + paradicE[2]['--' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['--' + edfactor.parent_word_baseform]
        paradicE[4]['--' + edfactor.child_word_baseform] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[4]['--' + edfactor.child_word_baseform], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        
        #for '+-':
        sumop1 = paradicE[0]['--'] + paradicE[1]['--' + edfactor.parent_word_r] + paradicE[2]['--' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['--' + edfactor.parent_word_baseform] + paradicE[4]['--' + edfactor.child_word_baseform]
        sumop2 = paradicE[0]['++'] + paradicE[1]['++' + edfactor.parent_word_r] + paradicE[2]['++' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['++' + edfactor.parent_word_baseform] + paradicE[4]['+-' + edfactor.child_word_baseform]
        sumop3 = paradicE[0]['-+'] + paradicE[1]['-+' + edfactor.parent_word_r] + paradicE[2]['-+' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['-+' + edfactor.parent_word_baseform] + paradicE[4]['-+' + edfactor.child_word_baseform]
        a, b, c, d = edfactor.downvec0[0]*edfactor.upvec[1], edfactor.downvec0[1]*edfactor.upvec[1], edfactor.downvec0[0]*edfactor.upvec[0], edfactor.downvec0[1]*edfactor.upvec[0]
        e, f, g, h = edfactor.downvec1[0]*edfactor.upvec[1], edfactor.downvec1[1]*edfactor.upvec[1], edfactor.downvec1[0]*edfactor.upvec[0], edfactor.downvec1[1]*edfactor.upvec[0]

        #parameters for featureset s
        #for edgefeature '+-'
        sum_other = paradicE[1]['+-' + edfactor.parent_word_r] + paradicE[2]['+-' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['+-' + edfactor.parent_word_baseform] + paradicE[4]['++' + edfactor.child_word_baseform]
        paradicE[0]['+-'] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[0]['+-'], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)
        
        #parameters for featureset parent_r
        #for edgefeature '+-' & parent_r
        sum_other = paradicE[0]['+-'] + paradicE[2]['+-' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['+-' + edfactor.parent_word_baseform] + paradicE[4]['+-' + edfactor.child_word_baseform]
        paradicE[1]['+-' + edfactor.parent_word_r] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[1]['+-' + edfactor.parent_word_r], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_r & and parent_q
        #for edgefeature '+-' & parent_r & parent_q
        sum_other = paradicE[0]['+-'] + paradicE[1]['+-' + edfactor.parent_word_r] + paradicE[3]['+-' + edfactor.parent_word_baseform] + paradicE[4]['+-' + edfactor.child_word_baseform]
        paradicE[2]['+-' + edfactor.parent_word_r + edfactor.parent_word_q] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[2]['+-' + edfactor.parent_word_r + edfactor.parent_word_q], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_word_baseform
        #for edgefeature '+-' & parent_word_baseform
        sum_other = paradicE[0]['+-'] + paradicE[1]['+-' + edfactor.parent_word_r] + paradicE[2]['+-' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[4]['+-' + edfactor.child_word_baseform]
        paradicE[3]['+-' + edfactor.parent_word_baseform] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[3]['+-' + edfactor.parent_word_baseform], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_r
        #for edgefeature '+-' & parent_r
        sum_other = paradicE[0]['+-'] + paradicE[1]['+-' + edfactor.parent_word_r] + paradicE[2]['+-' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['+-' + edfactor.parent_word_baseform]
        paradicE[4]['+-' + edfactor.child_word_baseform] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[4]['+-' + edfactor.child_word_baseform], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)


        #for '-+':
        sumop1 = paradicE[0]['--'] + paradicE[1]['--' + edfactor.parent_word_r] + paradicE[2]['--' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['--' + edfactor.parent_word_baseform] + paradicE[4]['--' + edfactor.child_word_baseform]
        sumop2 = paradicE[0]['+-'] + paradicE[1]['+-' + edfactor.parent_word_r] + paradicE[2]['+-' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['+-' + edfactor.parent_word_baseform] + paradicE[4]['+-' + edfactor.child_word_baseform]
        sumop3 = paradicE[0]['++'] + paradicE[1]['++' + edfactor.parent_word_r] + paradicE[2]['++' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['++' + edfactor.parent_word_baseform] + paradicE[4]['++' + edfactor.child_word_baseform]
        a, b, c, d = edfactor.downvec0[1]*edfactor.upvec[0], edfactor.downvec0[1]*edfactor.upvec[1], edfactor.downvec0[0]*edfactor.upvec[1], edfactor.downvec0[1]*edfactor.upvec[1]
        e, f, g, h = edfactor.downvec1[1]*edfactor.upvec[0], edfactor.downvec1[1]*edfactor.upvec[1], edfactor.downvec1[0]*edfactor.upvec[1], edfactor.downvec1[1]*edfactor.upvec[1]

        #parameters for featureset s
        #for edgefeature '-+'
        sum_other = paradicE[1]['-+' + edfactor.parent_word_r] + paradicE[2]['-+' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['-+' + edfactor.parent_word_baseform] + paradicE[4]['-+' + edfactor.child_word_baseform]
        paradicE[0]['-+'] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[0]['-+'], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)
        
        #parameters for featureset parent_r
        #for edgefeature '-+' & parent_r
        sum_other = paradicE[0]['-+'] + paradicE[2]['-+' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['-+' + edfactor.parent_word_baseform] + paradicE[4]['-+' + edfactor.child_word_baseform]
        paradicE[1]['-+' + edfactor.parent_word_r] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[1]['-+' + edfactor.parent_word_r], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_r & and parent_q
        #for edgefeature '-+' & parent_r & parent_q
        sum_other = paradicE[0]['-+'] + paradicE[1]['-+' + edfactor.parent_word_r] + paradicE[3]['-+' + edfactor.parent_word_baseform] + paradicE[4]['-+' + edfactor.child_word_baseform]
        paradicE[2]['-+' + edfactor.parent_word_r + edfactor.parent_word_q] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[2]['-+' + edfactor.parent_word_r + edfactor.parent_word_q], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_word_baseform
        #for edgefeature '-+' & parent_word_baseform
        sum_other = paradicE[0]['-+'] + paradicE[1]['-+' + edfactor.parent_word_r] + paradicE[2]['-+' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[4]['-+' + edfactor.child_word_baseform]
        paradicE[3]['-+' + edfactor.parent_word_baseform] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[3]['-+' + edfactor.parent_word_baseform], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)

        #parameters for featureset parent_r
        #for edgefeature '-+' & parent_r
        sum_other = paradicE[0]['-+'] + paradicE[1]['-+' + edfactor.parent_word_r] + paradicE[2]['-+' + edfactor.parent_word_r + edfactor.parent_word_q] + paradicE[3]['-+' + edfactor.parent_word_baseform]
        paradicE[4]['-+' + edfactor.child_word_baseform] = fmin_bfgs(f=edobjf, fprime=edobjf_der, x0=paradicE[4]['-+' + edfactor.child_word_baseform], args=(sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h), disp = False)


def inference(ndls, paradicN, paradicE):
    #belief propagation algo
    #just need forword stage in inference, all messages passed to the root
    #! information stored on the edgefactor as upvec is from node to factor
    for nd in ndls[:-1]:
        prodvec = [1.0, 1.0] #factor to node information
        #if nd.nodefactor != None:
        nfvec = nd.nodefactor.calnfv(paradicN) # a vector of two values from the node to factor
        # print nodevec
        if nd.children_edgefactor != []:
            for childedf in nd.children_edgefactor:
                efvec = childedf.calefv(paradicE) #this contains four values (parent, child: ++, +-, -+, --)
                prodvec[0] *= (childedf.upvec[0]*efvec[0] + childedf.upvec[1]*efvec[1]) #positive polarity: ++ & +-
                prodvec[1] *= (childedf.upvec[0]*efvec[2] + childedf.upvec[1]*efvec[3]) #negative polarity: -+ & --
        # print nd, nodevec
        for parentedf in nd.parents_edgefactor:            
            #marginalized later, each edge factor keeps the information of two values 
            parentedf.upvec[0] = nfvec[0]*prodvec[0] #+
            parentedf.upvec[1] = nfvec[1]*prodvec[1] #-
    
    #deal with the root:
    efvec = ndls[-1].children_edgefactor[0].calefv(paradicE) #this contains four values (parent, child: ++, +-, -+, --)
    #get the polarity of the root
    rootpos = ndls[-1].children_edgefactor[0].upvec[0] * efvec[0] + ndls[-1].children_edgefactor[0].upvec[0] * efvec[1]
    rootneg = ndls[-1].children_edgefactor[0].upvec[1] * efvec[2] + ndls[-1].children_edgefactor[0].upvec[1] * efvec[3]
    if rootpos >= rootneg:
        sntpolarity = '+'
    elif rootpos < rootneg:
        sntpolarity = '-'

    return sntpolarity

def testfunc(parsedrev, poldic, revdic, paradicN, paradicE):
    snowball_stemmer = SnowballStemmer("english")
    #rev[0][3] is the review, and rev[0][2] is the polarity
    # eg = parsedrev[0]
    for eg in parsedrev:
        nodels, ndfactorls, edfactorls, treedepth = construct_tree(eg, snowball_stemmer, poldic, revdic)
        '''
        ## small print test
        print 'ndfactorls'
        for nd in ndfactorls:
        print nd.downvec0, nd.downvec1
        print 'edfactorls'
        for nd in edfactorls:
        print nd.upvec, nd.downvec0, nd.downvec1
        '''
        estimate(nodels, ndfactorls, edfactorls, paradicN, paradicE, '+')
        '''
        ## small print test
        print 'ndfactorls'
        for nd in ndfactorls:
        print nd.downvec0, nd.downvec1
        print 'edfactorls'
        for nd in edfactorls:
        print nd.upvec, nd.downvec0, nd.downvec1
        '''
    print 'sentence sentiment is %s' % inference(nodels, paradicN, paradicE)
    

def main():
    if len(sys.argv) != 2:
        print 'Usage: %s [datafile path]' % sys.argv[0]
        sys.exit("please provide data file path")
    
    datafilename = sys.argv[1]
    # f = codecs.open(datafilename, 'rb', encoding="utf-8")
    f=open(datafilename, 'rb')
    parsedrev = cpcl.load(f)
    f.close()
    #load dictionaries
    poldic, revdic = loadbinpoldic(), loadrevdic() #poldic[0] is positive, poldic[1] is negative
    paradicN = [{}, {}, {}, {}, {}]
    paradicE = [{}, {}, {}, {}, {}]
    # print parsedrev[2][3][3][2]
    testfunc(parsedrev, poldic, revdic, paradicN, paradicE)
                

if __name__ == "__main__":
    main()
