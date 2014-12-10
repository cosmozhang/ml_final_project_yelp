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
        return self.parent.word + '~' + self.child.word

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
        self.idx = None
        self.pol = None
        self.word = word
        self.parents_edgefactor = []
        self.children_edgefactor = []
        self.nodefactor = None
        self.tag = tag 
        self.toplgod = float("-inf")
        self.toparentrelation = {}

    def __str__(self):
        return self.word + ' ' + str(self.idx)
    
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
    # try:
    return - log((a + b*exp(sumop-x-sum_other))/(c + d*exp(sumop-x-sum_other))) + 0.5*x**2
    '''
    except:
        print a, b, c, d
        print 'nd', (a + b*exp(sumop-x-sum_other)), (c + d*exp(sumop-x-sum_other))
        # break
    '''    
        
#nodefactor deravative that we sent to BFGS
def ndobjf_der(x, sum_other, sumop, a, b, c, d):
    return - a/(a + b*exp(sumop - x -sum_other)) + c/(c + d*exp(sumop - x -sum_other)) + x

#edgefactor objective function that we minimize by BFGS
def edobjf(x, sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h):
    # try:
    return - log((a + b*exp(sumop1-x-sum_other) + c*exp(sumop2-x-sum_other) + d*exp(sumop2-x-sum_other))/(e + f*exp(sumop1-x-sum_other) + g*exp(sumop2-x-sum_other) + h*exp(sumop3-x-sum_other))) + 0.5*x**2
    '''
    except:
        print a, b, c, d, e, f, g, h
        print 'ed', (a + b*exp(sumop1-x-sum_other) + c*exp(sumop2-x-sum_other) + d*exp(sumop2-x-sum_other)), (e + f*exp(sumop1-x-sum_other) + g*exp(sumop2-x-sum_other) + h*exp(sumop3-x-sum_other))
        # break
    '''

#edgefactor deravative that we sent to BFGS
def edobjf_der(x, sum_other, sumop1, sumop2, sumop3, a, b, c, d, e, f, g, h):
    return - a/(a + b*exp(sumop1-x-sum_other) + c*exp(sumop2-x-sum_other) + d*exp(sumop2-x-sum_other)) + e/(e +  f*exp(sumop1-x-sum_other) + g*exp(sumop2-x-sum_other) + h*exp(sumop3-x-sum_other)) + x

def simplebfs(nd):
    if nd.children != []:
        depthls = []
        for child in nd.children:
            # print child
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

def construct_tree(snt, snowball_stemmer, pold, revd):
    wtp, depend = reformat(snt)
    # print wtp
    # print depend
    # print '\n'
    nodels = []
    # ndfactorls = []
    for item in wtp:
        if item[0] != 'ROOT':
            try:
                newnode = Node(item[0], item[1])
            except:
                print snt, wtp
                sys.exit()
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
            # ndfactorls.append(nodefactor)
        # print 'node', newnode
        nodels.append(newnode) #nodels[0] is the root

    # for nd in nodels:
        # print nd
    #tree construction part!
    edfactorls = []
    # print '\n'
    for each in depend:
        # print each
        if each[0] != 'conj_and' and each[0] != 'rcmod' and each[0] != 'conj_but' and each[0] != 'conj_or' and each[0] != 'conj_as' and each[0] != 'conj_times' and  each[0] != 'vmod' and each[0] != 'conj_only' and each[0] != 'conj_just':
            # print each
            parentidx = each[1][1]
            nodels[parentidx].idx = parentidx
            childidx = each[2][1]
            nodels[childidx].idx = childidx
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
    try:
        treedepth = simplebfs(nodels[0]) #a simple bfs
    except:
        print depend
        sys.exit()
    cndfactorls = []
    cnodels = filter(lambda x: x.toplgod != float('-inf'), nodels)
    # for nd in cnodels:
        # print nd, nd.toplgod
    for nd in cnodels: #remove non-informative words
        if nd.toplgod != 0: #non root node has factor
            cndfactorls.append(nd.nodefactor)
    # for nd in cndfactorls:
        # print nd
    # print treedepth
    cnodels = sorted(cnodels, key=lambda nd: nd.toplgod, reverse=True) #sort the nodelist based on nodes' topological order
    
    for cnd in cnodels[:-1]:
        cnd.parents = filter(lambda x: x.toplgod != float('-inf'), cnd.parents)
        cnd.parents_edgefactor = filter(lambda x: x.parent.toplgod != float('-inf'), cnd.parents_edgefactor)
            
    cedfactorls = filter(lambda x: x.parent.toplgod != float('-inf') and x.child.toplgod != float('-inf'), edfactorls)
    # print '\n'
    # for nd in cndfactorls:
        # print nd
    # print len(cnodels)
    return cnodels, cndfactorls, cedfactorls, treedepth


def estimate(ndls, ndfactorls, edfactorls, paradicN, paradicE, rtlabel):

    #forword stage, all messages passed to the root
    #! information stored on the edgefactor as upvec is from node to factor
    for nd in ndls[:-1]:
        prodvec = [1.0, 1.0] #factor to node information
        #if nd.nodefactor != None:
        # try:
        nfvec = nd.nodefactor.calnfv(paradicN) # a vector of two values from the node to factor
        '''
        except:
            for each in ndls:
                print each.toplgod
        '''        
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
        # try:
        ndls[-1].children_edgefactor[0].downvec0 = [1.0, 0.0]
        '''
        except:
            print '\n'
            for nd in ndls:
                print nd
            sys.exit()
        '''    
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
            # try:
            prodvec0[0] *= (parentedf.downvec0[0]*efvec[0] + parentedf.downvec0[1]*efvec[2]) #pos: ++ & -+
            prodvec0[1] *= (parentedf.downvec0[1]*efvec[3] + parentedf.downvec0[0]*efvec[1]) #neg: -- & +-
            prodvec1[0] *= (parentedf.downvec1[0]*efvec[0] + parentedf.downvec1[1]*efvec[2]) #pos: ++ & -+
            prodvec1[1] *= (parentedf.downvec1[1]*efvec[3] + parentedf.downvec1[0]*efvec[1]) #neg: -- & +-
            '''    
            except:
                print '\n', nd
                for parentedf in nd.parents_edgefactor:
                    print parentedf, parentedf.downvec0, parentedf.downvec1, parentedf.parent.toplgod
                for nd in reversed(ndls[:-1]):
                    print nd, nd.toplgod
                for end in edfactorls:
                    print end
                # sys.exit()
            '''
    
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

def test_func(valdata, paradicN, paradicE, poldic, revdic):
    snowball_stemmer = SnowballStemmer("english")
    match_sum = 0.0
    truep_sum = 0.0
    prep_sum = 0.0
    for eg in valdata:
        tlb = eg[2]
        pcount, ncount = 0, 0
        for snt in eg[3]:
            nodels, ndfactorls, edfactorls, treedepth = construct_tree(snt, snowball_stemmer, poldic, revdic)
            if len(nodels) > 1:
                sntp = inference(nodels, paradicN, paradicE)
            elif len(nodels) <= 1:
                tossacoin = uniform(0.0, 1.0)
                if tossacoin > 0.5:
                    sntp = '+'
                elif tossacoin <= 0.5:
                    sntp = '-'
            if sntp == '+':
                pcount += 1
            elif sntp == '-':
                ncount += 1
        if pcount >= ncount: prelb = '+'
        elif pcount < ncount: prelb = '-'

        if prelb == tlb: match_sum += 1
            
        if prelb == '+': prep_sum += 1
        if tlb == '+': truep_sum += 1
        acc, recall, precision = 1.0*match_sum/len(valdata), 1.0*match_sum/truep_sum, 1.0*match_sum/prep_sum #calculate accuracy, recall, precision
        fscore = 2*(recall*precision)/(recall + precision)
    return acc, recall, precision, fscore

def dlcrf_func(snowball_stemmer, eg, poldic, revdic, paradicN, paradicE):
    
    #rev[0][3] is the review, and rev[0][2] is the polarity
    # eg = parsedrev[0]
    labl = eg[2]
    # print labl
    for snt in eg[3]:
        nodels, ndfactorls, edfactorls, treedepth = construct_tree(snt, snowball_stemmer, poldic, revdic)
        if len(nodels) > 1:
            estimate(nodels, ndfactorls, edfactorls, paradicN, paradicE, labl)
            
    # print 'sentence sentiment is %s' % inference(nodels, paradicN, paradicE)
    # print paradicE, paradicN

def train(paradicN, paradicE, poldic, revdic, traindata, epochs, valdata):
    snowball_stemmer = SnowballStemmer("english")
    prevvaliacc = 0.0
    epoch = 1
    valiaccls = []
    while True:
        time_b_v = time.clock()
        print "***************\nthis is epoch %d of epochs %d" % (epoch, epochs)
        widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=len(traindata)).start()
        for idx in range(len(traindata)):
            eg = traindata[idx]
            #print eg
            dlcrf_func(snowball_stemmer, eg, poldic, revdic, paradicN, paradicE)
            pbar.update(idx + 1)
        pbar.finish()               

        valiacc, valirecall, valiprecision, valifscore = test_func(valdata, paradicN, paradicE, poldic, revdic)
        valiaccls.append(valiacc)
        print "\naccuracy, recall, precision, fscore on validation is %0.2f%%, %0.2f%%, %0.2f%%, %0.2f%%" % (valiacc*100, valirecall*100, valiprecision*100, valifscore*100)
        if epoch > epochs: #early stop creterion
            if (epoch - epochs)%1 == 0:
                if valiacc <= prevvaliacc: break
        prevvaliacc = valiacc
        # print "\nused time in this epoch is %f\n*****************" % (time.clock() - time_b_v) #timing
        epoch += 1
    return (epoch, valiaccls)

def main():
    if len(sys.argv) != 5:
        print 'Usage: %s [traindatafile path] [validationdatafile path] [testdatafile path] [epochs]' % sys.argv[0]
        sys.exit("please provide data file path")
    epochs = int(sys.argv[4])
    trainfilename, valfilename, testfilename = sys.argv[1], sys.argv[2], sys.argv[3]
    # f = codecs.open(datafilename, 'rb', encoding="utf-8")
    f=open(trainfilename, 'rb')
    traindata = cpcl.load(f)
    f.close()
    
    f=open(valfilename, 'rb')
    valdata = cpcl.load(f)
    f.close()

    f=open(testfilename, 'rb')
    testdata = cpcl.load(f)
    f.close()

    #load dictionaries
    poldic, revdic = loadbinpoldic(), loadrevdic() #poldic[0] is positive, poldic[1] is negative
    paradicN = [{}, {}, {}, {}, {}]
    paradicE = [{}, {}, {}, {}, {}]
    # print parsedrev[2][3][3][2]
    # testfunc(parsedrev, poldic, revdic, paradicN, paradicE)
    train(paradicN, paradicE, poldic, revdic, traindata, epochs, valdata)
    fnlacc, fnlrecall, fnlprecision, fnlfscore = test_func(testdata, paradicN, paradicE, poldic, revdic)
    print "\naccuracy, recall, precision, fscore on test is %0.2f%%, %0.2f%%, %0.2f%%, %0.2f%%" % (fnlacc*100, fnlrecall*100, fnlprecision*100, fnlfscore*100)

if __name__ == "__main__":
    main()
