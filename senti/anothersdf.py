## Cosmo Zhang @ Purdue 11/2014
## cs578 final project on yelp
## Filename:anothersdf.py
## -*- coding: utf-8 -*-

import gc
import os
import codecs
import cPickle as cpcl
import sys
import numpy
reload(sys)
import nltk
# from nltk.parse.stanford import StanfordParser
from pprint import pprint
sys.setdefaultencoding("utf-8")
os.environ["STANFORD_PARSER"] = "/home/cosmo/Dropbox/Purdue/nlp/stanford-corenlp-full-2014-08-27/"
os.environ["CLASSPATH"] = os.environ["STANFORD_PARSER"] + "stanford-corenlp-3.4.1.jar:"+ os.environ["STANFORD_PARSER"] +"stanford-corenlp-3.4.1-models.jar"

import pln_inco.syntax_trees
import pln_inco.graphviz as gv
from IPython.display import *
import pln_inco.stanford_parser

def sdfprocess(rvdata, partidx):
    my_sentence='Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas.'
    pr = pln_inco.stanford_parser.lexicalized_parser_parse([my_sentence], model='englishPCFG',output='basicDependencies')
    for p in pr:
        print p
    tagged_sentences=pln_inco.stanford_parser.lexicalized_parser_tag([my_sentence],model='englishPCFG') 
    words_and_pos=[nltk.tag.str2tuple(t) for t in tagged_sentences[0].split()]
    malt_tab_rep=pln_inco.syntax_trees.stanford_dependency_to_malt_tab(pr[0],words_and_pos)
    print malt_tab_rep
    dg = nltk.dependencygraph.DependencyGraph(malt_tab_rep)  
    print dg.tree().pprint()
    dep_tree=pln_inco.syntax_trees.dependency_to_dot(dg)
    print dep_tree
    # tree_png=Image(data=gv.generate(dep_tree,format='png'))
    # display_png(tree_png)
    


'''
    sdfdata=[]
    cnn = 1
    for eg in rvdata:
        if cnn%100 == 0: print "%f%% of document %d finished" % (cnn*100*1.0/len(rvdata), partidx+1)
        cmt = eg[3].decode('utf-8') #3 is the idx of comment
        sentences = nltk.sent_tokenize(cmt)
        sdfparsed = parser.raw_parse_sents(sentences)
        sdfdata.append(eg[:3]+[sdfparsed])
        # print cnn
        print sdfparsed
        # print sdfdata
        cnn += 1        
        if cnn > 5: break
    # return sdfdata
'''

def main():
    if len(sys.argv) != 3:
        print 'Usage: %s [datafile path] [part number]' % sys.argv[0]
        sys.exit("please provide data file path")
    
    datafilename = sys.argv[1]
    # f = codecs.open(datafilename, 'rb', encoding="utf-8")
    f=open(datafilename, 'rb')
    revdata = cpcl.load(f)
    f.close()
    partlen = (len(revdata)-len(revdata)%9)/9
    # for i in range(10):
    i = int(sys.argv[2])-1
    if len(revdata[i*partlen:]) >= partlen: partdata = revdata[i*partlen:(i+1)*partlen]
    elif len(revdata[i*partlen:]) < partlen: partdata = revdata[i*partlen:]
    # print i*partlen, (i+1)*partlen, partlen, len(revdata)
    sdfdata = sdfprocess(partdata, i)
    # sdfdata = []
    # g = file('sdfdata.data', 'wb')
    g = file('../data/sdfdata'+str(i+1)+'.data', 'wb')
    cpcl.dump(sdfdata, g)
    g.close()
    del sdfdata
    gc.collect()
    print "Stanford Parser Process Done on part %d!" % (i+1)
    # print revdata[0] rev[0][3] is the review, and rev[0][2] is the polarity

if __name__ == "__main__":
    main()
