## Cosmo Zhang @ Purdue 11/2014
## cs578 final project on yelp
## Filename:txttoformat.py
## -*- coding: utf-8 -*-
import sys

def reformat(cnt):
    wtp  = [tuple(x.rsplit('/', 1)) for x in cnt[0][0].split(" ")] #word/tag
    # print wtp, '\n'
    wtp.insert(0, tuple([u'ROOTNODE', u'S'])) #add the root 'word'
    # print wtp, '\n'

    #dependency relation reformat
    step1 = [x.strip(')').split('(',1) for x in cnt[2]]
    step2 = [[x[0]] + x[1].split(', ') for x in step1]
    step3 = [[x[0]] + x[1].rsplit('-', 1) + x[2].rsplit('-', 1) for x in step2]
    # try:
    depend = [[x[0]] + [tuple([x[1], int(x[2].replace("'", ""))])] + [tuple([x[3], int(x[4].replace("'", ""))])] for x in step3]
    '''
    except:
        print step3
        sys.exit()
    '''
    return wtp, depend

def main():
    print "test me"

if __name__ == "__main__":
    main()
