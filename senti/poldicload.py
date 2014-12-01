## Cosmo Zhang @ Purdue 11/2014
## CS578 final project on yelp
## filename: poldicload.py
## -*- coding: utf-8 -*-

def loadbinpoldic():
    posils = []
    f = open('./liu_dic/positive-words.txt', 'r')
    worddata = f.readlines()
    for line in worddata:
        posils.append(line.strip())
        
    negals = []
    f = open('./liu_dic/negative-words.txt', 'r')
    worddata = f.readlines()
    for line in worddata:
        negals.append(line.strip())

    return [posils] + [negals]
    

def main():
    print "module test"
    res = loadbinpoldic()
    print "print some posi words"
    print res[0][0], res[0][1], res[0][2]
    print "print some nega words"
    print res[1][0], res[1][1], res[1][2]

if __name__ == "__main__":
    main()
