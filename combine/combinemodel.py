## Cosmo Zhang @ Purdue 12/2014
## cs578 final project on yelp
## Filename:combinemodel
## -*- coding: utf-8 -*-
import sys
import numpy as np
reload(sys)


def crfsymboltoint(line):
    res = line.replace("\n", "").split(" ")
    if res[1] == '+':
        return 1
    elif res[1] == '-':
        return 0
    



def nbsymboltoint(line):
    res = line.replace("\n", "").split(",")
    return int(res[0]), int(res[1])

def findalpha(nbresfile, crfresfile):
    f = open(nbresfile, "r")
    nbres = f.readlines()
    f.close()

    g = open(crfresfile, "r")
    crfres = g.readlines()
    g.close()

    lsnb =[]
    lscrf = []
    lsreal = []
    
    for (eachnb, eachcrf) in zip(nbres, crfres):
        crfres = crfsymboltoint(eachcrf)
        realv, nbres = nbsymboltoint(eachnb)
        lsnb.append(nbres)
        lscrf.append(crfres)
        lsreal.append(realv)

    # print lsnb, lscrf, lsreal
    prevaccuracy = 0.0
    bestalpha = 0.0
    for alpha in np.linspace(0, 1, 101):
        correctcount = 0
        for idx in range(len(lsreal)):
            if lscrf[idx] * alpha + lsnb[idx] * (1-alpha) >= 0.5:
                pred = 1
            elif lscrf[idx] * alpha + lsnb[idx] * (1-alpha) < 0.5:
                pred = 0
            if pred == lsreal[idx]:
                correctcount += 1
        accuracy = correctcount*1.0/len(lsreal)
        # print alpha, accuracy
        if accuracy >= prevaccuracy:
            prevaccuracy = accuracy
            bestalpha = alpha

    return bestalpha

def finaltest(nbresfile, crfresfile, alpha):
    f = open(nbresfile, "r")
    nbres = f.readlines()
    f.close()

    g = open(crfresfile, "r")
    crfres = g.readlines()
    g.close()

    lsnb =[]
    lscrf = []
    lsreal = []
    
    for (eachnb, eachcrf) in zip(nbres, crfres):
        crfres = crfsymboltoint(eachcrf)
        realv, nbres = nbsymboltoint(eachnb)
        lsnb.append(nbres)
        lscrf.append(crfres)
        lsreal.append(realv)
        
    correctcount =  0.0
    tp = 0.0
    actualp = 0.0
    predp = 0.0
    for idx in range(len(lsreal)):
        if lscrf[idx] * alpha + lsnb[idx] * (1-alpha) >= 0.5:
            pred = 1
            predp += 1
        elif lscrf[idx] * alpha + lsnb[idx] * (1-alpha) < 0.5:
            pred = 0
        if pred == lsreal[idx]:
            correctcount += 1
            if pred == 1:
                tp += 1
        if lsreal[idx] == 1:
            actualp += 1

        
    accuracy, recall, precision = correctcount*1.0/len(lsreal), tp/actualp, tp/predp
    fscore = 2.0*(recall*precision)/(recall + precision)
    print "\naccuracy, recall, precision, fscore on test is %0.2f%%, %0.2f%%, %0.2f%%, %0.2f%%" % (accuracy*100, recall*100, precision*100, fscore*100)
    
    

def main():
    if len(sys.argv) != 5:
        print 'Usage: %s [nb results val file path] [ldcrf results val file path] [nb test file path] [crf test file path]' % sys.argv[0]
        sys.exit("please provide data file path correctly")

    nbresfileval = sys.argv[1]
    crfresfileval = sys.argv[2]

    bestalpha = findalpha(nbresfileval, crfresfileval)
    print bestalpha

    nbtestfile = sys.argv[3]
    crftestfile = sys.argv[4]
    
    finaltest(nbtestfile, crftestfile, bestalpha)
    
   
    


if __name__ == "__main__":
    main()
