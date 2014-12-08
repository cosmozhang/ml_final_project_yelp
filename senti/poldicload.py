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
    
def loadrevdic():
  dwords = []
  f = open('./inquirer_dic/Decreas.txt', 'r')
  for line in f.readlines():
    line = line.strip('\n')
    if ' ' not in line and '#' in line:
        line = line.split('#')[0]
        dwords.append(line.lower())
  dwords = filter(None, dwords)
  f.close()

  nwords = []
  f = open('./inquirer_dic/NotLw.txt', 'r')
  for line in f.readlines():
    line = line.strip('\n')
    if ' ' not in line and '#' in line:
        line = line.split('#')[0]
        nwords.append(line.lower())
  nwords = filter(None, nwords)
  f.close()
  return list(set(dwords).union(nwords))

def main():
    print "module test"
    pres = loadbinpoldic()
    print "print some posi words"
    print pres[0][0], pres[0][1], pres[0][2]
    print "print some nega words"
    print pres[1][0], pres[1][1], pres[1][2]

    rres = loadrevdic()
    print "print some r words"
    print rres[0:10]

if __name__ == "__main__":
    main()
