# -*- coding: iso-8859-1 -*-

import time
import csv
import sys
import random
import math


classIdx = 0
num_feature = 5000
attrIdxs = range(1,num_feature+1)

smoothing = True
squaredloss = False
verbose = False

		
		

	
# learn NBC
def learnNBC(f_train):
	# computes prior and CPDs from training data
	# returns a NBC
	N = len(f_train)
	#print f_train[0]
	pos_prior_n = 0.0
	neg_prior_n = 0.0
	user=[]
	bus=[]
	for index in f_train:
		if index[0] == '1':
			pos_prior_n += 1.0
		else:
			neg_prior_n += 1.0
		#print index
		user.append(index[num_feature+1])
		bus.append(index[num_feature+2])
	##### obtain prior P(+) and P(-)
	pos_prior=pos_prior_n/N
	neg_prior=neg_prior_n/N
	
	#### find user feature
	#### user and bus set
	user_feature=list(set(user))
	bus_feature=list(set(bus))
	#### user num
	user_n=len(user_feature)
	bus_n=len(bus_feature)
	
	
	####compute p(user|+), p(user|-), p(bus|+) and p(bus|-)
	neg_cond_bus_n=[0.0] * bus_n
	pos_cond_bus_n=[0.0] * bus_n
	neg_cond_user_n=[0.0] * user_n
	pos_cond_user_n=[0.0] * user_n
	
	cond_bus_p=[]
	cond_user_p=[]
	cond_bus_n=[]
	cond_user_n=[]
	for i in range(bus_n):
		for index in f_train:
			if index[num_feature+2]==bus_feature[i]:
				if index[0]=='1':
					pos_cond_bus_n[i] += 1.0
				if index[0]=='0':
					neg_cond_bus_n[i] += 1.0
		cond_bus_p.append(1.0*pos_cond_bus_n[i]/pos_prior_n)	
		cond_bus_n.append(1.0*neg_cond_bus_n[i]/neg_prior_n)
	for j in range(user_n):
		for index in f_train:
			if index[num_feature+1]==user_feature[j]:
				if index[0]=='1':
					pos_cond_user_n[j] += 1.0
				if index[0]=='0':
					neg_cond_user_n[j] += 1.0
		cond_user_p.append(1.0*pos_cond_user_n[j]/pos_prior_n)				
		cond_user_n.append(1.0*neg_cond_user_n[j]/neg_prior_n)
	return pos_prior, neg_prior, bus_feature, cond_bus_p, cond_bus_n, user_feature, cond_user_p, cond_user_n
	
def applyNBC(pos_prior, neg_prior, bus_feature, cond_bus_p, cond_bus_n, user_feature, cond_user_p, cond_user_n, f_test):
	result = []
	user_n=len(user_feature)
	bus_n=len(bus_feature)
	for test in f_test:
		p_pos_u=1.0
		p_pos_b=1.0
		p_neg_u=1.0
		p_neg_b=1.0
		for u in range(user_n):
			if test[num_feature+1]==user_feature[u]:
				p_pos_u=cond_user_p[u]
				p_neg_u=cond_user_n[u]
			for b in range(bus_n):
				if test[num_feature+2]==bus_feature[b]:
					p_pos_b=cond_bus_p[b]
					p_neg_b=cond_bus_n[b]
		p_pos=pos_prior*p_pos_u*p_pos_b
		p_neg=neg_prior*p_neg_u*p_neg_b
		if p_pos>p_neg:
			result.append([test[0],'1'])
		if p_pos<=p_neg:
			result.append([test[0],'0'])
	return result
		
		
		
testfile = sys.argv[2]
f = open(testfile,'rt')
f_test = []
while True:
	line = f.readline()
	if not line: break
	f_test.append(line.replace('\n', '').split(','))
#print datals[0][5001], type(datals[0])


trainfile = sys.argv[1]
g= open(trainfile,'rt')
f_train = []
while True:
	line = g.readline()
	if not line: break
	f_train.append(line.replace('\n', '').split(','))
# create partitions


####in predIterm [true, pred]
def allresult(predItems):
	match=0
	truepos=0 ###true positive
	totalpos=0 #### num predicted as true
	num=len(predItems)
	realpos=0 #### num of real true in training data
	
	for i in predItems:
		if (i[0] == i[1]):
			match=match+1	
			if i[0]=='1':
				truepos=truepos+1
		if i[1]=='1':
			totalpos=totalpos+1
		if i[0]=='1':
			realpos=realpos+1
	accuracy=float(match)/num
	precision=float(truepos)/totalpos
	recall=float(truepos)/realpos
	f1=2.0*(precision*recall)/(precision+recall)
	overallpos=float(realpos)/num
	#print realpos, num
	return accuracy, recall, precision, f1, overallpos

	
def writeresult(predItems, outputfile):
	f2 = open(outputfile,'w')
	for item in predItems:
		f2.write(','.join(map(str,item)) + '\n')
	
# main function
def main():


	print 'Accuracy\tRecall\tPrecision\tF1\tOverall'
	
	pos_prior, neg_prior, bus_feature, cond_bus_p, cond_bus_n, user_feature, cond_user_p, cond_user_n = learnNBC(f_train)
	#print '0'
	# apply NBC
	resultData = applyNBC(pos_prior, neg_prior, bus_feature, cond_bus_p, cond_bus_n, user_feature, cond_user_p, cond_user_n, f_test)
	#print '1'
	writeresult(resultData, sys.argv[3])
	evl=allresult(resultData)
	accuracy=evl[0]
	recall=evl[1]
	precision=evl[2]
	f1=evl[3]
	overallpos=evl[4]
	print "%0.2f%%, %0.2f%%, %0.2f%%, %0.2f%%, %0.2f%%" % (100*accuracy,100*recall,100*precision,100*f1,100*overallpos)
	
if __name__=='__main__':
	main()
