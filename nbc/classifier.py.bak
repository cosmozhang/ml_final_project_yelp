# -*- coding: iso-8859-1 -*-

import time
import csv
import sys
import random
import math

classIdx = 0
attrIdxs = range(1,5001)

smoothing = True
squaredloss = False
verbose = False
for arg in sys.argv[1:]:
	if arg == '--nosmoothing' or arg == '-nosmoothing' or arg == '-n':
		smoothing = False;
	elif arg == '--squaredloss' or arg == '-squaredloss' or arg == '-s':
		squaredloss = True;
	elif arg == '--verbose' or arg == '-verbose' or arg == '-v':
		verbose = True
	else:
		print 'usage: ', sys.argv[0], '[--nosmoothing] [--squaredloss] [--verbose]'
		sys.exit(2)

# create partitions
def createPartitions(f):
	# splits data into ten random partitions
	indexl = range(0, 34012)
	random.shuffle(indexl)
	lines = f.readlines()
	partition_idx = []
	partition = []
	# first 8 partitions are of size of 3401
	# last 2 partitions are of size of 3402
	for i in range(0, 8):
		partition_idx.append(indexl[i*3401 : (i+1)*3401])
	partition_idx.append(indexl[8*3401 : 9*3401+1])
	partition_idx.append(indexl[9*3401+1 : 34012])
	for j in partition_idx:
		data = []
		for k in j:
			data.append(lines[k].strip().replace(',', ''))
		partition.append(data)
	return partition

# learn NBC
def learnNBC(f_train):
	# computes prior and CPDs from training data
	# returns a NBC
	N = len(f_train)
	prior_n = 0.0
	neg_prior_n=0.0
	cond_n = [0] * 5000
	neg_cond_n = [0] * 5000
	for index in f_train:
		if index[0] == '1':
			prior_n += 1.0
		else:
			neg_prior_n += 1.0
		for i in range(5000):
			if index[i+1] == '1' and index[0] == '1':
				cond_n[i] += 1
			elif index[i+1] == '1' and index[0] == '0':
				neg_cond_n[i] += 1
    	epsilon = 10 ** (-10)
	prior_p, neg_prior_p = 0.0, 0.0
	cond_p = [0.0] * 5000
	neg_cond_p = [0.0] * 5000
	if smoothing:
		if prior_n == 0.0 or neg_prior_n == 0.0:
			prior_n += 1.0
			neg_prior_n += 1.0
		for i in range(5000):
			cond_p[i] = (cond_n[i] + 1.0) / (prior_n + 2.0)
			neg_cond_p[i] = (neg_cond_n[i] + 1.0) / (neg_prior_n + 2.0)
		prior_p = prior_n / N
		#print 'prior_p:', prior_p
		neg_prior_p = neg_prior_n / N
    	else:
		for i in range(0, 5000):
			cond_p[i] = (cond_n[i] + epsilon) / prior_n
			neg_cond_p[i] = (neg_cond_n[i] + epsilon) / neg_prior_n
		prior_p = prior_n  / N
		neg_prior_p = neg_prior_n / N
	#print 'Naive Bayes model training finished.'
	return (prior_p, neg_prior_p, cond_p, neg_cond_p)

# apply NBC to test data
def applyNBC(prior_p, neg_prior_p, cond_p, neg_cond_p, f_test):
	# applies the NBC to test data
	# returns a list of [predClass,trueClass]
	result = []
	log_prior, log_neg_prior = 0.0, 0.0
	for test in f_test:
		trueClass = test[0]
		for i in range(5000):
			if test[i + 1] == '1':
				log_prior += math.log(cond_p[i])
				log_neg_prior += math.log(neg_cond_p[i])
			else:
				log_prior += math.log(1 - cond_p[i])
				log_neg_prior += math.log(1 - neg_cond_p[i])
		log_prior += math.log(prior_p)
		log_neg_prior += math.log(neg_prior_p)
		predClass = ''
		p = 0.0
		p_p = 1.0 / (1.0 + math.e ** (log_neg_prior - log_prior))
		p_n = 1.0 - p_p
		if log_prior >= log_neg_prior:
			predClass = '1'
			p = p_p
		else:
			predClass = '0'
			p = p_n
		log_prior, log_neg_prior = 0, 0
		result.append([predClass, trueClass, p])
	#print 'Naive Bayes predication finished.'	
	return result

# score predictions
def evaluatePredictions(predItems):
	# computes loss
	loss = 0
	counter = 0
	for i in predItems:
		# compute loss and add it to running total
		if not squaredloss:
			if (i[0] != i[1]):
				loss += 1
		else:
            		if (i[0] == i[1]):
				loss += (1-i[2]) ** 2
			else:
				loss += i[2] ** 2
		counter += 1.0
	return loss / counter

# print out results
def printResults(tss,loss):
	# compute mean of loss
	m = 0
	for elm in loss:
		m += elm
	m = m / len(loss) * 1.0

	# compute stdev
	loss2 = 0
	for elm in loss:
		loss2 += (elm - m) ** 2
	loss2 = loss2 / (len(loss) - 1)
	s = math.sqrt(loss2)

	# compute std error
	e = s / math.sqrt(len(loss))
	print str(tss) + '\t' + str(m) + '\t' + str(e)

# main function
def main():

	f = open('data.csv','rt')
	partition = createPartitions(f)
	f.close()

	print 'TSS\tLOSS\tSTD ERROR'

	for num_tr_idx in range(1, 10):
		if verbose:
			print str(num_tr_idxs)+' training partitions'
			print '%d%% is completed.' % (10*num_tr_idx)
		tss = 0
		loss = []
		for test_set_idx in range(10):
			if verbose:
				print 'test set #'+str(test_set_idx)
			# select test set partitions
			f_test = partition[test_set_idx]
			# build train idx
			train_idx = range(0, 10)
			train_idx.remove(test_set_idx)
			random.shuffle(train_idx)
			train_idx_list=train_idx
			train_idx = train_idx_list[0:num_tr_idx]
			f_train = []
			for k in train_idx:
				f_train.extend(partition[k])

				# train on data partitions
				prior_p, neg_prior_p, cond_p, neg_cond_p = learnNBC(f_train)
				tss = len(f_train)
    
			# apply NBC
			resultData = applyNBC(prior_p, neg_prior_p, cond_p, neg_cond_p, f_test)

			loss.append(evaluatePredictions(resultData))

		printResults(tss,loss)

if __name__='__main__':
	main()
