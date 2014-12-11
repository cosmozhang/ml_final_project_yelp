# -*- coding: iso-8859-1 -*-
# python code to construct bags of words from reviews.csv

import sys
import operator
import time
import re
from operator import itemgetter
import ast


bigrams = False
for arg in sys.argv[1:]:
	if arg == '-bigrams':
		bigrams = True
	else:
		print 'usage: ', sys.argv[0], '[-bigrams]'
		sys.exit(2)

# read reviews
f = open('dataset_500.csv','rt')
lines = f.readlines()
f.close()
# format: user?, <stars>,<review_text>

# count ALL the words
word_count = {} # dictionary of words
if bigrams:
	# initialize dictionary of bigrams
	#print('MISSING: bigram counting initialization'
	bi_words_count = {}
	# YOU NEED TO FILL IN THE DETAILS HERE


cnt = 0
new_lines = []
for line in lines:
	# split line on commas
	row = line.split(',')

	# get review text
	review = row[2]

	# contract contractions, e.g. i've --> ive
	review = review.replace("'","")

	# remove all non-word characters
	review = re.sub('[^\w]',' ',review.lower())

	# update line
	row[2] = review
	new_lines.append(','.join(map(str,row)))

	# split into words
	review = review.split()

	# increment word counts
	# print 'MISSING: word counting'
	for w in review:
		 if word_count.has_key(w): word_count[w] = word_count[w] + 1
		 else: word_count[w] = 1
		# YOU NEED TO FILL IN THE DETAILS HERE

	if bigrams:
		# increment bigram counts
		#print 'MISSING: bigram counting'
		for i in range(len(review)-1):
		    w = review[i].join(' ').join(review[i+1])
		    if bi_words_count.has_key(w): bi_words_count[w] = bi_words_count[w]+1
		    else: bi_words_count[w]=1
		# YOU NEED TO FILL IN THE DETAILS HERE

	cnt += 1

l = {}
k = {}
# find the most frequent words
if not bigrams:
# sort and take first 5000
	l = sorted(word_count.items(), key=itemgetter(1),reverse=True)
	# YOU NEED TO FILL IN THE DETAILS HERE
else:
	# sort
	# take first 2500 words and 2500 bigrams
	#print 'MISSING: bigram sorting and feature selection'
	l = sorted(word_count.items(), key=itemgetter(1),reverse=True)
	k = sorted(bi_words_count.items(), key=itemgetter(1),reverse=True)
	# YOU NEED TO FILL IN THE DETAILS HERE


# go back through all reviews and create a bag of words for each
f2 = open('data.csv','w')
for line in new_lines:
	# split line on commas
	row = line.split(',')
	# get isPositive
	isPositive = (row[1]==' +')

	# get review text
	review = row[2].split()

	d = {}
	for w in review:
		d[w] = 1

	# make bag of words
	bow = [0]*5000
	for idx in range(5000):
		#print 'MISSING: bag of words transform'
		if row[2].find( l[idx][0] ) > 0 : bow[idx] = 1
     	# YOU NEED TO FILL IN THE DETAILS HERE

	if bigrams:
		#print 'MISSING: bag of bigrams transform'
		for idx in range(2500,5000):
		    if row[2].find( k[idx][0] ) > 0 : bow[idx] = 1
		# YOU NEED TO FILL IN THE DETAILS HERE

	# assemble record
	record = []
	record.append(int(isPositive))
	
	for idx in range(len(bow)):
		record.append(bow[idx])

	# write record to file
	f2.write(','.join(map(str,record)) + '\n')

f2.close()
print 'Data generated'
