# -*- coding: iso-8859-1 -*-
# python code to construct bags of words from reviews.csv

import sys
import operator
import time
import re
from operator import itemgetter
import ast
import cPickle as cpl

bigrams = False

# read reviews
inputfile = sys.argv[1]
outputfile = sys.argv[2]

f = open(inputfile, 'rb')
dt = cpl.load(f)
#lines = f.readlines()
f.close()
# format: user?, <stars>,<review_text>

# count ALL the words
word_count = {} # dictionary of words
if bigrams:
	# initialize dictionary of bigrams
	#print('MISSING: bigram counting initialization'
	bi_words_count = {}
	# YOU NEED TO FILL IN THE DETAILS HERE

#print dt[1], type(dt[1])

cnt = 0
new_lines = []
for line in dt:
	row=line
	# get review text
	review = row[3]

	# contract contractions, e.g. i've --> ive
	review = review.replace("'","")

	# remove all non-word characters
	review = re.sub('[^\w]',' ',review.lower())

	# update line
	row[3] = review
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
#print l[1]

while len(l) <5000:
        l.append(('notpossiblecharrrrrrrrrrrrrrrrr', 0))

# go back through all reviews and create a bag of words for each
f2 = open(outputfile, 'w')
for line in new_lines:
	# split line on commas
	row = line.split(',')
	# get isPositive
	isPositive = (row[2]=='+')
	
	# get review text
	review = row[3].split()

	d = {}
	for w in review:
		d[w] = 1

	# make bag of words
	bow = [0]*5000
	for idx in range(5000):
		#print 'MISSING: bag of words transform'
                #print len(l)
		if row[3].find( l[idx][0] ) > 0 : bow[idx] = 1
     	# YOU NEED TO FILL IN THE DETAILS HERE

	if bigrams:
		#print 'MISSING: bag of bigrams transform'
		for idx in range(2500,5000):
		    if row[3].find( k[idx][0] ) > 0 : bow[idx] = 1
		# YOU NEED TO FILL IN THE DETAILS HERE

	# assemble record
	record = []
	record.append(int(isPositive))
	
	for idx in range(len(bow)):
		record.append(bow[idx])

	# write record to file

	record.append(row[0])
	record.append(row[1])
	f2.write(','.join(map(str,record)) + '\n')

f2.close()
print 'Data generated'
