# Completes the user-rating matrix using social network data

import cPickle
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
import copy

h = open('../data/user_user_network_500.data', 'r')
uuNetwork = cPickle.load(h)
h.close()
h = open('../data/dataset_500.data', 'r')
ubRatings = cPickle.load(h)
h.close()

usrs = {line[0] for line in ubRatings}
bizs = {line[1] for line in ubRatings}
usrDict = dict(zip(usrs, range(len(usrs))))
bizDict = dict(zip(bizs, range(len(bizs))))

# Adjacency matrix (0-1, asymmetric)
network = np.zeros((len(usrs), len(usrs)), dtype = 'int')
for u1 in uuNetwork.keys():
	for u2 in uuNetwork[u1]:  # List of neighbors 
		network[usrDict[u1], usrDict[u2]] = 1
# float(np.sum(network)) / np.prod(network.shape) # Sparsity

uuNetworkNew = {usrDict[key]:uuNetwork[key] for key in uuNetwork.keys()}

# ind =[]
# for i in range(network.shape[0]):
# 	if np.sum(network[i,:]) + np.sum(network[:,i]) == 0:
# 		ind.append(i)

# Rating matrix
ratings = np.zeros((len(usrs), len(bizs)))
for line in ubRatings:
	if line[2] == '+':
		ratings[usrDict[line[0]], bizDict[line[1]]] = 1
	else:
		ratings[usrDict[line[0]], bizDict[line[1]]] = -1
# np.sum(ratings) / np.prod(ratings.shape) # Sparsity

train = rand.binomial(1, .7, ratings.shape) * ratings
trainCopy = copy.deepcopy(train)

# alpha = .1 # Learning rate
# np.sum(train) / np.prod(train.shape) # Sparsity
for k in range(200):  # Number of passes
	for i in range(train.shape[0]):
		friends = np.nonzero(network[i,:])
		if len(friends) > 0:
			for j in range(train.shape[1]):
				if trainCopy[i,j] == 0:
					friendRatings = train[friends,j]
					friendRatings = friendRatings[friendRatings > 0]
					if len(friendRatings) > 0:
						train[i,j] = np.mean(friendRatings)

# Evaluation
np.sum(np.abs(ratings - train) * np.abs(ratings - trainCopy)) / np.sum(ratings - trainCopy != 0)
# 0.47551789077212808

networkAug = copy.deepcopy(network)
for biz in trainCopy:
  commonPos = np.nonzero(trainCopy[:,biz] == 1)
  commonNeg = np.nonzero(trainCopy[:,biz] == -1)
  for i in commonPos:
    for j in commonPos:
      networkAug[i, j] = 1

