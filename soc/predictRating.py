# Completes the user-rating matrix using social network data

import cPickle
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand

h = open('user_user_network_500.data', 'r')
uuNetwork = cPickle.load(h)
h.close()
h = open('dataset_500.data', 'r')
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

alpha = .1 # Learning rate
train = rand.binomial(1, .7, ratings.shape) * ratings
trainCopy = train
# np.sum(train) / np.prod(train.shape) # Sparsity
for k in range(1000):  # Number of passes
	for i in range(train.shape[0]):
		train[i,:] += alpha * np.dot(network[i,:], train) # * (train[i,:] == 0)

np.sum((ratings - train) * (ratings - train) * np.abs(ratings - trainCopy)) / np.sum(np.abs(ratings - trainCopy))
