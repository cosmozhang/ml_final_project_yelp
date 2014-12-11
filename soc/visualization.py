# Network visualization

import cPickle
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
import copy
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from numpy import random as rand
import copy
import cPickle
import networkx as nx
from collections import Counter

h = open('../data/user_user_network_500.data', 'r')
network = cPickle.load(h)
h.close()

h = open('../data/dataset_500_train.data', 'r')
train_list = cPickle.load(h)
h.close()

h = open('../data/dataset_500_test.data', 'r')
test_list = cPickle.load(h)
h.close()

users = list(set([line[0] for line in train_list + test_list]))
bizs = list(set([line[1] for line in train_list + test_list]))
userD = dict(zip(users, range(len(users))))
bizD = dict(zip(bizs, range(len(bizs))))

G = dict()
for u in users:
  G[userD[u]] = [userD[v] for v in users if v in network[u]]

def edge_set(G):
  # Returns the edge-set of a graph stored in an adjacency list
  return [(v,u) for v in G.keys() for u in G[v]]

# def visualize(G, fileName):
#   # Visualizes graph G using NetworkX and Gephi
#   V = G.keys()
#   E = edge_set(G)
#   # Construct networkX graph
#   g = nx.Graph()
#   for v in V:
#     g.add_node
#   for e in E:
#     g.add_edge(e[0], e[1])
#   # plt.figure()
#   nx.write_graphml(g, fileName + '.graphml')  # Save to Gephi file
#   # nx.draw(g, node_size = 700, node_shape = 'o', width = 3.0, linewidths = 2.0)
#   # plt.show()
#   print 'Gephi file generated!'

# visualize(G, 'user-user-graph')

# Rating matrix
train = np.zeros((len(users), len(bizs)))
for line in train_list:
	if line[2] == '+':
		train[userD[line[0]], bizD[line[1]]] = 1
	else:
		train[userD[line[0]], bizD[line[1]]] = -1

test = np.zeros((len(users), len(bizs)))
for line in test_list:
  if line[2] == '+':
    test[userD[line[0]], bizD[line[1]]] = 1
  else:
    test[userD[line[0]], bizD[line[1]]] = -1
# np.sum(ratings) / np.prod(ratings.shape) # Sparsity

# # Redefine edges according to taste similarity
# networkTaste = np.zeros((len(usrs), len(usrs)), dtype = 'int')
# for i in range(len(usrs)):
# 	for j in range(len(usrs)):
# 		r1 = ratings[i,:]
# 		r2 = ratings[j,:]
# 		num = min(np.sum(r1 != 0), np.sum(r2 != 0)) # Number of common businesses
# 		if num > 0 and np.sum(r1 * r2) / float(num) > 0.50: # Threshold value
# 			networkTaste[i,j] = 1

# out = open('networkTaste.data', 'wb')
# cPickle.dump(networkTaste, out)
# f.close()

# Training
train_copy = copy.deepcopy(train)

for k in range(100):  # Number of passes
  for u in G.keys():
    friends = G[u]
    if len(friends) > 0:
      for b in range(len(bizs)):
        if train_copy[u,b] == 0:
          train[u,b] = np.mean(train[friends, b])

# Evaluation
np.sum((train * np.abs(test) > 0) == test) / float(np.sum(test != 0))
