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

h = open('../data/dataset_500_train_small.data', 'r')
train_list = cPickle.load(h)
h.close()

h = open('../data/dataset_500_test_small.data', 'r')
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

# out = open('networkTaste.data', 'wb')
# cPickle.dump(networkTaste, out)
# f.close()

# Training
def predict(G, maxIter):
  train_copy = copy.deepcopy(train)
  for k in range(maxIter):  # Number of passes
    for u in G.keys():
      friends = G[u]
      if len(friends) > 0:
        for b in range(len(bizs)):
          if train_copy[u,b] == 0:
            train[u,b] = np.mean(train[friends, b])
  # Sentiment prediction accuracy
  return np.sum((2 * (train > 0) - 1) * np.abs(test) != test) / float(np.sum(test != 0))

print predict(G, maxIter = 100)

# Redefine edges according to taste similarity
T = np.zeros((len(users), len(users)))
for b in range(len(bizs)) :
  samePos = np.nonzero(train[:,b] == 1)[0]
  sameNeg = np.nonzero(train[:,1] == -1)[0]
  if len(samePos) > 1:
    for i in range(len(samePos)):
      for j in range(i + 1, len(samePos)):
        T[i,j] += 100
        T[j,i] += 1
  if len(sameNeg) > 1:
    for i in range(len(sameNeg)):
      for j in range(i + 1, len(sameNeg)):
        T[i,j] += 1
        T[j,i] += 1

TG = dict(zip(userD.values(), [[] for u in users]))
for i in range(T.shape[0]):
  for j in range(T.shape[1]):
    if T[i,j] >= 5:
      TG[i].append(j)
      TG[j].append(i)

print predict(TG, maxIter = 100)

