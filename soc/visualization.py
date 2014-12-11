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
train = cPickle.load(h)
h.close()

h = open('../data/dataset_500_test.data', 'r')
test = cPickle.load(h)
h.close()

users = list(set([line[0] for line in train]))
bizs = list(set([line[1] for line in train]))
userD = dict(zip(users, range(len(users))))
bizD = dict(zip(bizs, range(len(bizs))))

G = dict()
for u in users:
  G[userD[u]] = [userD[v] for v in users if v in network[u]]

def visualize(G, fileName):
  # Visualizes graph G using NetworkX and Gephi
  V = G.keys()
  E = edge_set(G)
  # Construct networkX graph
  g = nx.Graph()
  for v in V:
    g.add_node
  for e in E:
    g.add_edge(e[0], e[1])
  # plt.figure()
  nx.write_graphml(g, fileName + '.graphml')  # Save to Gephi file
  # nx.draw(g, node_size = 700, node_shape = 'o', width = 3.0, linewidths = 2.0)
  # plt.show()
  print 'Gephi file generated!'

def edge_set(G):
  # Returns the edge-set of a graph stored in an adjacency list
  return [(v,u) for v in G.keys() for u in G[v]]

visualize(G, 'user-user-graph')

