import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path_length
import numpy as np
import random
from env import *

"""
this is a short sighted greedy defender

only cares about its neighbour's reward and degree
"""

class ParamRandDef(object):

	def __init__(self):

		self.w1 = 1.0
		self.w2 = 1.0

	
	def act(self,defState, defNode, eps=0):

		# add in the chase part

		if (defState.nodes[defNode]["isDef"] != 1):
			raise ValueError("def location doesn't match")

		
		if (defState.graph["isFound"] == True):
			""" if position of the attacker is known: chase """
			
			attNode = defState.graph["attNode"]
			neighbors = list(defState.neighbors(defNode))

			bestNode = neighbors[0]
			minDist = 1000000000
			for n in neighbors:
				curDist = shortest_path_length(defState, n, attNode)
				if curDist < minDist:
					bestNode = n
					minDist = curDist

			return (defNode, bestNode)
		else:
			""" else do parameterized random walk """
			neighbors = list(defState.neighbors(defNode))

			neighborRewards = []
			neighborDegrees = []

			for n in neighbors:
				neighborRewards.append(defState.nodes[n]["r"])
				neighborDegrees.append(len(list(defState.neighbors(n))))

			neighborRewards = np.array(neighborRewards)
			neighborDegrees = np.array(neighborDegrees)

			comb = self.w1 * neighborRewards + self.w2 * neighborDegrees

			softmaxProb = np.exp(comb) / sum(np.exp(comb))

			nxtNode = np.random.choice(neighbors, p = softmaxProb)

			return (defNode, nxtNode)

	def train(self,*arg):
		return 42

	def reset(self):
		return 42




































