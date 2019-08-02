import networkx as nx
import numpy as np
from env import *


class RandDef(object):
	
	def __init__(self):

		self.seed = 3
		self.rand = np.random.RandomState(seed=self.seed)


	""" 
	take in defender state
	output a random valid action 
	defState: the networkx graph representing defender state
	defNode: defender's current location
	"""
	def act(self, defState, defNode, eps=0):

		if (defState.nodes[defNode]["isDef"] != 1):
			raise ValueError("def location doesn't match")

		validActions = list(defState.out_edges([defNode]))
		a = self.rand.randint(0, len(validActions),size=1)[0]


		return validActions[a]



	def train(self,*arg):
		return 42

	def reset(self):
		return 42












































