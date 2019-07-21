import networkx as nx
import numpy as np
from env import *


class RandUav(object):
	
	def __init__(self):

		self.seed = 7
		self.rand = np.random.RandomState(seed=self.seed)


	""" 
	take in defender state
	output a random valid action 
	uavState: the networkx graph representing defender state
	uavNode: defender's current location
	"""
	def act(self, uavState, uavNode):

		if (uavState.nodes[uavNode]["numUav"] == 0):
			raise ValueError("uav location doesn't match")

		validActions = list(uavState.out_edges([uavNode]))
		a = self.rand.randint(0, len(validActions),size=1)[0]


		return validActions[a]



	def train(self,*arg):
		return 42

	def reset(self):
		return 42












































