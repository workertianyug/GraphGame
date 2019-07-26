import networkx as nx
import numpy as np
from env import *


class RandUav2(object):
	
	def __init__(self,uav2Range=1.0):

		# self.seed = 7
		# self.rand = np.random.RandomState(seed=self.seed)
		self.rand = np.random.RandomState()
		self.uav2Range = uav2Range

	""" 
	take in uav state
	output a random valid action 
	uavState: the networkx graph representing defender state
	return: move: (dir, vel)
	"""
	def act(self, uavState,eps=0):

		direc = self.rand.uniform(0.0,2*np.pi)
		veloc = 1.0

		return (direc, veloc)



	def train(self,*arg):
		return 42

	def reset(self):
		return 42












































