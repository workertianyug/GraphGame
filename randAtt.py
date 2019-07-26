import networkx as nx
import numpy as np
from env import *


class RandAtt(object):
	
	def __init__(self):

		self.seed = 17
		self.rand = np.random.RandomState(seed=self.seed)
		self.lastAction = None

	def reset(self):
		self.lastAction = None

	""" 
	take in attacker state
	output a random valid action 
	attState: the networkx graph representing attacker state
	attNode: attacker's current location
	"""
	def act(self, attState, attNode, eps=0):

		if (attState.nodes[attNode]["isAtt"] != 1):
			raise ValueError("att location doesn't match")

		""" if decides to attack, then finish the attack """
		if ((self.lastAction != None) 
			and (self.lastAction[0] == self.lastAction[1]) 
			and (attState.nodes[attNode]["r"]>0)):
		
			return self.lastAction

		validActions = list(attState.out_edges([attNode]))
		a = self.rand.randint(0, len(validActions),size=1)[0]

		self.lastAction = validActions[a]
		return validActions[a]


	def train(self,*arg):
		return 42




