


import networkx as nx
import numpy as np

import random


def getDefaultGraph5x5():
	g = nx.DiGraph()

	for i in range(0,25):
		if i == 12:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=3.0, d=3, ctr=0)
		elif i in [6,7,8,11,13,16,17,18]:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=1.0, d=2, ctr=0)
		else:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=0.0, d=0, ctr=0)

		g.add_edge(i,i)

	for i in range(0,24):
		if i in [4,9,14,19,24]:
			continue
		g.add_edge(i,i+1)
		g.add_edge(i+1,i)

	for i in range(0,4):
		for j in range(0,5):
			g.add_edge(i*5 + j, i*5+5 + j)
			g.add_edge(i*5+5 + j, i*5 + j)			

	g.graph["utilDefC"] = 2.0
	g.graph["utilAttC"] = -2.0

	pos = dict()
	pos[0] = np.array([0.0,4.0])
	pos[1] = np.array([1.0,4.0])
	pos[2] = np.array([2.0,4.0])
	pos[3] = np.array([3.0,4.0])
	pos[4] = np.array([4.0,4.0])
	pos[5] = np.array([0.0,3.0])
	pos[6] = np.array([1.0,3.0])
	pos[7] = np.array([2.0,3.0])
	pos[8] = np.array([3.0,3.0])
	pos[9] = np.array([4.0,3.0])
	pos[10] = np.array([0.0,2.0])
	pos[11] = np.array([1.0,2.0])
	pos[12] = np.array([2.0,2.0])
	pos[13] = np.array([3.0,2.0])
	pos[14] = np.array([4.0,2.0])
	pos[15] = np.array([0.0,1.0])
	pos[16] = np.array([1.0,1.0])
	pos[17] = np.array([2.0,1.0])
	pos[18] = np.array([3.0,1.0])
	pos[19] = np.array([4.0,1.0])
	pos[20] = np.array([0.0,0.0])
	pos[21] = np.array([1.0,0.0])
	pos[22] = np.array([2.0,0.0])
	pos[23] = np.array([3.0,0.0])
	pos[24] = np.array([4.0,0.0])

	for i in range(0,25):
		g.node[i]["x"] = pos[i][0]
		g.node[i]["y"] = pos[i][1]
	return g,pos




class RandAtt():

	def __init__(self, gtemp):

		self.SEED = 3

		self.gtemp = gtemp

		self.lastActionEdge = None
		self.lastAction = None
		self.lastEdgeQVec = None



	def reset(self):
		self.lastActionEdge = None
		self.lastAction = None
		self.lastEdgeQVec = None

	"""
	from template graph to example target graph to create placeholder
	input: networkx
	output: networkx
	"""
	def action(self, attState, attNode):

		if (attState.nodes[attNode]["isAtt"] != 1):
			raise ValueError("att location doesn't match")

		""" if decides to attack, then finish the attack """
		if ((self.lastActionEdge != None) 
			and (self.lastActionEdge[0] == self.lastActionEdge[1]) 
			and (attState.nodes[attNode]["r"]>0)):
		
			return self.lastAction, self.lastActionEdge, self.lastEdgeQVec

		""" construct random graph of q values """
		ttmp = attState.copy()
		
		for nodeIndex, nodeFeature in attState.nodes(data=True):
			ttmp.add_node(nodeIndex, features=[])
			
		for nodeIndex, nodeFeature in ttmp.nodes(data=True):
			for attr in list(nodeFeature):
				if attr != "features":
					del nodeFeature[attr]

		for u, v, features in attState.edges(data=True):
			q_val = np.random.uniform(-1,1,1)[0]
			ttmp.add_edge(u, v, features=[q_val]) # this is q value


		ttmp.graph["features"] = [0.0]

		""" now select valid action """
		validActions = list(attState.out_edges([attNode]))

		qdict = dict()

		for e in validActions:
			qdict[e] = ttmp.get_edge_data(*e)["features"][0]

		maxEdge = max(qdict, key=qdict.get)


		""" from ttmp, compute the vector containing all values """
		edgeQvec = []
		N_edges = len(ttmp.edges)
		for e in ttmp.edges:
			edgeQvec.append(ttmp.get_edge_data(*e)["features"][0])

		edgeQvec = np.reshape(edgeQvec,[1,N_edges])

		self.lastActionEdge = maxEdge
		self.lastAction = ttmp
		self.lastEdgeQVec = edgeQvec
		
		return ttmp, maxEdge, edgeQvec











































