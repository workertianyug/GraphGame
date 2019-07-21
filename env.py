import networkx as nx
import numpy as np


""" return a default graph for use """
""" [isDef, isAtt, numUav, reward, penetration time] """
def getDefaultGraph0():

	g0 = nx.DiGraph()
	g0.add_node(0, isDef=0, isAtt=0, numUav=0, r=1, d=1, ctr=0)
	g0.add_node(1, isDef=0, isAtt=0, numUav=0, r=0, d=0, ctr=0)
	g0.add_node(2, isDef=0, isAtt=0, numUav=0, r=2, d=2, ctr=0)
	g0.add_node(3, isDef=0, isAtt=0, numUav=0, r=1, d=2, ctr=0)

	g0.add_edge(0,0)
	g0.add_edge(1,1)
	g0.add_edge(2,2)
	g0.add_edge(3,3)
	
	g0.add_edge(0,1)
	g0.add_edge(1,0)
	g0.add_edge(1,2)
	g0.add_edge(2,1)
	g0.add_edge(2,3)
	g0.add_edge(3,2)
	g0.add_edge(0,3)
	g0.add_edge(3,0)
	g0.add_edge(0,2)
	g0.add_edge(2,0)

	g0.graph["utilDefC"] = 1
	g0.graph["utilAttC"] = -1

	return g0



class Env(object):
	
	def __init__(self, gfn, numUav):
		self.gfn = gfn
		self.g = gfn()
		self.end = False
		self.t = 0
		self.defNode = -1
		self.attNode = -1
		self.numUav = numUav
		self.uavNodes = [-1 for i in range(numUav)]
		

	
	def resetGame(self):
		self.end = False
		self.g = self.gfn()
		self.t = 0
		self.maxT = 10
		# initial locations will be randomized in the future
		self.g.nodes[0]["isDef"] = 1
		self.defNode = 0
		self.g.nodes[2]["isAtt"] = 1
		self.attNode = 2
		self.g.nodes[0]["numUav"] = self.numUav
		self.uavNodes=[0 for i in range(self.numUav)]

		stateDict = {"defState": self._parseStateDef(), "defR":0,
					 "defNode": self.defNode,
		             "attState": self._parseStateAtt(), "attR":0,
		             "attNode": self.attNode,
		             "end":self.end,
		             "uavState":self._parseStateUav(), "uavRs":[0 for i in range(self.numUav)],
		             "uavNodes":self.uavNodes}

		return stateDict

	""" defender move: move along edges """
	def _defMove(self, edge):
		if (not self.g.has_edge(*edge)):
			raise ValueError('edge move has to exist in graph')
		if (self.g.nodes[edge[0]]["isDef"] == 0):
			raise ValueError('move has to start from current location')

		""" make the move """
		self.g.nodes[edge[0]]["isDef"] = 0
		self.g.nodes[edge[1]]["isDef"] = 1
		self.defNode = edge[1]

		return 42

	""" attacker move: move along edges / stay and attack """
	def _attMove(self, edge):
		if (not self.g.has_edge(*edge)):
			raise ValueError('edge move has to exist in graph')
		if (self.g.nodes[edge[0]]["isAtt"] == 0):
			raise ValueError('move has to start from current location')
			
		if (edge[0] == edge[1]):
			""" the attacker chooses to attack, increase the counter """
			if (self.g.nodes[edge[0]]["ctr"] < self.g.nodes[edge[0]]["d"]):
				self.g.nodes[edge[0]]["ctr"] += 1
		else:
			""" reset the counter if the the attacker move away """
			self.g.nodes[0]["ctr"] = 0
			self.g.nodes[edge[0]]["isAtt"] = 0
			self.g.nodes[edge[1]]["isAtt"] = 1
			self.attNode = edge[1]

		return 42

	def _uavMoves(self, edges):

		for i in range(0,len(edges)):
			edge = edges[i]
			if (not self.g.has_edge(*edge)):
				raise ValueError('edge move has to exist in graph')
			if (self.g.nodes[edge[0]]["numUav"] == 0):
				raise ValueError("move has to start from current location")

			self.g.nodes[edge[0]]["numUav"] = self.g.nodes[edge[0]]["numUav"] - 1
			self.g.nodes[edge[1]]["numUav"] = self.g.nodes[edge[1]]["numUav"] + 1

			self.uavNodes[i] = edge[1]

		return 42

	""" 
	return the state that defender can see 
	isDef
	isAtt: only when UAV has found the attacker (how to discount this?)
	numUav:
	r
	d
	"""
	def _parseStateDef(self):

		gDef = self.g.copy()
		for i in gDef.nodes:
			if gDef.nodes[i]["isAtt"] == 1 and gDef.nodes[i]["numUav"] == 0:
				gDef.nodes[i]["isAtt"] = 0
		return gDef

	""" 
	return the state that attacker can see 
	isAtt:
	numUav: only when the attacker has encountered the UAV
	r
	d
	ctr
	"""
	def _parseStateAtt(self):

		gAtt = self.g.copy()
		for i in gAtt.nodes:
			del gAtt.nodes[i]["isDef"]
			if gAtt.nodes[i]["numUav"] == 1 and gAtt.nodes[i]["isAtt"] == 0:
				gAtt.nodes[i]["numUav"] = 0
		return gAtt

	def _parseStateUav(self):

		gUav = self.g.copy()
		for i in gUav.nodes:
			if gUav.nodes[i]["isAtt"] == 1 and gUav.nodes[i]["numUav"] == 0:
				gUav.nodes[i]["isAtt"] = 0
		return gUav


	def step(self, defAct, attAct, uavActs):
		
		self._defMove(defAct)
		self._attMove(attAct)
		self._uavMoves(uavActs)

		self.t += 1

		defR = 0
		attR = 0

		if (self.t == self.maxT):
			self.end = True

		""" check if def att in the same cell """
		if self.defNode == self.attNode:
			self.end = True
			defR = self.g.graph["utilDefC"]
			attR = self.g.graph["utilAttC"]

		""" check if attacker done attacking """
		if ((self.g.nodes[self.attNode]["ctr"] 
			== self.g.nodes[self.attNode]["d"]) and
			 (self.g.nodes[self.attNode]["d"]>0)):
			 self.end = True
			 defR = - self.g.nodes[self.attNode]["r"]
			 attR = self.g.nodes[self.attNode]["r"]

		stateDict = {"defState": self._parseStateDef(), "defR":defR,
					 "defNode": self.defNode,
		             "attState": self._parseStateAtt(), "attR":attR,
		             "attNode": self.attNode,
		             "end":self.end,
		             "uavState": self._parseStateUav(), "uavR":[0 for i in range(0,self.numUav)],
		             "uavNodes": self.uavNodes}
		return stateDict




# e = Env(getDefaultGraph0)













































