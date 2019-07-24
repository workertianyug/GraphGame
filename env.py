import networkx as nx
import numpy as np
import random

""" return a default graph for use """
""" [isDef, isAtt, numUav, reward, penetration time] """
def getDefaultGraph0():

	g0 = nx.DiGraph()
	g0.add_node(0, isDef=0, isAtt=0, numUav=0, r=1.0, d=1, ctr=0)
	g0.add_node(1, isDef=0, isAtt=0, numUav=0, r=0.0, d=0, ctr=0)
	g0.add_node(2, isDef=0, isAtt=0, numUav=0, r=2.0, d=2, ctr=0)
	g0.add_node(3, isDef=0, isAtt=0, numUav=0, r=1.0, d=2, ctr=0)

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

	g0.graph["utilDefC"] = 1.0
	g0.graph["utilAttC"] = -1.0

	return g0

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

	return g,pos

class Env(object):
	
	def __init__(self, gfn, numUav, numUav2):
		""" game information """
		self.gfn = gfn
		self.g, self.pos = gfn()
		self.end = False
		self.t = 0
		""" def state """
		self.defNode = -1
		self.defPos = None
		""" att state """
		self.attNode = -1
		self.attPos = None
		""" uav state """
		self.numUav = numUav
		self.uavNodes = [-1 for i in range(numUav)]
		self.uavPoss = [None for i in range(numUav)]
		""" uav2 state """
		self.numUav2 = numUav2
		self.uav2Poss = [None for i in range(numUav2)]

	
	def resetGame(self):
		""" game information """
		self.end = False
		self.g, self.pos = self.gfn()
		self.t = 0
		self.maxT = 100
		# initial locations will be randomized in the future
		""" def state """
		self.g.nodes[12]["isDef"] = 1 #TODO: this hardcode will be changed
		self.defNode = 12
		self.defPos = self.pos[12]
		""" att state """
		attNode = random.choice([0,1,2,3,4,5,10,15,20,9,14,19,24,21,22,23])
		self.g.nodes[attNode]["isAtt"] = 1
		self.attNode = attNode
		self.attPos = self.pos[attNode]
		""" uav state """
		self.g.nodes[0]["numUav"] = self.numUav
		self.uavNodes=[0 for i in range(self.numUav)]
		self.uavPoss = [self.pos[0] for i in range(self.numUav)]
		""" uav2 state """
		self.uav2Poss = [self.pos[12] for i in range(self.numUav2)]

		# TODO: add in uav2 state information
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
		self.defPos = self.pos[edge[1]]

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
				self.attPos = self.pos[edge[0]]
		else:
			""" reset the counter if the the attacker move away """
			self.g.nodes[0]["ctr"] = 0
			self.g.nodes[edge[0]]["isAtt"] = 0
			self.g.nodes[edge[1]]["isAtt"] = 1
			self.attNode = edge[1]
			self.attPos = self.pos[edge[1]]

		return 42

	""" uav move: along the edges """
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
	uav2 move: continuously above the graph 
	moves: (direction, speed)

	update position of each uav2 
	"""
	def _uav2Moves(self,moves):
		for i in range(0,len(moves)):
			(curDir, curVel) = moves[i]
			prevPos = self.uav2Poss[i]
			curPosx = prevPos[0] + curVel * np.sin(curDir)
			curPosy = prevPos[1] + curVel * np.cos(curDir)
			self.uav2Poss[i] = np.array([curPosx,curPosy])
			""" TODO: check whether moving out of boundary here """
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


	def _parseStateUav2(self):
		gUav = self.g.copy()
		for i in gUav.nodes:
			if gUav.nodes[i]["isAtt"] == 1 and gUav.nodes[i]["numUav"] == 0:
				gUav.nodes[i]["isAtt"] = 0
		""" should also know the position of every uav """
		gUav.graph["uav2Poss"] = self.uav2Poss
		return gUav

	def step(self, defAct, attAct, uavActs, uav2Acts):
		
		self._defMove(defAct)
		self._attMove(attAct)
		self._uavMoves(uavActs)
		self._uav2Moves(uav2Acts)

		self.t += 1

		defR = 0.0
		attR = 0.0

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

		""" TODO: check if uav find an attacker """

		stateDict = {"defState": self._parseStateDef(), "defR":defR,
					 "defNode": self.defNode,
		             "attState": self._parseStateAtt(), "attR":attR,
		             "attNode": self.attNode,
		             "end":self.end,
		             "uavState": self._parseStateUav(), "uavR":[0 for i in range(0,self.numUav)],
		             "uavNodes": self.uavNodes}
		return stateDict




# e = Env(getDefaultGraph0)













































