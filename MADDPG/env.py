import networkx as nx
import numpy as np
import random


"""
To create a valid graph:
node feature:
	isDef: whether defender is on this node
	isAtt: whether attacker is on this node
	numUav: deprecated
	r: the reward of this node
	d: penetration time of this node
	ctr: local time tracker for completing an attack
	x: x coordinate of this node
	y: y coordinate of this node
edge feature:
	None
graph feature:
	utilDefC: utility for defender if he catches the attacker
	utilAttC: utility for attacker if she is caught by the defender
"""
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

def getGridGraphNxN(n):
	g = nx.DiGraph()
	random.seed(10)

	low_reward_nodes = np.random.randint(0, n*n - 1, size=(1, random.randint(0, n*n-1)))
	high_reward_nodes = np.random.randint(0, n*n - 1, size=(1, random.randint(0, n*n-1)))
	
	for i in range(0, n*n):
		if i in high_reward_nodes:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=3.0, d=3, ctr=0)
		elif i in low_reward_nodes and i not in high_reward_nodes:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=1.0, d=2, ctr=0)
		else:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=0.0, d=0, ctr=0)

		g.add_edge(i, i)
		if i % (n - 1) != 0:
			g.add_edge(i, i + 1)
		if i < (n * n - n):
			g.add_edge(i, i + n)
		if i >= n:
			g.add_edge(i, i - n)

	g.graph["utilDefC"] = 2.0
	g.graph["utilAttC"] = -2.0

	pos = dict()
	# for each possible n*n position,
	# assign each position to [0.0, 4.0], then [1.0, 4.0], and so on...
	idx = 0
	for i in range(n, -1, -1):
		for j in range(0, n):
			pos[idx] = np.array([float(j), float(i)])

	for i in range(0, n*n):
		g.node[i]["x"] = pos[i][0]
		g.node[i]["y"] = pos[i][1]
	return g,pos


# not fully implemented -- unsure of what to do about action space 
def getDefaultGraph10x10():
	g = nx.DiGraph()
	random.seed(10)

	num_zero_reward_nodes = random.randint(0,99)
	num_low_reward_nodes = random.randint(0,99)
	num_high_reward_nodes = random.randint(0,99)
	num_additional_edges = random.randint(0,49)

	for i in range(0,100):
		if i in np.random.randint(0, 99, size=(1,num_high_reward_nodes)):
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=3.0, d=3, ctr=0)
		# also need to test that node isn't already in graph
		elif i in np.random.randint(0, 99, size=(1,num_low_reward_nodes)):
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=1.0, d=2, ctr=0)
		else:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=0.0, d=0, ctr=0)

		g.add_edge(i,i)

	for i in range(0, num_additional_edges):
		g.add_edge(i, random.randint(0,99))

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

	for i in range(0,100):
		g.node[i]["x"] = pos[i][0]
		g.node[i]["y"] = pos[i][1]
	return g,pos

"""
The simulation environment
"""
class Env(object):
	
	def __init__(self, gfn, numUav, numUav2, uav2Range=1.0):

		""" game information """
		self.gfn = gfn
		self.g, self.pos = gfn()
		self.end = False
		self.t = 0
		self.minX, self.maxX, self.minY, self.maxY = self._getBoundary(self.pos)

		""" def state """
		self.defNode = -1
		self.defPos = None

		""" att state """
		self.attNode = -1
		self.attPos = None
		# whether attacker is found in this episode
		self.isFound = False 

		""" uav state """
		self.numUav = numUav
		self.uavNodes = [-1 for i in range(numUav)]
		self.uavPoss = [None for i in range(numUav)]

		""" uav2 state """
		self.numUav2 = numUav2
		self.uav2Poss = [None for i in range(numUav2)]
		self.uav2Range = uav2Range
		#whether each uav has found attacker
		self.uav2Found = [False for i in range(numUav2)] 

	"""
	reset the game state after each episode
	"""
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
		self.isFound = False
		""" uav state """
		self.g.nodes[0]["numUav"] = self.numUav
		self.uavNodes=[0 for i in range(self.numUav)]
		self.uavPoss = [self.pos[0] for i in range(self.numUav)]
		""" uav2 state """
		self.uav2Poss = [self.pos[12] for i in range(self.numUav2)]
		self.uav2Found = [False for i in range(self.numUav2)]

		# TODO: add in uav2 state information
		stateDict = {"defState": self._parseStateDef(), "defR":0,
					 "defNode": self.defNode,

		             "attState": self._parseStateAtt(), "attR":0,
		             "attNode": self.attNode,

		             "end":self.end,

		             "uavState":self._parseStateUav(), 
		             "uavRs":[0 for i in range(self.numUav)],
		             "uavNodes":self.uavNodes,

		             "uav2State": self._parseStateUav2(),
		             "uav2R":[0 for i in range(0,self.numUav2)],
		             "uav2Poss":self.uav2Poss,
		             "uav2Found":self.uav2Found}

		return stateDict

	"""
	take in position of every node on the graph
	output the rectangle boundary of the graph
	uav2 cannot move across these boundaries
	"""
	def _getBoundary(self, pos):
		minX, maxX = pos[0][0], pos[0][0]
		minY, maxY = pos[0][1], pos[0][1]
		for i in pos:
			if pos[i][0] < minX:
				minX = pos[i][0]
			if pos[i][0] > maxX:
				maxX = pos[i][0]
			if pos[i][1] < minY:
				minY = pos[i][1]
			if pos[i][1] > maxY:
				maxY = pos[i][1]
		return minX-0.5,maxX+0.5,minY-0.5,maxY+0.5


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
			if self.uav2Found[i] == True:
				self.uav2Poss[i] = self.attPos
				continue

			(curDir, curVel) = moves[i]
			prevPos = self.uav2Poss[i]
			curPosx = prevPos[0] + curVel * np.sin(curDir)
			curPosy = prevPos[1] + curVel * np.cos(curDir)
			if (curPosx < self.minX 
				or curPosx > self.maxX 
				or curPosy < self.minY 
				or curPosy > self.maxY):
				continue

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
	x
	y
	"""
	def _parseStateDef(self):

		gDef = self.g.copy()
		if not self.isFound:
			gDef.nodes[self.attNode]["isAtt"] = 0
		return gDef

	""" 
	return the state that attacker can see 
	isAtt:
	numUav: only when the attacker has encountered the UAV
	r
	d
	ctr
	x
	y
	"""
	#TODO: update isFound
	def _parseStateAtt(self):

		gAtt = self.g.copy()
		for i in gAtt.nodes:
			del gAtt.nodes[i]["isDef"]
			if gAtt.nodes[i]["numUav"] == 1 and gAtt.nodes[i]["isAtt"] == 0:
				gAtt.nodes[i]["numUav"] = 0
		return gAtt

	"""
	deprecated: uav will not be used
	"""
	def _parseStateUav(self):

		gUav = self.g.copy()
		for i in gUav.nodes:
			if gUav.nodes[i]["isAtt"] == 1 and gUav.nodes[i]["numUav"] == 0:
				gUav.nodes[i]["isAtt"] = 0
		return gUav

	"""
	return the state that uav2s can see
	node:
		isDef
		isAtt: only when attacker is found
		r
		d
		x
		y
	uav2s states:
		uav2Poss
		uav2Found
	"""
	def _parseStateUav2(self):

		gUav2 = self.g.copy()
		""" no information abut ctr """
		for i in gUav2.nodes:
			del gUav2.nodes[i]["ctr"]
		if not self.isFound:
			gUav2.nodes[self.attNode]["isAtt"] = 0
		""" should also know the position of every uav """
		gUav2.graph["uav2Poss"] = self.uav2Poss
		gUav2.graph["uav2Found"] = self.uav2Found
		gUav2.graph["defPos"] = self.defPos
		if self.isFound:
			gUav2.graph["attPos"] = self.attPos
		else:
			gUav2.graph["attPos"] = np.array([0.0,0.0])
		return gUav2

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

		""" check if uav2 find an attacker """
		for i in range(0,len(self.uav2Poss)):
			curUav2Pos = self.uav2Poss[i] 
			if np.linalg.norm(curUav2Pos - self.attPos) < self.uav2Range:
				self.isFound = True
				self.uav2Found[i] = True

		stateDict = {"defState": self._parseStateDef(), "defR":defR,
					 "defNode": self.defNode,

		             "attState": self._parseStateAtt(), "attR":attR,
		             "attNode": self.attNode,

		             "end":self.end,

		             "uavState": self._parseStateUav(), 
		             "uavR":[0 for i in range(0,self.numUav)],
		             "uavNodes": self.uavNodes,
		             
		             "uav2State": self._parseStateUav2(),
		             "uav2R":[0 for i in range(0,self.numUav2)],
		             "uav2Poss":self.uav2Poss,
		             "uav2Found":self.uav2Found}
		return stateDict







































