import networkx as nx
import numpy as np

from env import *
from randDef import *
from randAtt import *
from msgDef import *
from randUav import *

import time

"""
the main loop where each player make actions
receive feedback
perform training
evaluate performance 
"""
def run(env, defender, attacker, uavs, numEpisode,gui):
	""" place to setup gui """
	if (gui):
		fig,ax = plt.subplots(figsize=(6,4))
		pos = nx.spring_layout(env.g)

	for eps in range(0, numEpisode):
		print("episode: %d" % eps)
		stateDict = env.resetGame()
		attacker.reset()

		if (gui):
			nodeColors = updateNodeColor(env.g)
			nx.draw_networkx_nodes(env.g, pos, node_color = nodeColors)
			nx.draw_networkx_edges(env.g, pos, edge_color = "black")
			plt.show(block=False)
			plt.pause(1)
			plt.clf()
				
		while(env.end == False):
			defAct = defender.act(stateDict["defState"], stateDict["defNode"])
			attAct = attacker.act(stateDict["attState"], stateDict["attNode"])

			uavActs = []
			for i in range(len(uavs)):
				uavAct = uavs[i].act(stateDict["uavState"],stateDict["uavNodes"][i])
				uavActs.append(uavAct)

			
			print ("t=%d def act: %s att act:%s uav acts:[%s]" % (env.t, str(defAct), str(attAct), str(uavActs)))
			
			stateDictAfter = env.step(defAct, attAct,uavActs)

			""" place to train if player need to train """
			defender.train(stateDict["defState"],
				           defAct,
				           stateDictAfter["defR"],
				           stateDictAfter["defState"],
				           env.end)
			stateDict = stateDictAfter

			""" place to draw the current environment """
			if (gui):
				nodeColors = updateNodeColor(env.g)
				nx.draw_networkx_nodes(env.g, pos, node_color = nodeColors)
				nx.draw_networkx_edges(env.g, pos, edge_color = "black")
				plt.show(block=False)
				plt.pause(1)
				plt.clf()

	if gui:
		plt.close()
	return 42

def updateNodeColor(g):
	values = []
	for i in g.nodes:
		if g.nodes[i]["isDef"] == 1 and g.nodes[i]["isAtt"] == 1:
			values.append("green")
		elif g.nodes[i]["isDef"] == 1:
			values.append("blue")
		elif g.nodes[i]["isAtt"] == 1:
			values.append("red")
		elif g.nodes[i]["numUav"] > 0:
			values.append("yellow")
		else:
			values.append("black")
	return values

def main():
	isGui = True
	numUav = 1
	env = Env(getDefaultGraph0,numUav)
	defender = RandDef()
	# defender = MsgDef(env.g)
	attacker = RandAtt() 
	uavs = [RandUav() for i in range(numUav)]

	run(env, defender, attacker, uavs, 20, isGui)



if __name__ == '__main__':
    main()



















