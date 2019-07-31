import networkx as nx
import numpy as np

from env import *
from randDef import *
from randAtt import *
from msgDef import *
from randUav import *
from randUav2 import *
from ddpgUav2 import *
import time

import pickle

"""
the main loop where each player make actions
receive feedback
perform training
evaluate performance 
"""
def run(env, defender, attacker, uavs, uav2s, numEpisode,gui):
	""" place to setup gui """
	if (gui):
		fig,ax = plt.subplots(figsize=(6,4))
		# pos = nx.spring_layout(env.g, iterations=100)
		pos = env.pos

	""" initialize reward tracking """
	cumDefR = 0.0
	cumAttR = 0.0
	catch = 0
	avgDefUtilList = []

	for eps in range(0, numEpisode):
		print("episode: %d" % eps)
		stateDict = env.resetGame()
		attacker.reset()

		if (gui):
			nodeColors = updateNodeColor(env.g)
			nx.draw_networkx_nodes(env.g, pos, node_color = nodeColors)
			nx.draw_networkx_edges(env.g, pos, edge_color = "black")
			plt.axis("scaled")
			plt.show(block=False)
			plt.pause(1)
			plt.clf()
				
		while(env.end == False):
			""" defender make move """
			defAct = defender.act(stateDict["defState"], stateDict["defNode"],eps)
			""" attacker make move """
			attAct = attacker.act(stateDict["attState"], stateDict["attNode"])
			""" uavs make move together """
			uavActs = []
			for i in range(len(uavs)):
				uavAct = uavs[i].act(stateDict["uavState"],stateDict["uavNodes"][i])
				uavActs.append(uavAct)

			""" TODO: uav2s make move together """
			uav2Acts = []
			for i in range(len(uav2s)):
				uav2Act = uav2s[i].act(stateDict["uav2State"])
				uav2Acts.append(uav2Act)
			
			print ("t=%d def act: %s att act:%s uav acts:[%s]" 
				% (env.t, str(defAct), str(attAct), str(uav2Acts)))
			
			stateDictAfter = env.step(defAct, attAct,uavActs,uav2Acts)

			""" place to train if player need to train """
			defender.train(stateDict["defState"],
				           defAct,
				           stateDictAfter["defR"],
				           stateDictAfter["defState"],
				           env.end,eps)
			stateDict = stateDictAfter

			for i in range(0,len(uav2s)):
				uav2s[i].train(stateDict["uav2State"],
							   uav2Acts[i],
							   0,
							   stateDictAfter["uav2State"],
							   env.end, eps)

			""" place to draw the current environment """
			if (gui):
				for i in env.uav2Poss:
					cir2 = plt.Circle((i[0], i[1]), 0.25, color='yellow')
					plt.gcf().gca().add_artist(cir2)
					
				nodeColors = updateNodeColor(env.g)
				nx.draw_networkx_nodes(env.g, pos, node_color = nodeColors)
				nx.draw_networkx_edges(env.g, pos, edge_color = "black")
				nx.draw_networkx_labels(env.g, pos)
				plt.axis("scaled")
				plt.show(block=False)
				plt.pause(1)
				plt.clf()

			""" place to increase cumulative defender and attacker utility """
			# print (stateDict["defR"], stateDict["attR"])
			cumDefR += stateDict["defR"]
			cumAttR += stateDict["attR"]
			if stateDict["defR"] > 0:
				catch += 1

		avgDefUtilList.append(cumDefR/(eps+1.0))


	if gui:
		plt.close()

	""" save avg def utility for learning curve plot """
	with open("data/avgDefUtilEps.pkl","w") as f:
		pickle.dump([avgDefUtilList, numEpisode], f)

	""" compute average defender and attacker utility """
	avgDefR = cumDefR / numEpisode
	avgAttR = cumAttR / numEpisode 

	print(cumDefR,cumAttR,catch)

	print ("numEpisode=%d avgDefR=%s avgAttR=%s" 
		% (numEpisode, str(avgDefR), str(avgAttR)))

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
			values.append("gray")
	return values

def main():
	isGui = True
	numUav = 0
	numUav2 = 2
	env = Env(getDefaultGraph5x5,numUav,numUav2)
	defender = RandDef()
	# defender = MsgDef(env.g)
	attacker = RandAtt() 
	uavs = [RandUav() for i in range(numUav)]
	# uav2s = [RandUav2() for i in range(numUav2)]
	uav2s = [DdpgUav2(i) for i in range(numUav2)]
	run(env, defender, attacker, uavs, uav2s, 1000, isGui)



if __name__ == '__main__':
    main()



# rand def seed=3, rand att seed=17
# (-107.0, 107.0, 329)
# numEpisode=1000 avgDefR=-0.107 avgAttR=0.107

# with training rand att seed=17
# (-216.0, 216.0, 288)
# numEpisode=1000 avgDefR=-0.216 avgAttR=0.216


# rand def seed=3 rand att seed=17
# (-265.0, 265.0, 1742)
# numEpisode=5000 avgDefR=-0.053 avgAttR=0.053


# msg def rand att seed=17
# (-102.0, 102.0, 1713)
# numEpisode=5000 avgDefR=-0.0204 avgAttR=0.0204




