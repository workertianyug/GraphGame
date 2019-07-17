import networkx as nx
import numpy as np

from env import *
from randDef import *
from randAtt import *
from msgDef import *


"""
the main loop where each player make actions
receive feedback
perform training
evaluate performance 
"""
def run(env, defender, attacker, numEpisode):

	for eps in range(0, numEpisode):
		print("episode: %d" % eps)
		stateDict = env.resetGame()
		attacker.reset()
		while(env.end == False):
			defAct = defender.act(stateDict["defState"], stateDict["defNode"])
			attAct = attacker.act(stateDict["attState"], stateDict["attNode"])
			
			print ("t=%d def act: %s att act:%s" % (env.t, str(defAct), str(attAct)))
			stateDictAfter = env.step(defAct, attAct)

			""" place to train if player need to train """
			defender.train(stateDict["defState"],
				           defAct,
				           stateDictAfter["defR"],
				           stateDictAfter["defState"],
				           env.end)
			stateDict = stateDictAfter

	return 42


def main():

	env = Env(getDefaultGraph0)
	# defender = RandDef()
	defender = MsgDef(env.g)
	attacker = RandAtt() 

	run(env, defender, attacker, 20)



if __name__ == '__main__':
    main()



















