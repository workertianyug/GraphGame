import networkx as nx
import numpy as np

from env import *
from maddpgDef import *
from maddpgUav2 import *
from randAtt import *

from replay_buffer import ReplayBuffer

import time
import pickle


TAU = 0.01

def create_init_update(oneline_name, target_name, tau=0.99):
	online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
	target_var = [i for i in tf.trainable_variables() if target_name in i.name]

	target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
	target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

	return target_init, target_update


def _create_feature( attr, fields):
	return np.hstack([np.array(attr[field], dtype=float) for field in fields])


"""
from template graph to example input graph to create placeholder 
input: networkx
output: networkx
"""
def _gtmp2intmp(gtemp):
	intmp = gtemp.copy()
	inNodeFields = ["isDef","isAtt","numUav","r","d","x","y"]

	for nodeIndex, nodeFeature in gtemp.nodes(data=True):
		intmp.add_node(nodeIndex, 
					   features=_create_feature(nodeFeature, inNodeFields))
		
	for nodeIndex, nodeFeature in intmp.nodes(data=True):
		for attr in list(nodeFeature):
			if attr != "features":
				del nodeFeature[attr]


	for u, v, features in gtemp.edges(data=True):
		intmp.add_edge(u, v, features=[1.0]) #TODO: change this to []

	intmp.graph["features"] = [0.0]

	return intmp




def defenderMove(defender, stateDict, sess):

	defState = stateDict["defState"]
	defNode = stateDict["defNode"]

	gin = _gtmp2intmp(defState)

	state = utils_np.networkxs_to_graphs_tuple([gin])

	actDict = defender.action(state,sess)

	graphTupleOut,allEdgeQ = actDict["graph"], actDict["edge"]

	outg = utils_np.graphs_tuple_to_networkxs(graphTupleOut[-1])[0]

	outg = nx.DiGraph(outg)

	validActions = list(defState.out_edges([defNode]))
	qdict = dict()

	for e in validActions:
		qdict[e] = outg.get_edge_data(*e)["features"][0]


	cur_act = max(qdict, key=qdict.get)

	return outg, allEdgeQ, cur_act


"""
use gumbel trick to sample discrete actions and give its continuous representation

cur_act: (u,v) this is what env would read as action

allEdgeQ: this is the vector representation of the cont action after gumbel trick
"""
def defenderMove2(defender, stateDict, sess):
	defState = stateDict["defState"]
	defNode = stateDict["defNode"]

	gin = _gtmp2intmp(defState)

	state = utils_np.networkxs_to_graphs_tuple([gin])

	actDict = defender.action(state,sess)

	graphTupleOut,_ = actDict["graph"], actDict["edge"]

	outg = utils_np.graphs_tuple_to_networkxs(graphTupleOut[-1])[0]

	outg = nx.DiGraph(outg)

	# make the discrete action

	validActions = list(defState.out_edges([defNode]))
	qdict = dict()

	for e in validActions:
		qdict[e] = outg.get_edge_data(*e)["features"][0]

	cur_act = max(qdict, key=qdict.get)

	# generate continuous representation of the discrete action
	# set edges of invalid actions to negative infinity, take soft max
	for e in outg.edges:
		if e[0] != defNode:
			outg.add_edge(*e, features=[-100.0])

	# extract "q values" to a vector
	allEdgeQ = []
	for e in outg.edges:
		allEdgeQ.append(outg.get_edge_data(*e)["features"][0])

	allEdgeQ = np.array(allEdgeQ)

	# first take softmax to generate valid logits
	# warning: sometimes this would cause overflow 
	allEdgeQ = np.exp(allEdgeQ) / sum(np.exp(allEdgeQ))

	# take log probabilities
	allEdgeQ = np.log(allEdgeQ)

	# add gumbel samples
	gumbelSample = np.random.gumbel(size=len(allEdgeQ))
	allEdgeQ = allEdgeQ + gumbelSample

	allEdgeQ = allEdgeQ / TAU

	# take softmax
	allEdgeQ = np.exp(allEdgeQ) / sum(np.exp(allEdgeQ))

	allEdgeQ = np.reshape(allEdgeQ,[1,len(allEdgeQ)])

	return outg, allEdgeQ, cur_act








def stateDictToUav2State(stateDict):
	myPos = stateDict["uav2State"].graph["uav2Poss"][0]
	defPos = stateDict["uav2State"].graph["defPos"]
	attPos = stateDict["uav2State"].graph["attPos"]

	state = np.concatenate([myPos,defPos,attPos])

	state = np.reshape(state,[1,6])

	return state



def uav2Move(uav2, stateDict, sess):

	myPos = stateDict["uav2State"].graph["uav2Poss"][0]
	defPos = stateDict["uav2State"].graph["defPos"]
	attPos = stateDict["uav2State"].graph["attPos"]

	feature = np.concatenate([myPos,defPos,attPos])

	feature = np.reshape(feature,[1,6])

	uav2Act = uav2.action(feature,sess)

	return uav2Act["direc"]

def attackerMove(attacker, stateDict):

	cur_q_graph,edge,edgeQvec = attacker.action(stateDict["attState"],stateDict["attNode"])

	return cur_q_graph, edge, edgeQvec


"""
defender: defender object
oldState; graphTuple object
defActEdge: [1,N_edge] matrix 
attActEdge: [1,N_edge] matrix
uav2Act: [1,2] matrix
newState: graphTuple object
termnial: bool
eps: current episode number

note: currently does not support batch training
can only sample 1 at a time
"""
def trainDefender(defender, defender_target, D, oldState, defActEdge, attActEdge, 
uav2Act, reward, newState, terminal, eps, 
sess, actorTargetUpOp, criticTargetUpOp):


	D.append((oldState,defActEdge,attActEdge,uav2Act,reward,newState,terminal))

	if (eps < 5):
		return 42

	i = random.randint(0,len(D)-2)
	b_cur = D[i]

	s0 = b_cur[0]
	defActEdge = b_cur[1]
	attActEdge = b_cur[2]
	uav2Act = b_cur[3]
	r = b_cur[4]
	s1 = b_cur[5]
	term = b_cur[6]

	if term == False:
		b_nxt = D[i+1]
		target = r + 0.9 * defender_target.Q(b_nxt[0],b_nxt[1], b_nxt[2], b_nxt[3],sess)[0][0]
	else:
		target = r

	defender.train_actor(s0, attActEdge, uav2Act,sess)
	defender.train_critic(s0, defActEdge, attActEdge, uav2Act,[[target]],sess)

	""" now update target network """
	sess.run([actorTargetUpOp,criticTargetUpOp])

	return 42


"""
uav2: uav2 object
uav2_target: uav2 target object
D: list, uav2 replay memory
oldState: [1,6] matrix
defActEdge: [1,N_edges] matrix
attActEdge: [1,N_edges] matrix
uav2Act: [1,1] matrix
reward: scalar
newState: [1,6] matrix
terminal: bool
eps: episode number
sess: current session
actorTargetUpOp: uav2_actor_target_update
criticTargetUpOp: uav2_critic_target_update
"""
def trainUav2(uav2, uav2_target, D,
 oldState, defActEdge, attActEdge, uav2Act, reward, newState,terminal, eps, sess, 
 actorTargetUpOp, criticTargetUpOp,
 oldFound, newFound):
	
	if (not oldFound) and (newFound):
		# fake reward
		reward = 5.0
	else:
		reward = 0.0

	if (oldFound) and (newFound):
		return 42

	D.append((oldState,defActEdge,attActEdge,uav2Act,reward,newState,terminal))

	if (eps < 5):
		return 42

	i = random.randint(0,len(D)-2)
	b_cur = D[i]

	s0 = b_cur[0]
	defActEdge = b_cur[1]
	attActEdge = b_cur[2]
	uav2Act = b_cur[3]
	r = b_cur[4]
	s1 = b_cur[5]
	term = b_cur[6]

	if term == False:
		b_nxt = D[i+1]
		target = r + 0.9 * uav2_target.Q(b_nxt[0],b_nxt[1], b_nxt[2], b_nxt[3],sess)[0][0]
	else:
		target = r

	uav2.train_actor(s0, defActEdge, attActEdge,sess)
	uav2.train_critic(s0, defActEdge, attActEdge, uav2Act, [[target]], sess)

	sess.run([actorTargetUpOp, criticTargetUpOp])
	
	return 42





isGui = False

numUav = 0
numUav2 = 1
n = 5
print("calling env creator")
env = Env(getGridGraphNxN(n), numUav, numUav2)
#env = Env(getDefaultGraph5x5, numUav, numUav2)
print("exiting env creator")

#g,_ = getDefaultGraph5x5()
g,_ = getGridGraphNxN(n)

""" initialize players """
defender = MaddpgDef("def",g)
defender_target = MaddpgDef("def_target",g)

uav2 = MaddpgUav2("uav2",g)
uav2_target = MaddpgUav2("uav2_target",g)

attacker = RandAtt(g)

""" initialize update operations """
defender_actor_target_init, defender_actor_target_update = create_init_update('def_actor','def_target_actor')
defender_critic_target_init, defender_critic_target_update = create_init_update('def_critic','def_target_critic')

uav2_actor_target_init, uav2_actor_target_update = create_init_update('uav2_actor','uav2_target_actor')
uav2_critic_target_init, uav2_critic_target_update = create_init_update('uav2_critic','uav2_target_critic')


""" initialize agent memory """
defender_D = list()
uav2_D = list()



""" only support one single uav2 now """
def run(env, defender, attacker, uav2, numEpisode, gui):


	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	""" initialize the target network """
	sess.run([defender_actor_target_init, defender_critic_target_init,
			  uav2_actor_target_init, uav2_critic_target_init])

	cumDefR = 0.0
	cumAttR = 0.0
	catch = 0
	avgDefUtilList = []

	for eps in range(0, numEpisode):

		stateDict = env.resetGame()
		attacker.reset()

		if eps % 100 == 0:
			print(eps)
		while(env.end == False):

			""" every agent make move """
			"""
			defOutG: networkxs object containing edge q values
			defEdgeQ: matrix of shape [1,N_edges] containing vector form of edges
			defAct: (u,v) containing the actual move of defender
			"""
			defOutG, defEdgeQ, defAct = defenderMove2(defender, stateDict, sess)
			"""
			uav2Act: here uav2Act is a scalar representting the direction of uav
			speed is hard coded as 1.0
			"""
			uav2Act = uav2Move(uav2, stateDict, sess)[0][0]
			"""
			attAct_graph: networkxs object containing edge q values of attacker
			attAct: (u,v) containing the actual move of attacker
			"""
			attAct_graph, attAct, attEdgeQ = attackerMove(attacker, stateDict)

			# print ("t=%d def act: %s att act:%s uav act dir:%s" 
			# 	% (env.t, str(defAct), str(attAct), str(uav2Act)))
			

			""" pass moves to the environment """
			""" get state after and reward from environment """
			stateDictAfter = env.step(defAct,attAct,[],[(uav2Act,1.0)])
		
			""" train each agent """
			""" train defender """
			defOldGraphTuple = utils_np.networkxs_to_graphs_tuple([_gtmp2intmp(stateDict["defState"])])
			defNewGraphTuple = utils_np.networkxs_to_graphs_tuple([_gtmp2intmp(stateDictAfter["defState"])])

			trainDefender(defender, defender_target, defender_D, 
				defOldGraphTuple, defEdgeQ, attEdgeQ,np.array([[uav2Act,1.0]]), stateDict["defR"], defNewGraphTuple, env.end, eps,
				sess, defender_actor_target_update, defender_critic_target_update)

			""" train uav2 """

			uav2OldState = stateDictToUav2State(stateDict)
			uav2NewState = stateDictToUav2State(stateDictAfter)

			oldFound = stateDict["uav2Found"][0]
			newFound = stateDictAfter["uav2Found"][0]

			trainUav2(uav2, uav2_target, uav2_D,
				uav2OldState, defEdgeQ, attEdgeQ, [[uav2Act]], stateDict["defR"], uav2NewState, env.end, eps, sess,
				uav2_actor_target_update, uav2_critic_target_update,
				oldFound, newFound)

			stateDict = stateDictAfter

			cumDefR += stateDict["defR"]
			cumAttR += stateDict["attR"]
			if stateDict["defR"] > 0:
				catch += 1

		avgDefUtilList.append(cumDefR/(eps+1.0))

	with open("data/avgDefUtilEps.pkl","wb") as f:
		pickle.dump([avgDefUtilList, numEpisode], f)

	""" compute average defender and attacker utility """
	avgDefR = cumDefR / numEpisode
	avgAttR = cumAttR / numEpisode 

	print(cumDefR,cumAttR,catch)
	print(catch)
	print ("numEpisode=%d avgDefR=%s avgAttR=%s" 
		% (numEpisode, str(avgDefR), str(avgAttR)))


	return 42










def main():

	run(env, defender, attacker, uav2, 5000, False)

	


if __name__ == '__main__':
	main()






























