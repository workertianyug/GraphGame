from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
import tensorflow as tf

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

def _create_feature(attr, fields):
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
		for attr in nodeFeature.keys():
			if attr != "features":
				del nodeFeature[attr]


	for u, v, features in gtemp.edges(data=True):
		intmp.add_edge(u, v, features=[1.0]) #TODO: change this to []

	intmp.graph["features"] = [0.0]

	return intmp

"""
from template graph to example target graph to create placeholder
input: networkx
output: networkx
"""
def _gtmp2ttmp(gtemp):
	ttmp = gtemp.copy()
	
	for nodeIndex, nodeFeature in gtemp.nodes(data=True):
		ttmp.add_node(nodeIndex, features=[])
		
	for nodeIndex, nodeFeature in ttmp.nodes(data=True):
		for attr in nodeFeature.keys():
			if attr != "features":
				del nodeFeature[attr]


	for u, v, features in gtemp.edges(data=True):
		ttmp.add_edge(u, v, features=[1.0]) # this is q value

	ttmp.graph["features"] = [0.0]

	return ttmp

def make_all_runnable_in_session(*args):
	"""Lets an iterable of TF graphs be output from a session as NP graphs."""
	return [utils_tf.make_runnable_in_session(a) for a in args]


gtemp,_ = getDefaultGraph5x5()

g0 = _gtmp2intmp(gtemp)
t0 = _gtmp2ttmp(gtemp)

inputPh = utils_tf.placeholders_from_networkxs([_gtmp2intmp(gtemp)])
targetPh = utils_tf.placeholders_from_networkxs([_gtmp2ttmp(gtemp)])


# input state
input_graphs = utils_np.networkxs_to_graphs_tuple([g0])

# target 
target_graphs = utils_np.networkxs_to_graphs_tuple([t0])

feed_dict = {inputPh: input_graphs, targetPh: target_graphs}

# for critic network

criticModel = models.EncodeProcessDecode(edge_output_size=1, node_output_size=0)


# number of edges 
N_edges = len(gtemp.edges)

# list of GraphTuple of placeholders
critic_output_ops = criticModel(inputPh, 1)

critic_output_ops_edge = critic_output_ops[0].edges

critic_output_ops_edge = tf.transpose(critic_output_ops_edge)

# now actor_output_ops_edge has shape [1,N_edges]
actor_output_ops_edge = tf.slice(critic_output_ops_edge,[0,0],[1,N_edges])

# now add in the action from other agents (attacker and uav)

attActHolder = tf.placeholder(tf.float64,shape=[1,N_edges])

uav2ActHolder = tf.placeholder(tf.float64,shape=[1,2])

exten = tf.concat([actor_output_ops_edge, attActHolder, uav2ActHolder],axis=1)

out = tf.layers.dense(exten,64)

out = tf.layers.dense(out,1)

inputPh, targetPh = make_all_runnable_in_session(inputPh, targetPh)

# critic loss

target_Q_ph = tf.placeholder(shape=[1,1],dtype = tf.float64)
critic_loss_op = tf.reduce_mean(tf.square(target_Q_ph - out))


critic_optimizer = tf.train.AdamOptimizer(1e-3)

step_op = critic_optimizer.minimize(critic_loss_op)

try:
  sess.close()
except NameError:
  pass
sess = tf.Session()
sess.run(tf.global_variables_initializer())


attActAry = np.random.rand(1,105)

uav2ActAry = np.random.rand(1,2)

target_Q_val = [[1.0]]

# print("test")
# critic_values = sess.run({
#     "outputs": critic_output_ops,
#     "out": out
# }, feed_dict={inputPh: input_graphs, attActHolder: attActAry, uav2ActHolder: uav2ActAry})


# this currently does not support batch training

print("train")
critic_train = sess.run({
	"step" : step_op,
	"loss": critic_loss_op
	}, feed_dict={inputPh:input_graphs, attActHolder: attActAry, uav2ActHolder: uav2ActAry, target_Q_ph:target_Q_val}) 


# actor network: should it output graph or vector
# -> output both: graph for making action, vector for other agent to train











