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



class MaddpgDef():
	
	def __init__(self, name, gtemp):

		self.SEED = 3
		self.gtemp = gtemp
		random.seed(a = self.SEED)
		tf.set_random_seed(self.SEED)
		self.name = name

		self.inputPh = utils_tf.placeholders_from_networkxs([self._gtmp2intmp(gtemp)])

		# the scenrio: 1 def 1 att 1 uav: N_edges + N_edges + 2(uav action dim)
		# def act dim: N_edges
		N_edges = len(gtemp.edges)
		# question: split actions or concat all together?
		# answer: need to split actions
		# self.allActPh = tf.placeholder(tf.float64, shape=[1,N_edges*2+2])

		self.defActPh = tf.placeholder(tf.float64, shape=[1,N_edges])
		self.attActPh = tf.placeholder(tf.float64, shape=[1,N_edges])
		self.uav2ActPh = tf.placeholder(tf.float64, shape=[1,2])

		""" setting up the actor network """
		self.actor_out_graphs, self.actor_out_edge = self.actor_network(gtemp,name+"_actor")


		""" setting up the critic network """
		self.critic_out_ph = self.critic_network(gtemp, name+"_critic", 
			tf.concat([self.defActPh,self.attActPh,self.uav2ActPh],axis=1))
		

		""" structure to compute loss and optimize actor network """
		self.actor_loss_op = -tf.reduce_mean(
			self.critic_network(gtemp, name+"_critic", 
				tf.concat([self.actor_out_edge,self.attActPh,self.uav2ActPh],axis=1),reuse=True))

		self.actor_optimizer = tf.train.AdamOptimizer(1e-3)
		# step operation to be placed in feed_dict
		self.actor_step_op = self.actor_optimizer.minimize(self.actor_loss_op)


		""" structure to compute loss and optimize critic network """
		self.targetQ_ph = tf.placeholder(shape=[1,1],dtype=tf.float64)
		self.critic_loss_op = tf.reduce_mean(tf.square(self.targetQ_ph - self.critic_out_ph))
		self.critic_optimizer = tf.train.AdamOptimizer(1e-3)
		# step operation to be placed in feed_dict
		self.critic_step_op = self.critic_optimizer.minimize(self.critic_loss_op)

	"""
	input placeholders of critic network: 
	self.inputPh: hold the GraphTuple of placeholders, containing the state
	self.defActPh, self.attActPh, self.uav2ActPh: concatenation of all agents's vector actions

	output placeholders of critic network:
	out: a single Q value representing the Q value of the action
	"""
	def critic_network(self,gtemp, name, action_input_ph, reuse=False):

		with tf.variable_scope(name) as scope:

			if reuse:
				scope.reuse_variables()

			N_edges = len(gtemp.edges)

			criticModel = models.EncodeProcessDecode(edge_output_size=1,node_output_size=0)

			critic_output_ops = criticModel(self.inputPh,1)

			critic_output_ops_edge = critic_output_ops[0].edges

			critic_output_ops_edge = tf.transpose(critic_output_ops_edge)

			critic_output_ops_edge = tf.slice(critic_output_ops_edge,[0,0],[1,N_edges])

			exten = tf.concat([critic_output_ops_edge, action_input_ph],axis=1)

			out = tf.layers.dense(exten,64)

			out = tf.nn.relu(out)

			out = tf.layers.dense(out,1)

		return out
    
	"""
	input placeholders of actor network:
	self.inputPh: hold the GraphTuple of placeholders, containing the state

	output placeholders of actor network:
	actor_output_ops_graphs: GraphTuple of placeholders containing the full output 
	actor_putput_ops_edge: containing the values of all edge in vector form
	"""
	def actor_network(self,gtemp,name):
		
		with tf.variable_scope(name) as scope:

			N_edges = len(gtemp.edges)

			actorModel = models.EncodeProcessDecode(edge_output_size=1,node_output_size=0)

			actor_output_ops_graphs = actorModel(self.inputPh,1)

			actor_output_ops_edge = actor_output_ops_graphs[0].edges
			actor_output_ops_edge = tf.transpose(actor_output_ops_edge)
			actor_output_ops_edge = tf.slice(actor_output_ops_edge,[0,0],[1,N_edges])

		return actor_output_ops_graphs, actor_output_ops_edge


	"""
	state: graphTuple containing the current state
	allActPh: [1,N] vector containing action of all agents
	uav2Act: [1,2] vector containg direction, velocity
	"""
	def train_actor(self,state,attAct,uav2Act,sess):
		sess.run(self.actor_step_op, 
			feed_dict={self.inputPh:state, self.attActPh:attAct, self.uav2ActPh:uav2Act})

	def train_critic(self,state,defAct,attAct,uav2Act,target,sess):
		sess.run(self.critic_step_op,
			feed_dict={self.inputPh:state, self.defActPh:defAct,
			           self.attActPh:attAct, self.uav2ActPh:uav2Act, 
			           self.targetQ_ph:target})

	""" special note: the returned value is a dictionary """
	def action(self,state,sess):
		return sess.run({"graph":self.actor_out_graphs,"edge":self.actor_out_edge},
			feed_dict={self.inputPh:state})


	def Q(self,state, defAct, attAct, uav2Act, sess):
		return sess.run(self.critic_out_ph, 
			feed_dict={self.inputPh:state, self.defActPh:defAct,
					   self.attActPh:attAct, self.uav2ActPh:uav2Act})





	def _create_feature(self, attr, fields):
		return np.hstack([np.array(attr[field], dtype=float) for field in fields])


	"""
	from template graph to example input graph to create placeholder 
	input: networkx
	output: networkx
	"""
	def _gtmp2intmp(self,gtemp):
		intmp = gtemp.copy()
		inNodeFields = ["isDef","isAtt","numUav","r","d","x","y"]

		for nodeIndex, nodeFeature in gtemp.nodes(data=True):
			intmp.add_node(nodeIndex, 
						   features=self._create_feature(nodeFeature, inNodeFields))
			
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
	def _gtmp2ttmp(self,gtemp):
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



# testing code below

if __name__ == '__main__':

	g,_ = getDefaultGraph5x5()

	d = MaddpgDef("def",g)

	try:
		sess.close()
	except NameError:
		pass

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())


	N_edges = 105
	defAct = np.random.rand(1,N_edges)
	attAct = np.random.rand(1,N_edges)
	uav2Act = np.random.rand(1,2)

	state = utils_np.networkxs_to_graphs_tuple([d._gtmp2intmp(g)])


	print("train_actor")
	# allAct = np.random.rand(1,2*N_edges+2)

	d.train_actor(state,attAct,uav2Act,sess)


	print("train_critic")
	target = [[5.0]]

	d.train_critic(state,defAct,attAct,uav2Act, target, sess)


	print("action")

	d.action(state,sess)

	print("Q")

	d.Q(state,defAct,attAct,uav2Act,sess)



























