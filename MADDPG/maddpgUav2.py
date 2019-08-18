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



class MaddpgUav2():

	def __init__(self, name,gtemp):



		self.name = name
		self.SEED = 3
		""" self pos, def pos, att pos """
		
		self.state_dim = 6
		self.inputPh = tf.placeholder(shape=[None,self.state_dim],dtype=tf.float64)


		""" self action, action from all other agents """
		N_edges = len(gtemp.edges)
		# self.allActPh = tf.placeholder(shape=[None,N_edges*2+2],dtype=tf.float64)

		self.defActPh = tf.placeholder(shape=[None,N_edges],dtype=tf.float64)
		self.uav2ActPh = tf.placeholder(shape=[None,1], dtype=tf.float64)
		self.attActPh = tf.placeholder(shape=[None,N_edges], dtype=tf.float64)

		""" set up the actor network """
		self.actor_out_ph = self.actor_network(name+"_actor")

		""" set up the critic network """
		self.critic_out_ph = self.critic_network(name+"_critic", 
			tf.concat([self.defActPh,self.attActPh,self.uav2ActPh],axis=1))

		""" structure to compute loss and optimize actor network """
		self.actor_loss_op = -tf.reduce_mean(
			self.critic_network(name+"_critic",
				tf.concat([self.defActPh, self.actor_out_ph, self.attActPh],axis=1),reuse=True))
		self.actor_optimizer = tf.train.AdamOptimizer(1e-3)
		# step operation to be placed in feed_dict
		self.actor_step_op = self.actor_optimizer.minimize(self.actor_loss_op)



		""" structure to compute loss and optimize critic network """
		self.targetQ_ph = tf.placeholder(shape=[None,1],dtype=tf.float64)
		self.critic_loss_op = tf.reduce_mean(tf.square(self.targetQ_ph - self.critic_out_ph))
		self.critic_optimizer = tf.train.AdamOptimizer(1e-3)
		self.critic_step_op = self.critic_optimizer.minimize(self.critic_loss_op)

		
	"""
	ciritc_network:
	take in state
	take in self action and action from all other agents
	output a single Q value
	"""
	def critic_network(self, name, action_input, reuse = False):

		with tf.variable_scope(name) as scope:

			if reuse:
				scope.reuse_variables()

			net = tf.layers.dense(action_input,256)
			net = tf.nn.relu(net)

			net = tf.concat([net,action_input], axis=-1)

			net = tf.layers.dense(net,64)

			net = tf.nn.relu(net)

			out = tf.layers.dense(net,1)

		return out


	"""
	actor network:
	take in state
	output the action of the uav2 (currently only return the direction)
	"""
	def actor_network(self,name):

		with tf.variable_scope(name) as scope:

			net = tf.layers.dense(self.inputPh, 256)
			net = tf.nn.relu(net)

			net = tf.layers.dense(net, 128)
			net = tf.nn.relu(net)

			out = tf.layers.dense(net, 1)

		return out



	def train_actor(self, state, defAct, attAct, sess):

		train_values = sess.run({
					   "step": self.actor_step_op,
					   "loss": self.actor_loss_op},
						feed_dict={self.inputPh: state, self.defActPh:defAct, 
						self.attActPh:attAct})

		# print(train_values["loss"])


	def train_critic(self,state,defAct,attAct,uav2Act,target,sess):

		train_values = sess.run({
			"step":self.critic_step_op,
			"loss": self.critic_loss_op},
			feed_dict={self.inputPh:state, self.defActPh:defAct,
			           self.attActPh:attAct, self.uav2ActPh:uav2Act, 
			           self.targetQ_ph:target})
		# print(train_values["loss"])

	def action(self,state,sess):

		return sess.run({"direc":self.actor_out_ph},
			feed_dict={self.inputPh:state})


	def Q(self, state, defAct, attAct, uav2Act, sess):

		return sess.run(self.critic_out_ph,
			feed_dict={self.inputPh:state, self.defActPh:defAct, 
			           self.attActPh:attAct,self.uav2ActPh:uav2Act})



if __name__ == '__main__':
	g,_ = getDefaultGraph5x5()

	u = MaddpgUav2("uav2",g)


	try:
		sess.close()
	except NameError:
		pass

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())


	N_edges = 105

	state = np.random.rand(1,6)
	defAct = np.random.rand(1,N_edges)
	attAct = np.random.rand(1,N_edges)
	uav2Act = np.random.rand(1,1)

	target = [[1.0]]

	print("train_actor")

	u.train_actor(state,defAct,attAct,sess)



	print("train_critic")

	u.train_critic(state, defAct,attAct,uav2Act,target,sess)


	print("action")

	u.action(state,sess)

	print("Q")

	u.Q(state, defAct, attAct, uav2Act, sess)



















