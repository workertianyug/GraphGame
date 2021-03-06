from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
from env import *

import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import matplotlib.pyplot as plt

from scipy import spatial
import tensorflow as tf

from collections import deque

import random

OBSERVEEPS = 5
EPSILON = 0.01

"""
things todo:

"""
class MsgDef(object):

	"""
	gtemp: template graph used to create placeholder
		   should be a networkx graph given by env
	"""
	def __init__(self,gtemp):
		tf.reset_default_graph()
		with tf.variable_scope("defscope"):
			self.SEED = 3
			self.GAMMA = 0.99
			self.BATCH = 5
			random.seed(a=self.SEED)
			self.random = np.random.RandomState(seed=self.SEED)
			np.random.seed(self.SEED)
			tf.set_random_seed(self.SEED)
			""" replay memory """
			self.D = deque()

			
			""" initialize and setup the network """
			self.inputPh = utils_tf.placeholders_from_networkxs(
				[self._gtmp2intmp(gtemp)])
			self.targetPh = utils_tf.placeholders_from_networkxs(
				[self._gtmp2ttmp(gtemp)])

			self.numProcessingSteps = 10

			self.model = models.EncodeProcessDecode(edge_output_size=1,
													node_output_size=0)

			self.output_ops_tr = self.model(self.inputPh, self.numProcessingSteps)

			self.loss_ops_tr = self._create_loss_ops(self.targetPh, self.output_ops_tr)

			self.loss_op_tr = sum(self.loss_ops_tr) / self.numProcessingSteps

			self.learning_rate = 1e-3
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.step_op = self.optimizer.minimize(self.loss_op_tr)

			self.inputPh, self.targetPh = self._make_all_runnable_in_session(self.inputPh, 
																		 self.targetPh)
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())

			self.epsilon = 0.01

			self.outg = None

	def _create_feature(self, attr, fields):
		return np.hstack([np.array(attr[field], dtype=float) for field in fields])


	def _create_loss_ops(self,target_op, output_ops):
		loss_ops = [
		  tf.losses.sigmoid_cross_entropy(target_op.edges, output_op.edges)
		  for output_op in output_ops
		]
		return loss_ops

	def _make_all_runnable_in_session(self, *args):
		"""Lets an iterable of TF graphs be output from a session as NP graphs."""
		return [utils_tf.make_runnable_in_session(a) for a in args]


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

	def act(self,defState, defNode, eps):
		"""
		parse defstate to the input format for feed_dict
		evaulate network to give action
		"""
		if (defState.nodes[defNode]["isDef"] != 1):
			raise ValueError("def location doesn't match")

		""" if episod is low or random value is les than epsilon
			take random action
		"""
		if (eps < OBSERVEEPS) or (random.random() < EPSILON):
			validActions = list(defState.out_edges([defNode]))
			a = self.random.randint(0, len(validActions),size=1)[0]
			return validActions[a]

		gin = self._gtmp2intmp(defState)
		test_values = self.sess.run({
			"outputs":self.output_ops_tr
			}, feed_dict={self.inputPh: utils_np.networkxs_to_graphs_tuple([gin])})

		outg = utils_np.graphs_tuple_to_networkxs(test_values["outputs"][-1])[0]
		outg = nx.DiGraph(outg)
		self.outg = outg

		validActions = list(defState.out_edges([defNode]))

		qdict = dict()

		for e in validActions:
			qdict[e] = outg.get_edge_data(*e)["features"][0]

		return max(qdict, key=qdict.get)
		# return max(qdict, key=qdict.get), outg, qdict



	"""
	oldState: networkx graph
	action: (x,y)
	reward: float
	newState: networkx graph
	terminal: bool
	"""
	def train(self,oldState, action, reward, newState, terminal, eps):
		"""
		store newState in the replay memory
		sample transitions into feed_dict
		train the network
		"""
		self.D.append((oldState,action,reward,newState,terminal))

		""" if epsilon is too low, don't train, just observe """
		if (eps < OBSERVEEPS):
			return 42

		minibatch = random.sample(self.D, self.BATCH)
		"""
		evaluate network from newState
		change output based on r & a
		retrain network using oldState
		"""
		inputs = []
		targets = []

		for b in minibatch:
			s0 = b[0]
			a = b[1]
			r = b[2]
			s1 = b[3]
			term = b[4]

			# parse s1 to feed
			gs1 = self._gtmp2intmp(s1)
			# get output from feeding s1
			testValues = self.sess.run({
				"outputs":self.output_ops_tr
				}, feed_dict={self.inputPh: utils_np.networkxs_to_graphs_tuple([gs1])})

			outs1 = utils_np.graphs_tuple_to_networkxs(testValues["outputs"][-1])[0]
			outs1 = nx.DiGraph(outs1)
			# change output based on r and a

			
			if terminal == False:
				validActions = list(outs1.out_edges([a[1]]))
				qdict = dict()
				for e in validActions:
					qdict[e] = outs1.get_edge_data(*e)["features"][0]

				v = max(qdict.values())
				outs1.add_edge(*a, features=np.array([r + v]))
			else:
				outs1.add_edge(*a, features=np.array([r]))
			# now outs1 can be trained as target for s0

			outs1.graph['features'] = [0.0]
			# TODO: write new functions to parse state to standard graphs
			inputs.append(self._gtmp2intmp(s0))
			targets.append(outs1)

		feed_dict = {self.inputPh: utils_np.networkxs_to_graphs_tuple(inputs),
					 self.targetPh: utils_np.networkxs_to_graphs_tuple(targets)}

		# apply gradient descent
		train_value = self.sess.run({
					  "step":self.step_op,
					  "target":self.targetPh,
					  "loss":self.loss_op_tr,
					  "outputs":self.output_ops_tr
					  }, feed_dict = feed_dict)

		
		# print(train_value["loss"])
		
		return 42



# g = getDefaultGraph0()
# d = MsgDef(g)
# g.add_node(0, isDef=1)
# a,dg,d = d.act(g,0)


# dg.add_edge(0,0,features=[77.0])











