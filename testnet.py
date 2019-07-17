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

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

# create an example input
# create an example placeholder
# connect to model

seed = 1 
rand = np.random.RandomState(seed=seed)

# """
# features: [isDef, isAtt, isUAV, reward, penetration time]
#     0
#   /   \
#  1  |	 3
#   \   /
#     2
# """

g0 = nx.DiGraph()
g0.graph['features'] = [0.0]
g0.add_node(0,features=[0.0,1.0,2.0,0.0,1.0])
g0.add_node(1,features=[1.0,1.0,2.0,0.0,1.0])
g0.add_node(2,features=[0.0,1.0,2.0,0.0,0.0]) # they are the features
g0.add_node(3,features=[0.0,2.0,4.0,1.0,0.0])
g0.add_edge(0,1,features=[0.0]) # edge should have no value
g0.add_edge(1,0,features=[0.0])
g0.add_edge(1,2,features=[0.0])
g0.add_edge(2,1,features=[0.0])
g0.add_edge(2,3,features=[0.0])
g0.add_edge(3,2,features=[0.0])
g0.add_edge(0,3,features=[0.0])
g0.add_edge(3,0,features=[0.0])
g0.add_edge(0,2,features=[0.0])
g0.add_edge(2,0,features=[0.0])
g0.add_edge(0,0,features=[0.0])
g0.add_edge(1,1,features=[0.0])
g0.add_edge(2,2,features=[0.0])
g0.add_edge(3,3,features=[0.0])

t0 = nx.DiGraph()
t0.graph['features'] = [0.0]
t0.add_node(0,features=[]) # node should have no feature
t0.add_node(1,features=[])
t0.add_node(2,features=[])
t0.add_node(3,features=[])
t0.add_edge(0,1,features=[0.0]) # this should be the q value for each action
t0.add_edge(1,0,features=[0.0])
t0.add_edge(1,2,features=[0.0])
t0.add_edge(2,1,features=[0.0])
t0.add_edge(2,3,features=[0.0])
t0.add_edge(3,2,features=[0.0])
t0.add_edge(0,3,features=[0.0])
t0.add_edge(3,0,features=[0.0])
t0.add_edge(0,2,features=[0.0])
t0.add_edge(2,0,features=[0.0])
t0.add_edge(0,0,features=[0.0])
t0.add_edge(1,1,features=[0.0])
t0.add_edge(2,2,features=[0.0])
t0.add_edge(3,3,features=[0.0])

t1 = nx.DiGraph()
t1.graph['features'] = [0.0]
t1.add_node(0,features=[]) # node should have no feature
t1.add_node(1,features=[])
t1.add_node(2,features=[])
t1.add_node(3,features=[])
t1.add_edge(0,1,features=[1.0]) # this should be the q value for each action
t1.add_edge(1,0,features=[1.0])
t1.add_edge(1,2,features=[1.0])
t1.add_edge(2,1,features=[1.0])
t1.add_edge(2,3,features=[1.0])
t1.add_edge(3,2,features=[1.0])
t1.add_edge(0,3,features=[0.0])
t1.add_edge(3,0,features=[0.0])
t1.add_edge(0,2,features=[0.0])
t1.add_edge(2,0,features=[0.0])
t1.add_edge(0,0,features=[0.0])
t1.add_edge(1,1,features=[0.0])
t1.add_edge(2,2,features=[0.0])
t1.add_edge(3,3,features=[0.0])


input_ph = utils_tf.placeholders_from_networkxs([g0])
target_ph = utils_tf.placeholders_from_networkxs([t0])


input_graphs = utils_np.networkxs_to_graphs_tuple([g0])
target_graphs = utils_np.networkxs_to_graphs_tuple([t0])

feed_dict = {input_ph: input_graphs, target_ph: target_graphs}



num_processing_steps_tr = 10
num_processing_steps_ge = 10

model = models.EncodeProcessDecode(edge_output_size=1, node_output_size=0)

output_ops_tr = model(input_ph, num_processing_steps_tr)


def create_loss_ops(target_op, output_ops):
	loss_ops = [
	  tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
	  for output_op in output_ops
	]
	return loss_ops

def make_all_runnable_in_session(*args):
	"""Lets an iterable of TF graphs be output from a session as NP graphs."""
	return [utils_tf.make_runnable_in_session(a) for a in args]

loss_ops_tr = create_loss_ops(target_ph, output_ops_tr)

loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr



# Optimizer.
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)


# Lets an iterable of TF graphs be output from a session as NP graphs.
input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

try:
  sess.close()
except NameError:
  pass
sess = tf.Session()
sess.run(tf.global_variables_initializer())


print("train")
train_values = sess.run({
      "step": step_op,
      "target": target_ph,
      "loss": loss_op_tr,
      "outputs": output_ops_tr
  }, feed_dict=feed_dict)


"""
evaluate but not train
"""
print("test")
test_values = sess.run({
        "outputs": output_ops_tr
    }, feed_dict={input_ph: input_graphs})




# take the last one
print("second test")


target_graphs = utils_np.networkxs_to_graphs_tuple([t1])

feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
"""
evaluate but compute loss
"""
test_values_2 = sess.run({
      "target": target_ph,
      "loss": loss_op_tr,
      "outputs": output_ops_tr
  }, feed_dict=feed_dict)


outGs = utils_np.graphs_tuple_to_networkxs(test_values["outputs"][-1])






















