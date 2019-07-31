import networkx as nx
import numpy as np
import random
import tensorflow as tf
import tflearn

from replay_buffer import *

class ActorNetwork(object):
	"""
	Input to the network is the state, output is the action
	under a deterministic policy.

	The output layer activation is a tanh to keep the action
	between -action_bound and action_bound
	"""

	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, myid):
		self.myid = myid
		with tf.variable_scope("actor"+str(myid)):
			self.sess = sess
			self.s_dim = state_dim
			self.a_dim = action_dim
			self.action_bound = action_bound
			self.learning_rate = learning_rate
			self.tau = tau
			self.batch_size = batch_size

			# Actor Network
			self.inputs, self.out, self.scaled_out = self.create_actor_network()

			self.network_params = tf.trainable_variables(scope = "actor"+str(myid))

			# Target Network
			self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

			self.target_network_params = tf.trainable_variables(scope="actor"+str(myid))[
				len(self.network_params):]

			# Op for periodically updating target network with online network
			# weights
			self.update_target_network_params = \
				[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
													  tf.multiply(self.target_network_params[i], 1. - self.tau))
					for i in range(len(self.target_network_params))]

			# This gradient will be provided by the critic network
			self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

			# Combine the gradients here
			self.unnormalized_actor_gradients = tf.gradients(
				self.scaled_out, self.network_params, -self.action_gradient)
			# print "---"
			# # print self.unnormalized_actor_gradients
			# print self.network_params
			# print len(self.network_params)
			# return 
			self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

			# Optimization Op
			self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
				apply_gradients(zip(self.actor_gradients, self.network_params))

			self.num_trainable_vars = len(
				self.network_params) + len(self.target_network_params)

	def create_actor_network(self):
		inputs = tflearn.input_data(shape=[None, self.s_dim])
		net = tflearn.fully_connected(inputs, 400)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)
		net = tflearn.fully_connected(net, 300)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)
		# Final layer weights are init to Uniform[-3e-3, 3e-3]
		w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
		out = tflearn.fully_connected(
			net, self.a_dim, activation='tanh', weights_init=w_init)
		# Scale output to -action_bound to action_bound
		scaled_out = tf.multiply(out, self.action_bound)
		return inputs, out, scaled_out

	def train(self, inputs, a_gradient):
		self.sess.run(self.optimize, feed_dict={
			self.inputs: inputs,
			self.action_gradient: a_gradient
		})

	def predict(self, inputs):
		return self.sess.run(self.scaled_out, feed_dict={
			self.inputs: inputs
		})

	def predict_target(self, inputs):
		return self.sess.run(self.target_scaled_out, feed_dict={
			self.target_inputs: inputs
		})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars


class CriticNetwork(object):
	"""
	Input to the network is the state and action, output is Q(s,a).
	The action must be obtained from the output of the Actor network.

	"""

	def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars, myid):

		with tf.variable_scope("critic"+str(myid)):
			self.myid = myid
			self.sess = sess
			self.s_dim = state_dim
			self.a_dim = action_dim
			self.learning_rate = learning_rate
			self.tau = tau
			self.gamma = gamma

			# Create the critic network
			self.inputs, self.action, self.out = self.create_critic_network()

			self.network_params = tf.trainable_variables(scope="critic"+str(myid))[num_actor_vars:]

			# Target Network
			self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

			self.target_network_params = tf.trainable_variables(scope="critic"+str(myid))[(len(self.network_params) + num_actor_vars):]

			# Op for periodically updating target network with online network
			# weights with regularization
			self.update_target_network_params = \
				[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
				+ tf.multiply(self.target_network_params[i], 1. - self.tau))
					for i in range(len(self.target_network_params))]

			# Network target (y_i)
			self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

			# Define loss and optimization Op
			self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
			self.optimize = tf.train.AdamOptimizer(
				self.learning_rate).minimize(self.loss)

			# Get the gradient of the net w.r.t. the action.
			# For each action in the minibatch (i.e., for each x in xs),
			# this will sum up the gradients of each critic output in the minibatch
			# w.r.t. that action. Each output is independent of all
			# actions except for one.
			self.action_grads = tf.gradients(self.out, self.action)

	def create_critic_network(self):
		inputs = tflearn.input_data(shape=[None, self.s_dim])
		action = tflearn.input_data(shape=[None, self.a_dim])
		net = tflearn.fully_connected(inputs, 400)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)

		# Add the action tensor in the 2nd hidden layer
		# Use two temp layers to get the corresponding weights and biases
		t1 = tflearn.fully_connected(net, 300)
		t2 = tflearn.fully_connected(action, 300)

		net = tflearn.activation(
			tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

		# linear layer connected to 1 output representing Q(s,a)
		# Weights are init to Uniform[-3e-3, 3e-3]
		w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
		out = tflearn.fully_connected(net, 1, weights_init=w_init)
		return inputs, action, out

	def train(self, inputs, action, predicted_q_value):
		return self.sess.run([self.out, self.optimize], feed_dict={
			self.inputs: inputs,
			self.action: action,
			self.predicted_q_value: predicted_q_value
		})

	def predict(self, inputs, action):
		return self.sess.run(self.out, feed_dict={
			self.inputs: inputs,
			self.action: action
		})

	def predict_target(self, inputs, action):
		return self.sess.run(self.target_out, feed_dict={
			self.target_inputs: inputs,
			self.target_action: action
		})

	def action_gradients(self, inputs, actions):
		return self.sess.run(self.action_grads, feed_dict={
			self.inputs: inputs,
			self.action: actions
		})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
	def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
		self.theta = theta
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
				self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

	def __repr__(self):
		return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
	episode_reward = tf.Variable(0.)
	tf.summary.scalar("Reward", episode_reward)
	episode_ave_max_q = tf.Variable(0.)
	tf.summary.scalar("Qmax Value", episode_ave_max_q)

	summary_vars = [episode_reward, episode_ave_max_q]
	summary_ops = tf.summary.merge_all()

	return summary_ops, summary_vars



class DdpgUav2(object):

	def __init__(self,myid):
		self.myid = myid
		""" whether this uav has found the attacker """
		self.isMeFound = False

		self.sess = tf.Session()
		self.seed = myid

		""" self pos, def pos, att pos """
		state_dim = 6 
		""" direction """
		action_dim = 1
		""" todo: change this """
		action_bound = 2.0

		self.batch_size = 1

		self.actor = ActorNetwork(self.sess, state_dim, action_dim, action_bound,
					 0.0001, 0.001, self.batch_size,self.myid)
		self.critic = CriticNetwork(self.sess, state_dim, action_dim, 0.0001,
									0.0001,0.99,self.actor.get_num_trainable_vars(),self.myid)
		self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

		self.sess.run(tf.global_variables_initializer())

		self.actor.update_target_network()
		self.critic.update_target_network()

		self.replay_buffer = ReplayBuffer(1000000, self.seed)

	"""
	take in uav2State
	parse to list [self position, defender position, attacker position if any]
	TODO:
	if attacker is not found, should we use -1 or 0.0 or shouldnt include this?
	"""
	def _unparseState(self,uav2State):

		myPos = uav2State.graph["uav2Poss"][self.myid]
		defPos = uav2State.graph["defPos"]
		attPos = uav2State.graph["attPos"]

		# print "---"
		# print uav2State.graph["uav2Poss"]
		# print myPos
		# print defPos
		# print attPos
		feature = np.concatenate([myPos, defPos, attPos])



		isMeFound = uav2State.graph["uav2Found"][self.myid]

		return feature, isMeFound




	def train(self,oldState, action, reward, newState, terminal, eps=0):
		# s = oldState
		# a = action
		# r = reward
		# s2 = newState

		oldFeature, oldIsMeFound = self._unparseState(oldState)
		newFeature, newIsMeFound = self._unparseState(newState)
		s = oldFeature
		s2 = newFeature
		a = action[0]

		if (not oldIsMeFound) and (newIsMeFound):
			r = 5.0
		else:
			r = 0.0

		""" if already found the attacker, stop training """
		if (oldIsMeFound) and (newIsMeFound):
			self.isMeFound = True
			return

		self.replay_buffer.add(np.reshape(s, (self.actor.s_dim,)), np.reshape(a, (self.actor.a_dim,)), r,
							  terminal, np.reshape(s2, (self.actor.s_dim,)))

		# Keep adding experience to the memory until
		# there are at least minibatch size samples
		if self.replay_buffer.size() > 1:
			s_batch, a_batch, r_batch, t_batch, s2_batch = \
				self.replay_buffer.sample_batch(self.batch_size)

			# Calculate targets
			target_q = self.critic.predict_target(
				s2_batch, self.actor.predict_target(s2_batch))

			y_i = []
			for k in range(self.batch_size):
				if t_batch[k]:
					y_i.append(r_batch[k])
				else:
					y_i.append(r_batch[k] + self.critic.gamma * target_q[k])
		

			# Update the critic given the targets
			predicted_q_value, _ = self.critic.train(s_batch,
			 a_batch, np.reshape(y_i, (1, 1)))

			# ep_ave_max_q += np.amax(predicted_q_value)

			a_outs = self.actor.predict(s_batch)
			grads = self.critic.action_gradients(s_batch, a_outs)
			self.actor.train(s_batch, grads[0])

			# Update target networks
			self.actor.update_target_network()
			self.critic.update_target_network()



	def act(self, uav2State,eps=0):
		feature,isMeFound = self._unparseState(uav2State)
		a = self.actor.predict(np.reshape(feature, (1, self.actor.s_dim))) + self.actor_noise()
		""" velocity is limited to 1 for now """
		return (a[0][0],1.0)



	def reset(self):
		return 42




# d = DdpgUav2()
# a = d.act([1,1,1,1,1,1])
# d.train([1,1,1,1,1,1], 0.0, 1.0,[0,0,0,0,0,0],False,1)





# multiple instance does not work? why?
# why keep sticking to the edge?
# sess1 = tf.Session()
# a1 = ActorNetwork(sess1, 6, 1, 2.0 ,0.0001, 0.001, 1)

# sess2 = tf.Session()
# a2 = ActorNetwork(sess2, 6, 1, 2.0 ,0.0001, 0.001, 1)












