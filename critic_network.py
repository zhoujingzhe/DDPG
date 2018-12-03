from keras.optimizers import Adam
import tensorflow as tf 
from keras.layers import Input, Dense, merge
from keras.models import Model
import keras.backend as K
import math


LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class CriticNetwork:
	"""docstring for CriticNetwork"""
	def __init__(self,sess,state_dim,action_dim):
		self.time_step = 0
		self.sess = sess
		# create q network
		self.critic_net, self.state, self.action, self.Q_value = self.create_q_network(state_dim,action_dim)

		# create target q network (the same structure with q network)
		self.critic_target_net, self.target_state, self.target_action, self.Q_target_value = self.create_q_network(state_dim, action_dim)

		self.action_grads = tf.gradients(self.critic_net.output, self.action)  # GRADIENTS for policy update
		# initialization 
		self.sess.run(tf.initialize_all_variables())
			
		self.update_target()

	def create_q_network(self,state_dim,action_dim):
		# the layer size could be changed
		state_input = Input([None,state_dim])
		action_input = Input([None,action_dim])
		X_state = Dense(LAYER1_SIZE, input_dim=K.shape(state_input), activation='relu')(state_input)
		X_state = Dense(LAYER2_SIZE, input_dim=K.shape(X_state))(X_state)
		X_action = Dense(LAYER2_SIZE, input_dim=K.shape(action_input))(action_input)
		X = merge([X_state, X_action],mode='sum')
		q_value_output = Dense(1, input_dim=K.shape(X), activation='relu')(X)
		model = Model(input=[state_input, action_input], output=q_value_output)
		adam = Adam(lr=self.LEARNING_RATE)
		model.compile(loss='mse', optimizer=adam)
		return model, state_input, action_input, q_value_output

	def update_target(self):
		critic_weights = self.critic_net.get_weights()
		critic_target_weights = self.critic_target_net.get_weights()
		for i in range(len(critic_weights)):
			critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
		self.critic_target_net.set_weights(critic_target_weights)

	def gradients(self, state_batch, action_batch):
		return self.sess.run(self.action_grads, feed_dict={
			self.state: state_batch,
			self.action: action_batch
			})[0]

	def target_q(self, state_batch, action_batch):
		return self.critic_target_net([state_batch, action_batch])

	def q_value(self, state_batch, action_batch):
		return self.critic_net([state_batch, action_batch])
'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"

	def save_network(self,time_step):
		print 'save critic-network...',time_step
		self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''
		