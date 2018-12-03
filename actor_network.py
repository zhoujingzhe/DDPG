import tensorflow as tf
from keras.layers import Dense, Input
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import math


# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_dim,action_dim):
		self.TAU = TAU
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		K.set_session(sess)
		# create actor network
		self.actor_net, self.weights, self.state = self.create_network(state_dim, action_dim)
		# create target actor network
		self.actor_target_net, self.target_weights, self.target_state = self.create_network(state_dim, action_dim)

		self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])
		self.Q_value = tf.placeholder(tf.float32, [None, self.action_dim])
		self.params_grad = tf.gradients(self.actor_net.output, self.weights, -self.action_gradient)
		grads = zip(self.params_grad, self.weights)
		self.optimizer = Adam(LEARNING_RATE).apply_gradients(grads)
		self.sess.run(tf.initialize_all_variables())
		self.update_target()

		#self.load_network()

	def create_network(self,state_dim,action_dim):
		state_input = Input([None,state_dim])
		X = Dense(LAYER1_SIZE, input_dim=K.shape(state_input),activation='relu')(state_input)
		X = Dense(LAYER2_SIZE, input_dim=K.shape(X), activation='relu')(X)
		action_output = Dense(action_dim, input_dim=K.shape(X), activation='tanh')(X)
		model = Model(input=state_input, output=action_output)
		return model, model.trainable_weights, state_input

	def update_target(self):
		weights = self.actor_net.get_weights()
		target_weights = self.actor_target_net.get_weights()
		for i in range(len(weights)):
			target_weights[i] = self.TAU * weights[i] + (1 - self.TAU) * target_weights[i]
		self.actor_target_net.set_weights(target_weights)

	def train(self, q_gradient_batch,state_batch, Q_value):
		self.sess.run(self.optimizer,feed_dict={
			self.action_gradient: q_gradient_batch,
			self.state: state_batch
			})
		self.actor_net.train_on_batch()

	def actions(self,state_batch):
		return self.actor_net(inputs=state_batch)

	def action(self,state):
		return self.actor_net(inputs=[state])

	def target_actions(self, state_batch):
		return self.actor_target_net(inputs=state_batch)

'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
	def save_network(self,time_step):
		print 'save actor-network...',time_step
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''

		
