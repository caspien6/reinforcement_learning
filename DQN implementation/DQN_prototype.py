from random import randint
import skimage.transform as ST
from skimage.transform import resize
from skimage.color import rgb2gray
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import itertools, collections, gym, random, os, sys, logging
import numpy as np
from gym.monitoring import VideoRecorder
from gym.wrappers import Monitor
from pyvirtualdisplay import Display
import matplotlib.pyplot as plt
from keras import regularizers


global all_action
ACTION_SET_LENGTH=7
BATCH_SIZE= 32
STATE_LENGTH=4
EXPLORATION_STEPS = 1000000
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
INITIAL_REPLAY_SIZE = 20000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
LEARNING_FACTOR = 0.000025



class EpsilonGreedy():

	def __init__(self):
		self.epsilon = INITIAL_EPSILON
		self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
		self.t = 0

	def step_EPSILON(self):
		if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
			self.epsilon -= self.epsilon_step
		self.t += 1
		return self.epsilon


def get_all_action():
	allactionlist = np.zeros((ACTION_SET_LENGTH,6))
	allactionlist[0] = [0,1,0,0,0,0]
	allactionlist[1] = [0,0,0,1,0,0]
	allactionlist[2] = [1,0,0,0,0,0]
	allactionlist[3] = [0,0,0,0,0,0]
	allactionlist[4] = [0,0,1,0,0,0]
	allactionlist[5] = [1,1,0,0,0,0]
	allactionlist[6] = [1,0,0,1,0,0]
	return allactionlist

def get_random_action(Q, state):
	action_predictions = Q.predict_on_batch( np.float32(state / 255.0).reshape((1,84,84,4)) ).flatten()

	index = action_predictions.argmax(axis=0)
	global all_action
	max_action = list(all_action[index])

	random_index = random.randint(0,ACTION_SET_LENGTH-1)
	
	global all_action
	while max_action == list(all_action[random_index]):
		random_index = random.randint(0,ACTION_SET_LENGTH-1)

	return all_action[random_index]




def preprocess_image(observation, last_observation):
	processed_observation = np.maximum(observation, last_observation)[30:210]
	processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
	#plt.imshow(processed_observation)
	#plt.show()
	return np.reshape(processed_observation, (84, 84, 1))

def argmax(Q,state):
	action_predictions = Q.predict_on_batch( np.float32(state / 255.0).reshape((1,84,84,4)) ).flatten()
	index = action_predictions.argmax(axis=0)
	
	# print('index: ', index)
	#print(action_predictions)

	return all_action[index]



def init_network():
	model = Sequential()
	model.add(keras.layers.Conv2D(32, (8,8), strides=(4, 4), data_format='channels_last', activation='relu', input_shape=(84,84,STATE_LENGTH) ) )

	model.add(keras.layers.Conv2D(64, (4,4), strides=(2, 2), activation='relu'))
	model.add(keras.layers.Conv2D(64, (3,3), strides=(1, 1), activation='relu'))
	model.add(keras.layers.Flatten())
	model.add(Dense(units=512, activation='relu'))
	model.add(Dense(units=ACTION_SET_LENGTH))


	model.compile(optimizer='adam',
			  loss='mse',
			  metrics=['accuracy'])
	return model

def init_logger(outdir):
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	
	# create console handler and set level to info
	handler = logging.FileHandler(outdir)
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger

def VideoRecord(env, epoch):
	#Video recorder init
	video_recorder = VideoRecorder(env, './records/els'+str(epoch)+'.mp4', enabled=True)
	if epoch%10==1:
		video_recorder.enabled=True
	else:
		video_recorder.enabled=False

def get_initial_state(observation, last_observation):
	processed_observation = np.maximum(observation, last_observation)[30:210]
	processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
	state = [processed_observation for _ in range(STATE_LENGTH)]
	return np.stack(state, axis=2)

def get_action(epsilo, Q, state):
	if epsilo >= random.uniform(0,1):
		print("random:")
		action_t = get_random_action(Q, state)
	else:
		action_t = argmax(Q,state)
	return action_t

def train_network(D, Q, Q_freeze, LEARNING_FACTOR):

	batch = random.sample(D, BATCH_SIZE)

	y = np.zeros((BATCH_SIZE))
	actions_output = np.zeros((BATCH_SIZE, ACTION_SET_LENGTH))
	states = np.zeros((BATCH_SIZE, 84,84,4))

	global all_action

	for i in range(0, BATCH_SIZE):
		action_predictions = Q_freeze.predict_on_batch(np.float32( batch[i][3] / 255.0).reshape((1,84,84,4)) ).flatten() 
		index = action_predictions.argmax()
		y[i] = batch[i][2] + LEARNING_FACTOR * action_predictions[index]

		actions_output[i] = action_predictions[:]
		actions_output[i][index] = y[i]
		states[i] = batch[i][0]

	Q.train_on_batch(states[:], actions_output[:])
	return Q

def main():
	try:
		display = Display(visible=0, size=(1400,900))
		display.start()
		D = collections.deque(maxlen = NUM_REPLAY_MEMORY) 
		epsgrdy = EpsilonGreedy()
		#Initialize neural networks--

		Q = init_network()
		logger = init_logger('./records/els.log')

		if sys.argv[-1] == 'y' and os.path.isfile('./weights1.h5'):
			print('loading weights')
			Q.load_weights('weights1.h5')

		Q_freeze = init_network()
		#Q_freeze.set_weights(Q.get_weights())

		global all_action
		all_action = get_all_action()

		maxrange=2000

		for epoch in range(1,maxrange):
			
			env = gym.make('SuperMarioBros-1-1-v0')
			
			observation = env.reset()

			
			last_observation = observation
			observation, _, _, _ = env.step(env.action_space.sample())  # Do nothing
			
			#Preprocess image
			state = get_initial_state(observation, last_observation)

			
			
			sum_rewards = 0
			t = 0
			done=False
			while not done:
				
				#video_recorder.capture_frame()

				last_observation = observation
				epsilon = epsgrdy.step_EPSILON()

				print('EPSILON: ', epsilon)
				action_t = get_action(epsilon, Q, state)
				print(action_t)


				observation, reward, done, info = env.step(action_t.astype(np.uint8).tolist())
				

				if reward > 0: reward = 1
				elif reward < 0: reward = -1
				sum_rewards += reward

				processed_observation = preprocess_image(observation, last_observation)
				
				next_state = np.append(state[:, :, 1:], processed_observation, axis=2)

				# Store transition in replay memory
				D.append((state, action_t, reward, next_state, done))
				
				if epsgrdy.t >= INITIAL_REPLAY_SIZE and t%5 == 0:
					# Train network
					Q = train_network(D, Q, Q_freeze, LEARNING_FACTOR)
				
				state = next_state

				if (t) % 500 == 0:
					Q_freeze.set_weights(Q.get_weights())
				t = t + 1
			if epoch % 20 == 0:
				Q.save_weights('weights' + str(epoch/20) + '.h5')
			logger.info('EPSILON: '+ str(epsilon) + ' '+  str(t) + " lepesszam mellett, reward: " + str(sum_rewards))
	finally:
		print()
		#video_recorder.close()
		#video_recorder.enabled = False



main()
