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
ACTION_SET=6
BATCH_SIZE= 32
UPDATE_FREQUENCY=4
FRAME_COUNT_FOR_ONE_PICTURE=4
EXPLORATION_STEPS = 2000000
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
INITIAL_REPLAY_SIZE = 60000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
LEARNING_FACTOR = 0.00027
INFO_GRAPH_WEIGHTS_DIR='./mario_0512/'


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

def init_logger2(outdir):
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	
	# create console handler and set level to info
	handler = logging.FileHandler(outdir)
	handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger

logger = init_logger(INFO_GRAPH_WEIGHTS_DIR + 'info.log')
logger2 = init_logger2(INFO_GRAPH_WEIGHTS_DIR + 'action.log')

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
	allactionlist = np.zeros((ACTION_SET,6))
	allactionlist[0] = [0,1,0,0,0,0]
	allactionlist[1] = [0,0,0,1,0,0]
	allactionlist[2] = [0,0,0,0,1,0]
	allactionlist[3] = [0,0,0,0,0,0]
	allactionlist[4] = [0,1,0,0,1,0]
	allactionlist[5] = [0,0,0,1,1,0]
	return allactionlist

def get_random_action(Q, state):
	action_predictions = Q.predict_on_batch( np.float32(state / 255.0).reshape((1,84,84,4)) ).flatten()

	index = action_predictions.argmax(axis=0)
	global all_action
	max_action = list(all_action[index])

	random_index = random.randint(0,ACTION_SET-1)
	
	global all_action
	while max_action == list(all_action[random_index]):
		random_index = random.randint(0,ACTION_SET-1)

	return all_action[random_index]




def preprocess_image(observation, last_observation):
	processed_observation = np.maximum(observation, last_observation)[80:210]
	processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
	#plt.imshow(processed_observation)
	#plt.show()
	return np.reshape(processed_observation, (84, 84, 1))

def argmax(Q,state, t=1):
	action_predictions = Q.predict_on_batch( np.float32(state / 255.0).reshape((1,84,84,4)) ).flatten()
	if t%100 == 0: logger2.debug(action_predictions)
	index = action_predictions.argmax(axis=0)
	
	# print('index: ', index)
	#print(action_predictions)

	return all_action[index]



def init_network():
	model = Sequential()
	model.add(keras.layers.Conv2D(32, (8,8), strides=(4, 4), data_format='channels_last', activation='relu', input_shape=(84,84,FRAME_COUNT_FOR_ONE_PICTURE) ) )

	model.add(keras.layers.Conv2D(64, (4,4), strides=(2, 2), activation='relu'))
	model.add(keras.layers.Conv2D(64, (3,3), strides=(1, 1), activation='relu'))
	model.add(keras.layers.Flatten())
	model.add(Dense(units=512, activation='relu'))
	model.add(Dense(units=ACTION_SET))


	model.compile(optimizer='adam',
			  loss='mse',
			  metrics=['accuracy'])
	return model


def VideoRecord(env, epoch):
	#Video recorder init
	video_recorder = VideoRecorder(env, INFO_GRAPH_WEIGHTS_DIR +'els'+str(epoch)+'.mp4', enabled=True)
	if epoch%10==1:
		video_recorder.enabled=True
	else:
		video_recorder.enabled=False

def get_initial_state(observation, last_observation):
	processed_observation = np.maximum(observation, last_observation)[30:210]
	processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
	state = [processed_observation for _ in range(FRAME_COUNT_FOR_ONE_PICTURE)]
	return np.stack(state, axis=2)

def get_action(epsilon, Q, state, logger, t):
	if epsilon >= random.uniform(0,1):
		action_t = get_random_action(Q, state)
		if t%100==0: logger2.debug("random")
	else:
		action_t = argmax(Q,state, t)
	if t%100==0: logger2.debug(action_t)
	return action_t

def train_network(D, Q, Q_freeze, LEARNING_FACTOR):

	batch = random.sample(D, BATCH_SIZE)

	y = np.zeros((BATCH_SIZE))
	actions_output = np.zeros((BATCH_SIZE, ACTION_SET))
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

def printParameters(mylogger):
	mylogger.info('ACTION_SET: ' + str(ACTION_SET))
	mylogger.info('BATCH_SIZE: ' + str(BATCH_SIZE))
	mylogger.info('UPDATE_FREQUENCY: ' + str(UPDATE_FREQUENCY))
	mylogger.info('FRAME_COUNT_FOR_ONE_PICTURE: ' + str(FRAME_COUNT_FOR_ONE_PICTURE))
	mylogger.info('EXPLORATION_STEPS: ' + str(EXPLORATION_STEPS)) 
	mylogger.info('INITIAL_EPSILON: ' + str(INITIAL_EPSILON))
	mylogger.info('FINAL_EPSILON: ' + str(FINAL_EPSILON)) 
	mylogger.info('INITIAL_REPLAY_SIZE: ' + str(INITIAL_REPLAY_SIZE)) 
	mylogger.info('NUM_REPLAY_MEMORY: ' + str(NUM_REPLAY_MEMORY))
	mylogger.info('LEARNING_FACTOR: ' + str(LEARNING_FACTOR)) 
	mylogger.info('INFO_GRAPH_WEIGHTS_DIR: ' + INFO_GRAPH_WEIGHTS_DIR)


def main():
	try:
		display = Display(visible=0, size=(1400,900))
		display.start()
		rewards_list = []
		original_rew_list = []
		D = collections.deque(maxlen = NUM_REPLAY_MEMORY) 
		epsgrdy = EpsilonGreedy()
		#Initialize neural networks--

		Q = init_network()
		logger = init_logger(INFO_GRAPH_WEIGHTS_DIR + 'info.log')
		printParameters(logger)
		logger2 = init_logger(INFO_GRAPH_WEIGHTS_DIR + 'action.log')
		
		if sys.argv[-1] == 'y' and os.path.isfile(INFO_GRAPH_WEIGHTS_DIR+'weights34.h5'):
			print('loading weights')
			Q.load_weights(INFO_GRAPH_WEIGHTS_DIR + 'weights34.h5')
			epsgrdy.epsilon = 0.1

		Q_freeze = init_network()
		#Q_freeze.set_weights(Q.get_weights())

		global all_action
		all_action = get_all_action()

		maxrange=50000

		for epoch in range(1,maxrange):
			
			env = gym.make('SuperMarioBros-1-1-v0')
			
			observation = env.reset()

			
			last_observation = observation
			observation, _, _, _ = env.step(env.action_space.sample())  # Do nothing
			
			#Preprocess image
			state = get_initial_state(observation, last_observation)

			Q_freeze.set_weights(Q.get_weights())
			
			sum_rewards = 0
			original_rewards = 0
			t = 0
			done=False

			same_action_policy = 0

			while not done:
				
				#video_recorder.capture_frame()

				last_observation = observation
				epsilon = epsgrdy.step_EPSILON()

				if t == 0 :
					last_action = get_action(epsilon, Q, state, logger2, t)
				else:
					last_action = action_t

				
				if t % 2 == 0 or t == 0:
					action_t = get_action(epsilon, Q, state, logger2, t)

				#print(action_t)

				if last_action.astype(np.uint8).tolist() == action_t.astype(np.uint8).tolist() and same_action_policy<10:
					observation, reward, done, info = env.step(action_t.astype(np.uint8).tolist())
					same_action_policy += 1
				elif last_action.astype(np.uint8).tolist() == action_t.astype(np.uint8).tolist() and same_action_policy >= 10: 
					observation, reward, done, info = env.step([0,0,0,0,0,0])
					same_action_policy = 0
				else:
					same_action_policy = 0
					observation, reward, done, info = env.step(action_t.astype(np.uint8).tolist())

				original_rewards += reward;
				if reward <= 0: reward = -1
				else: reward = 1
				
				#if reward > 0: reward = 1
				#elif reward < 0: reward = -1
				sum_rewards += reward

				processed_observation = preprocess_image(observation, last_observation)
				
				next_state = np.append(state[:, :, 1:], processed_observation, axis=2)

				# Store transition in replay memory
				D.append((state, action_t, reward, next_state, done))
				
				if epsgrdy.t >= INITIAL_REPLAY_SIZE and t%UPDATE_FREQUENCY == 0:
					# Train network
					Q = train_network(D, Q, Q_freeze, LEARNING_FACTOR)
				
				state = next_state

				if (t) % 500 == 0:
					Q_freeze.set_weights(Q.get_weights())
				
				t = t + 1
			rewards_list.append(sum_rewards)
			original_rew_list.append(original_rewards)
			if epoch % 100 == 0:
				Q.save_weights(INFO_GRAPH_WEIGHTS_DIR + 'weights' + str(int(epoch/100)) + '.h5')
			if epoch % 100 == 0:
				plt.plot(rewards_list)
				plt.savefig(INFO_GRAPH_WEIGHTS_DIR+'reward_diagram'+str(epoch)+'.png')
				plt.close()
				plt.plot(original_rew_list)
				plt.savefig(INFO_GRAPH_WEIGHTS_DIR+'original_diagram'+str(epoch)+'.png')
				plt.close()
			logger.info('EPSILON: '+ str(epsilon) + ' '+  str(t) + " lepesszam mellett, reward: " + str(sum_rewards) + "  eredeti reward: " + str(original_rewards))
	finally:
		
		plt.plot(rewards_list)
		plt.savefig(INFO_GRAPH_WEIGHTS_DIR+'reward_diagram_final.png')
		plt.close()
		Q.save_weights( INFO_GRAPH_WEIGHTS_DIR+ 'weights_final.h5')
		logger.info('--Felbeszakitas: EPSILON: '+ str(epsilon) + ' '+  str(t) + " lepesszam mellett, reward: " + str(sum_rewards))
		#video_recorder.close()
		#video_recorder.enabled = False



main()
