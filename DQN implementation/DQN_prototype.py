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
ACTION_SET_LENGTH=13
BATCH_SIZE= 32
STATE_LENGTH=4



def get_all_action():
	allactionlist = np.zeros((13,6))
	allactionlist[0] = [0,1,0,0,0,0]
	allactionlist[1] = [0,0,0,1,0,0]
	allactionlist[2] = [1,0,0,0,0,0]
	allactionlist[3] = [0,0,0,0,0,0]
	allactionlist[4] = [0,0,1,0,0,0]
	allactionlist[5] = [1,1,0,0,0,0]
	allactionlist[6] = [1,0,0,1,0,0]
	allactionlist[7] = [0,0,0,0,1,0]
	allactionlist[8] = [0,0,0,0,0,1]
	allactionlist[9] = [0,0,0,1,1,0]
	allactionlist[10] = [0,0,0,1,0,1]
	allactionlist[11] = [0,1,0,0,1,0]
	allactionlist[12] = [0,1,0,0,0,1]
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

def get_epsilon(x, maxrange):
	percent = 0.95
	threshold = maxrange - 500
	if threshold < x: x = threshold

	return percent - (x*percent / threshold * 1.12 )

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

def get_action(epsilon, Q, state):
	if epsilon >= random.uniform(0,1):
		print("random:")
		action_t = get_random_action(Q, state)
	else:
		action_t = argmax(Q,state)
	return action_t

def train_network(D, Q, Q_freeze, learning_factor):

	batch = random.sample(D, BATCH_SIZE)

	y = np.zeros((BATCH_SIZE))
	actions_output = np.zeros((BATCH_SIZE, 13))
	states = np.zeros((BATCH_SIZE, 84,84,4))

	global all_action

	for i in range(0, BATCH_SIZE):
		action_predictions = Q_freeze.predict_on_batch(np.float32( batch[i][3] / 255.0).reshape((1,84,84,4)) ).flatten() 
		index = action_predictions.argmax()
		y[i] = batch[i][2] + learning_factor * action_predictions[index]

		actions_output[i] = action_predictions[:]
		actions_output[i][index] = y[i]
		states[i] = batch[i][0]

	Q.train_on_batch(states[:], actions_output[:])
	return Q

def main():
	try:
		display = Display(visible=0, size=(1400,900))
		display.start()
		D = collections.deque(maxlen = 100000) #Experience replay dataset (st,at,rt,st+1)

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

			epsilon = get_epsilon(float(epoch),float(maxrange))
			print('epsilon: ', epsilon)
			learning_factor = 0.000001
			sum_rewards = 0
			t = 0
			done=False
			while not done:
				
				#video_recorder.capture_frame()

				last_observation = observation

				action_t = get_action(epsilon, Q, state)
				print(action_t)


				observation, reward, done, info = env.step(action_t.astype(np.uint8).tolist())
				

				if reward > 0: reward = 1
				else: reward = -1
				sum_rewards += reward

				processed_observation = preprocess_image(observation, last_observation)
				
				next_state = np.append(state[:, :, 1:], processed_observation, axis=2)

				# Store transition in replay memory
				D.append((state, action_t, reward, next_state, done))

				if t >= BATCH_SIZE and t%2 == 0:
					# Train network
					Q = train_network(D, Q, Q_freeze, learning_factor)
				
				state = next_state

				if (t) % 500 == 0:
					Q_freeze.set_weights(Q.get_weights())
				t = t + 1
			if epoch % 5 == 0:
				Q.save_weights('weights' + str(epoch/5) + '.h5')
			logger.info('epsilon: '+ str(epsilon) + ' '+  str(t) + " lepesszam mellett, reward: " + str(sum_rewards))
	finally:
		print()
		#video_recorder.close()
		#video_recorder.enabled = False



main()
