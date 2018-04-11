from random import randint
import skimage.transform as ST
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

def get_all_action():
	allactionlist = []
	allactionlist.append([0,1,0,0,0,0])
	allactionlist.append([0,0,0,1,0,0])
	allactionlist.append([1,0,0,0,0,0])
	allactionlist.append([0,0,0,0,0,0])
	allactionlist.append([0,0,1,0,0,0])
	allactionlist.append([1,1,0,0,0,0])
	allactionlist.append([1,0,0,1,0,0])
	allactionlist.append([0,0,0,0,1,0])
	allactionlist.append([0,0,0,0,0,1])
	allactionlist.append([0,0,0,1,1,0])
	allactionlist.append([0,0,0,1,0,1])
	allactionlist.append([0,1,0,0,1,0])
	allactionlist.append([0,1,0,0,0,1])
	return allactionlist

def get_random_action(Q, fifo):
	data = np.concatenate( (fifo[3],fifo[2],fifo[1],fifo[0]), axis=3)
	action_predictions = Q.predict_on_batch( data).flatten()
	index = action_predictions.argmax(axis=0)
	global all_action
	max_action = list(all_action[index])

	random_index = random.randint(0,ACTION_SET_LENGTH-1)
	
	global all_action
	while max_action == list(all_action[random_index]):
		random_index = random.randint(0,ACTION_SET_LENGTH-1)
	return all_action[random_index]

def init_fifo():
	fifo = collections.deque(maxlen=4)
	return fifo

def preprocess_image(image):

	image = image[40:210]
	observation = ST.resize(image, (84,84), preserve_range=True)
	for rowi in range(observation.shape[0]):
		for pixeli in range(observation.shape[1]):
			element = observation[rowi,pixeli]

			observation[rowi,pixeli] = element[0]*0.299 + 0.587*element[1] + element[2]*0.114

	#plt.imshow(observation)
	#plt.show()

	prec = np.expand_dims(observation[:,:,0], axis=2)
	prec = np.expand_dims(prec, axis=0)
	prec = prec.astype(np.uint8)
	return prec

def argmax(Q,fifo):
	data = np.concatenate( (fifo[3],fifo[2],fifo[1],fifo[0]), axis=3)
	action_predictions = Q.predict_on_batch( data.astype(np.float32)).flatten()
	index = action_predictions.argmax(axis=0)
	
	# print('index: ', index)
	#print(action_predictions)

	return list(all_action[index])

def get_epsilon(x, maxrange):
	percent = 0.95
	threshold = maxrange - 500
	if threshold < x: x = threshold

	return percent - (x*percent / threshold * 1.12 )

def init_network():
	model = Sequential()
	model.add(keras.layers.Conv2D(32, (8,8), data_format='channels_last',
	 strides=(4, 4), activation='relu', input_shape=(84,84,4) ) )

	model.add(keras.layers.Conv2D(64, (4,4), strides=(2, 2), activation='relu'))
	model.add(keras.layers.Conv2D(64, (3,3), strides=(1, 1), activation='relu'))
	model.add(keras.layers.Flatten())
	model.add(Dense(units=512, activation='relu'))
	model.add(Dense(units=ACTION_SET_LENGTH, activation='relu'))


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

def get_yj(minibatch, Q_freeze, learning_factor):
	y = np.zeros([BATCH_SIZE,ACTION_SET_LENGTH],dtype=np.uint8)


	for i in range(BATCH_SIZE):
		if minibatch[i][3][1]:   #done igaz-e
			y[i] = minibatch[i][2] + np.zeros((1,ACTION_SET_LENGTH),dtype=np.float32)
		else:
			action_predictions = Q_freeze.predict_on_batch(minibatch[i][3][0].astype(np.float32)).flatten()
			#print("Before: ",action_predictions )
			index = action_predictions.argmax()

			action_predictions[index] = minibatch[i][2] + learning_factor*action_predictions[index]
			#print("After: ",action_predictions)
			y[i] = action_predictions[:]
	
	return y

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
		Q_freeze.set_weights(Q.get_weights())

		global all_action
		all_action = get_all_action()

		maxrange=2000

		for epoch in range(1,maxrange):
			#Start the environment
			env = gym.make('SuperMarioBros-1-1-v0')
			#env = Monitor(env, './records', lambda episode_id: episode_id%10==0, force=True)
			env.reset()

			#Video recorder init
			#video_recorder = VideoRecorder(env, './records/els'+str(epoch)+'.mp4', enabled=True)
			

			# if epoch%10==1:
			# 	video_recorder.enabled=True
			# else:
			# 	video_recorder.enabled=False

			#random action
			action = env.action_space.sample() 
			observation, reward, done, info = env.step(action)

			#Preprocess image
			fifo = init_fifo()
			fifo.appendleft(preprocess_image(observation))
			fifo.appendleft(preprocess_image(observation))
			fifo.appendleft(preprocess_image(observation))
			fifo.appendleft(preprocess_image(observation))
			preprocessed_img = []
			preprocessed_img.append(fifo.copy())


			epsilon = get_epsilon(float(epoch),float(maxrange))
			print('epsilon: ', epsilon)
			learning_factor = 0.000001
			sum_rewards = 0
			t = 0
			while not done:
				#video_recorder.capture_frame()
				
				if epsilon >= random.uniform(0,1):
					print("random:")
					action_t = get_random_action(Q, fifo)
				else:
					action_t = argmax(Q,fifo)
				print(action_t)


				observation, reward, done, info = env.step(action_t)

				if reward > 0: reward = 1
				else: reward = -1
				sum_rewards += reward
				
				
				fifo.appendleft(preprocess_image(observation))
				preprocessed_img.append(fifo.copy())
				pi_t = np.concatenate((preprocessed_img[t][3],preprocessed_img[t][2],preprocessed_img[t][1],preprocessed_img[t][0]), axis=3)
				pi_tt = np.concatenate((preprocessed_img[t+1][3],preprocessed_img[t+1][2],preprocessed_img[t+1][1],preprocessed_img[t+1][0]), axis=3)
				
				D.appendleft( [pi_t,action_t, reward, (pi_tt,done)] )
				
				if t >= BATCH_SIZE and t%2 == 0:

					minibatch = random.sample(D, BATCH_SIZE)
					yj = get_yj(minibatch, Q_freeze, learning_factor)
					

					#gradient descent !! TEST----------------------------------

					for i in range(BATCH_SIZE-1):
						if i != 0:
							data = np.append(data,minibatch[i+1][0], axis=0)
						else:
							data = np.append(minibatch[0][0],minibatch[1][0], axis=0)
					#data = data.astype(np.float32)
					#yj = yj.astype(np.float32)
					#print(data[1])
					loss = Q.train_on_batch(data,  yj )
					#print('loss: ', loss)
					#Q = gradient_descent_step(Q, minibatch[0][0], error )-----
				
				
				if (t) % 500 == 0:
					Q_freeze.set_weights(Q.get_weights())
				t = t + 1
			if epoch % 4 == 0:
				Q.save_weights('weights' + str(epoch/4) + '.h5')
			logger.info('epsilon: '+ str(epsilon) + ' '+  str(t) + " lepesszam mellett, reward: " + str(sum_rewards))
	finally:
		print()
		#video_recorder.close()
		#video_recorder.enabled = False



main()
