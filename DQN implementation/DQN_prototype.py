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


def get_random_action(Q, fifo):
	data = np.concatenate( (fifo[0],fifo[1],fifo[2],fifo[3]), axis=3)
	max_action = Q.predict_on_batch( data)
	rnd_action = [random.randint(0,1),	random.randint(0,1),	random.randint(0,1),	random.randint(0,1),	random.randint(0,1),	random.randint(0,1)]
	while [i for i,j in zip(rnd_action, max_action.tolist()) if i == j]:
		rnd_action = [random.randint(0,1),	random.randint(0,1),	random.randint(0,1),	random.randint(0,1),	random.randint(0,1),	random.randint(0,1)]

	return rnd_action

def init_fifo():
	fifo = collections.deque(maxlen=4)
	one = np.ones((84,84),dtype=np.uint8)
	two = np.ones((84,84),dtype=np.uint8)
	three = np.ones((84,84),dtype=np.uint8)
	four = np.ones((84,84),dtype=np.uint8)

	one = np.expand_dims(one, axis=2)
	two = np.expand_dims(two, axis=2)
	three = np.expand_dims(three, axis=2)
	four = np.expand_dims(four, axis=2)
	one = np.expand_dims(one, axis=0)
	two = np.expand_dims(two, axis=0)
	three = np.expand_dims(three, axis=0)
	four = np.expand_dims(four, axis=0)

	fifo.appendleft(one)
	fifo.appendleft(two)
	fifo.appendleft(three)
	fifo.appendleft(four)
	return fifo

def preprocess_image(image):
	image = image[40:210]
	observation = ST.resize(image, (84,84))
	for rowi in range(observation.shape[0]):
		for pixeli in range(observation.shape[1]):
			element = observation[rowi,pixeli]
			observation[rowi,pixeli] = element[0]*0.299 + 0.587*element[1] + element[2]*0.114
	return observation[:,:,0]

def argmax(Q,fifo):
	data = np.concatenate((fifo[0],fifo[1],fifo[2],fifo[3]), axis=3)
	return np.rint(Q.predict_on_batch( data)).flatten()

def gradient_descent_step(Q, fifo, yj):
	data = np.concatenate((fifo[0],fifo[1],fifo[2],fifo[3]), axis=3)
	Q.train_on_batch(data, np.reshape(yj, (1,6)) )
	return Q

def get_epsilon(x, maxrange):
	return (0.01-0.1)/(maxrange-1.0)*x + 0.1 + (-1.0*(0.01-0.1)/(maxrange-1.0))

def init_network():
	model = Sequential()
	model.add(keras.layers.Conv2D(32, (8,8), strides=(4, 4), activation='relu', input_shape=(84,84,4)))
	model.add(keras.layers.Conv2D(64, (4,4), strides=(2, 2), activation='relu'))
	model.add(keras.layers.Conv2D(64, (3,3), strides=(1, 1), activation='relu'))
	model.add(keras.layers.Flatten())
	model.add(Dense(units=512, activation='relu'))
	model.add(Dense(units=6, activation='relu'))

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
	if minibatch[0][3][1]:   #done igaz-e
		yj = minibatch[0][2] + np.zeros((1,6), dtype=np.uint8) #csak a rewardra figyelÃ¼nk
	else:
		s_next=argmax(Q_freeze,minibatch[0][3][0])
		r = learning_factor*s_next#fifo n+1
		#print("Minibatc: ", minibatch[0][2])
		yj = minibatch[0][2] + r
	#print(yj)
	return yj

def main():
	try:
		display = Display(visible=0, size=(1400,900))
		display.start()
		D = collections.deque(maxlen = 500) #Experience replay dataset (st,at,rt,st+1)

		#Initialize neural networks--

		Q = init_network()
		logger = init_logger('./records/els.log')

		if sys.argv[-1] == 'y' and os.path.isfile('./weights.h5'):
			#print(sys.argv)
			Q.load_weights('weights.h5')

		Q_freeze = init_network()
		Q_freeze.set_weights(Q.get_weights())

		maxrange=200

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
			preprocessed_img = []
			preprocessed_img.append(fifo.copy())


			epsilon = get_epsilon(float(epoch+1.0),float(maxrange))
			learning_factor = 0.5
			sum_rewards = 0
			t = 0
			while not done:
				#video_recorder.capture_frame()
				
				if epsilon >= random.uniform(0,1):
					#print("Random:")
					action_t = get_random_action(Q, fifo)
				else:
					action_t = argmax(Q,fifo)

				#print("Action_t: ", action_t)

				observation, reward, done, info = env.step(action_t)

				if reward > 0: reward = 1
				elif reward < 0: reward = -1
				sum_rewards += reward

				preprocessed_img.append(fifo.copy())

				D.appendleft( [preprocessed_img[t],action_t, reward, (preprocessed_img[t+1],done)] )
				minibatch = random.sample(D, 1)
				
				yj = get_yj(minibatch, Q_freeze, learning_factor)
							
				#Backpropagation

				Q = gradient_descent_step(Q, minibatch[0][0], yj)
				
				
				if (t) % 100 == 0:
					Q_freeze.set_weights(Q.get_weights())
				if (t) % 100 == 0 or done:
					Q.save_weights('weights.h5')
				t = t + 1
			logger.info(str(t) + " lÃ©pÃ©sszÃ¡m mellett, reward: " + str(sum_rewards))
	finally:
		print()
		#video_recorder.close()
		#video_recorder.enabled = False



main()

