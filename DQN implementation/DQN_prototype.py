from random import randint
import skimage.transform as ST
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import itertools, collections, gym, random
import numpy as np


def init_fifo():
	fifo = collections.deque(maxlen=4)
	one = np.ones((84,84),dtype=np.uint8)
	two = np.ones((84,84),dtype=np.uint8)
	three = np.ones((84,84),dtype=np.uint8)
	four = np.ones((84,84),dtype=np.uint8)
	fifo.appendleft(one)
	fifo.appendleft(two)
	fifo.appendleft(three)
	fifo.appendleft(four)
	return fifo

def preprocess_image(image):
	image = image[40:210]
	observation = ST.resize(image, (84,84))
	np.array(observation, dtype=np.uint8)
	for rowi in range(observation.shape[0]):
		for pixeli in range(observation.shape[1]):
			element = observation[rowi,pixeli]
			observation[rowi,pixeli] = element[0]*0.299 + 0.587*element[1] + element[2]*0.114
	return observation[:,:,0]

def argmax(Q,fifo, observation):
	fifo.appendleft(np.array(observation, dtype=np.uint8))
	data = np.append(fifo[0],[fifo[1],fifo[2],fifo[3]])
	return Q.predict_on_batch( np.reshape(data,(1,84*84*4) ))

D = collections.deque(maxlen = 20) #Experience replay dataset (st,at,rt,st+1)

#Initialize neural networks--

Q = Sequential()
Q.add(Dense(84*4, activation='relu', input_dim=84*84*4))
Q.add(Dense(6, activation='sigmoid'))
Q.compile(optimizer='sgd',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

Q_freeze = Sequential()
Q_freeze.add(Dense(84*4, activation='relu', input_dim=84*84*4))
Q_freeze.add(Dense(6, activation='sigmoid'))
Q_freeze.compile(optimizer='sgd',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

Q_freeze.set_weights(Q.get_weights())
fifo = init_fifo()
for epoch in range(1,10):
	#Start the environment

	env = gym.make('SuperMarioBros-1-1-v0')
	env.reset()
	#random action
	action = env.action_space.sample() 
	
	observation, reward, done, info = env.step(action)
	#Preprocess image
	preprocessed_img = []
	preprocessed_img.append(preprocess_image(observation))
	T = 2000
	learning_factor = 0.1
	for t in range(0,T):#T itt egy jól megválasztott maximum szám, vagy lehet hogy inkább a done-ra lenne szükség?

		if 1/(t+1) < randint(0, 100):
			action_t = env.action_space.sample()
		else:
			action_t = np.rint(argmax(Q,fifo,preprocessed_img[t]))
			action_t = action_t.flatten()


		print(action_t)

		observation, reward, done, info = env.step(action_t)

		preprocessed_img.append(preprocess_image(observation))

		D.appendleft( [preprocessed_img[t],
		action_t, reward, (preprocessed_img[t+1],done)] )
		minibatch = random.sample(D, 1) #minibatch = (s t, a t, r t, (s t+1, done))
		
		if minibatch[0][3][1]:   #done igaz-e
			yj = minibatch[0][2] #csak a rewardra figyelünk
		else:
			rr = argmax(Q_freeze, fifo,minibatch[0][3][0])
			print(rr)
			print(type(rr))
			yj = minibatch[0][2] + learning_factor*argmax(Q_freeze, fifo,minibatch[0][3][0]) #a tanulási tényező egy konstans csak 0 és 1 között.
		
		data = np.append(minibatch[0][0],[fifo[1],fifo[2],fifo[3]])
		Q.train_on_batch(np.reshape(data,(1,84*84*4)), yj)
		
		if (t) % 100 == 0:
			Q_freeze.set_weights(Q.get_weights())
