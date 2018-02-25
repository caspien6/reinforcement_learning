from random import randint
import skimage.transform as ST
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import itertools, collections, gym, random, os
import numpy as np



def get_random_action(Q, fifo):
	data = np.append(fifo[0],[fifo[1],fifo[2],fifo[3]])
	max_action = Q.predict_on_batch( np.reshape(data,(1,84*84*4) ))
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
	data = np.append(fifo[0],[fifo[1],fifo[2],fifo[3]])
	return np.rint(Q.predict_on_batch( np.reshape(data,(1,84*84*4) ))).flatten()

def gradient_descent_step(Q, fifo, yj):
	data = np.append(fifo[0],[fifo[1],fifo[2],fifo[3]])
	Q.train_on_batch(np.reshape(data,(1,84*84*4)), np.reshape(yj, (1,6)) )
	return Q

D = collections.deque(maxlen = 200) #Experience replay dataset (st,at,rt,st+1)

#Initialize neural networks--

Q = Sequential()
Q.add(Dense(84*6, activation='relu', input_dim=84*84*4))
Q.add(Dense(54, activation='relu'))
Q.add(Dense(6, activation='sigmoid'))
Q.compile(optimizer='sgd',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])
if os.path.isfile('./weights.h5'):
	Q.load_weights('weights.h5')

Q_freeze = Sequential()
Q_freeze.add(Dense(84*6, activation='relu', input_dim=84*84*4))
Q_freeze.add(Dense(54, activation='relu'))
Q_freeze.add(Dense(6, activation='sigmoid'))
Q_freeze.compile(optimizer='sgd',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

Q_freeze.set_weights(Q.get_weights())


for epoch in range(1,200):
	#Start the environment

	env = gym.make('SuperMarioBros-1-1-v0')
	env.reset()
	#random action
	action = env.action_space.sample() 
	
	observation, reward, done, info = env.step(action)
	#Preprocess image
	fifo = init_fifo()
	preprocessed_img = []
	preprocessed_img.append(fifo.copy())

	learning_factor = 0.3
	t = 0
	while not done:

		if -0.5*epoch + 100.5 > randint(0, 1000):
			action_t = get_random_action(Q, fifo)
		else:
			action_t = argmax(Q,fifo)

		print("Action_t: ", action_t)

		observation, reward, done, info = env.step(action_t)
		fifo.appendleft(np.array(preprocess_image(observation), dtype=np.uint8))

		preprocessed_img.append(fifo.copy())

		D.appendleft( [preprocessed_img[t],
		action_t, reward*10, (preprocessed_img[t+1],done)] )
		minibatch = random.sample(D, 1) 
		
		if minibatch[0][3][1]:   #done igaz-e
			yj = minibatch[0][2] + np.zeros((1,6), dtype=np.uint8) #csak a rewardra figyelünk
		else:
			r = learning_factor*argmax(Q_freeze,minibatch[0][3][0])#fifo n+1
			print("Minibatc: ", minibatch[0][2])
			yj = minibatch[0][2] + r 
		
		print("yj: ", yj)

		Q = gradient_descent_step(Q, minibatch[0][0], yj)
		
		
		if (t) % 100 == 0:
			Q_freeze.set_weights(Q.get_weights())
		if (t) % 1000 == 0 or done:
			Q.save_weights('weights.h5')
		t = t + 1
