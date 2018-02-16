from random import randint
import skimage.transform as ST
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


def preprocess_image(image):
	image = image[40:210]
	return ST.resize(image, (84,84))

def argmax(Q, observation):
	#tfh. az all_action-ben az összes action kombináció benne van, pl. egy eleme: [0,0,0,0,0,0]
	max_rew = 0
	for action in all_action:
		rew = Q.predict_on_batch((observation, action))
		if rew[0] > max_rew:
			max_action = action
			max_rew = rew[0]
	return max_action


D = collections.deque(maxlen = 20) #Experience replay dataset (st,at,rt,st+1)

#Initialize neural networks--

Q = Sequential()
Q.add(Dense(32, activation='relu', input_dim=84))
Q.add(Dense(1, activation='sigmoid'))
Q.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

Q_freeze = Sequential()
Q_freeze.add(Dense(32, activation='relu', input_dim=84))
Q_freeze.add(Dense(1, activation='sigmoid'))
Q_freeze.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

Q_freeze.set_weights(Q.get_weights)

for epoch in range(1,10):
	#Start the environment

	env = gym.make('SuperMarioBros-1-1-v0')

	#random action
	action = env.action_space.sample() 
	
	observation, reward, done, info = env.step(action)
	#Preprocess image
	preprocessed_img = []
	preprocessed_img.append(preprocess_image(observation))
	T = 2000
	for t in range(0,T):#T itt egy jól megválasztott maximum szám, vagy lehet hogy inkább a done-ra lenne szükség?

		if 98 < randint(0, 100):
			action_t = env.action_space.sample()
		else:
			action_t = argmax(Q,preprocessed_img[t])

		observation, reward, done, info = env.step(action_t)

		preprocessed_img.append(preprocess_image(observation))

		D.push( (preprocessed_img[t],
		action_t, reward, (preprocessed_img[t+1],done)) )
		
		minibatch = random.sample(D, 1) #minibatch = (s t, a t, r t, (s t+1, done))
		if minibatch[3][1]:   #done igaz-e
			yj = minibatch[2] #csak a rewardra figyelünk
		else:
			yj = minibatch[2] + learning_factor*argmax(Q_freeze,minibatch[3][0]) #a tanulási tényező egy konstans csak 0 és 1 között.
		
		Q.train_on_batch(minibatch[0:2], yj)
		
		if t % 100:
			Q_freeze.set_weights(Q.get_weights)
