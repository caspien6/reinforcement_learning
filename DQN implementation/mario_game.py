import gym
import keyboard #Using module keyboard
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def preprocess_image(observation, last_observation):
	processed_observation = np.maximum(observation, last_observation)[70:220]
	processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
	plt.imshow(processed_observation)
	plt.show()
	return np.reshape(processed_observation, (84, 84, 1))

for i in range(0,10):

	
	env=gym.make('SuperMarioBros-1-1-v0')

	observation = env.reset()
	done = False
	t = 0
	reward = 0.0
	while not done:

			# if keyboard.is_pressed('w'):#if key 'q' is pressed 
			# 	print('w')
			# 	observation, reward, _, info = env.step([1,0,0,0,0,0])
				
			# 	break#finishing the loop
			# elif keyboard.is_pressed('s'):#if key 'q' is pressed 
			# 	print('s')
			# 	observation, reward, _, info = env.step([0,0,1,0,0,0])
				
			# 	break#finishing the loop
			# elif keyboard.is_pressed('a'):#if key 'q' is pressed 
			# 	print('a')
			# 	observation, reward, _, info = env.step([0,1,0,0,0,0])
				
			# 	break#finishing the loop
			# elif keyboard.is_pressed('d'):#if key 'q' is pressed
			# 	print('d')
			# 	observation, reward, _, info = env.step([0,0,0,1,0,0])
				
			# 	break#finishing the loop
			# else:
			# 	print('else')
			# 	observation, reward, _, info = env.step([0,0,0,0,0,0])
			# 	break#finshing the loop
		observation, reward, done, info = env.step([0,1,0,0,0,0])
		preprocess_image(observation, observation)
		print(reward)