import gym
import keyboard #Using module keyboard




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
		print(reward)