from skimage.transform import resize
from random import randint


def preprocess_image(image):
    return resize(image, (84,84))

def argmax(Q, observation):
	#tfh. az all_action-ben az összes action kombináció benne van, pl. egy eleme: [0,0,0,0,0,0]
	max_rew = 0
	for action in all_action:
		rew = Q(picture, action)
		if rew > max_rew:
			max_action = action
			max_rew = rew
	return max_action

#a pszeudo kód kicsit átdolgozása innen kezdődik--------------------------------------


D = collections.deque(N*[0], maxlen = 5) #Experience replay dataset (st,at,rt,st+1)
Q = keras.random(10,10)#ez lesz a kezdo Neuralis Halo random ertekekkel feltoltese
Q2 = Q.deepcopy()
for epoch in range(1,10):
    env = gym.make('mario')
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    preprocessed_img = preprocess_image(observation) #Kepelofeldolgozas
    for t in range(1,T):#T itt egy jól megválasztott maximum szám, vagy lehet hogy inkább a done-ra lenne szükség?
		if E < randint(0, 100):
			action_t = env.action_spacle.sample()
		else:
			action_t = argmax(Q,preprocessed_img)
		observation, reward, done, info = env.step(action_t)
		preprocessed_img2 = preprocess_image(observation)
		D.push((preprocessed_img, action_t, reward, preprocessed_img2))
		minibatch = random.sample(D, 1) #minibatch = (s t, a t, r t, s t+1)
		if minibatch[2]:
			yj = minibatch[1] #ha a done true volt akkor csak a rewardra figyelünk
		else:
			yj = minibatch[1] + learning_factor*argmax(Q2,minibatch[3]) #a tanulási tényező egy konstans csak 0 és 1 között.
		#gradient decent step
		if t %100:
			Q2 = Q.deepcopy()
		

		
