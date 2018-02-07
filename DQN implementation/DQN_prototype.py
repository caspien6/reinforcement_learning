D = [x for x in range(1,10) ] #Experience replay dataset (st,at,rt,st+1)
Q = keras.random(10,10)#ez lesz a kezdo Neuralis Halo random ertekekkel feltoltese
for epoch in range(1,10):
    env = gym.make('mario')
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    #itt meg hianyzik a kepelofeldolgozas f[0] = f(observation)
    for time in range(1,1000):
		#ha E < 0.4 akkor random action egyebkent
		#a[time] = maximumkeresés Q(f(s[time]), a[0...N])
		x[time+1], reward, done, info = env.step(a[time])
		
		# az alabbi reszt nem teljesen ertem, ez azt akarja jelenteni, hogy
		#a kovetkezo allapot az elozo allapot + action es hozzaadodik a
		#mostani kep?
		s[time+1] = s[time],a[time], x[time+1] 
		
		D.push((f[time], a[time], r[time], f[time+1]) #az ujrafelhasznalas celjabol felvesszuk a D-be
		j = int.random(1,100)
		rj = D[j][2] #a 2es index a reward lesz
		

		
