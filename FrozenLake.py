import gym
import numpy as np 

env = gym.make("FrozenLake-v0")

Q = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = 0.8
y = 0.95
epochs = 2000 #?

rlist = []
for i in range(epochs):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    
    while j < 99:
        j+=1
        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)

print("Score over time: " + str(sum(rlist)/ epochs))
print("Final Q-Table Values")
print(Q)

# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0