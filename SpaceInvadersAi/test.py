import gym
import numpy as np
from DqnNetwork import *
import matplotlib.pyplot  as plt
from helper import prepare_frame

env = gym.make('SpaceInvaders-v0')
INPUT_SHAPE =(80,80,1)
#INPUT_SHAPE = (70,70,1)
network = DqnNetwork(INPUT_SHAPE, env.action_space.n)

network.main_model.load_weights('models1000')
#network.main_model.load_weights('models_weights')
rewards = []

for i in range(20):
    current_state = env.reset()
    current_state = prepare_frame(current_state)
    done = False
    tmp = 0
    print(i)
    while not done:
        action = np.argmax(network.get_qvalues(current_state))
        #action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action)
        new_state = prepare_frame(new_state)
        env.render()
        tmp +=reward
        current_state = new_state
    rewards.append(tmp)

print(rewards)
avg = []
avg = np.cumsum(rewards)

for i in range(0,len(avg)):
    avg[i] = avg[i] / (i+1)


fig, axs = plt.subplots(1, 2)
fig.suptitle('Akcje podejmowane przez wytrenowangeo agenta II')
axs[0].plot(rewards)
axs[0].set_xlabel("Numer epizodu")
axs[0].set_ylabel("Wynik")

axs[1].plot(avg)
axs[1].set_xlabel("Numer epizodu")
axs[1].set_ylabel("Średni wynik")

plt.show()
