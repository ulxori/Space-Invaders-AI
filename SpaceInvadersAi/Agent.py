import gym
import numpy as np
from tqdm import tqdm
from ReplayMemory import *
from DqnNetwork import *
from helper import *

ENVIRONMENT = "SpaceInvaders-v0"
NUMBER_OF_EPISODES = 1_000
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 20_000
EPSILON_MIN_VALUE = 0.1
UPDATE_NETWORK_EVERY = 10
INPUT_SHAPE = (80,80,1)
MIN_REPLAY_MEMORY_SIZE = 1000
CURRENT_STATE = 0
FUTURE_STATE = 3
MINIBATCH_SIZE = 6

class Agent:

    def __init__(self):

        self.number_of_episodes = NUMBER_OF_EPISODES
        self.discount = DISCOUNT
        self.epsilon = 1.0
        self.epsilon_decay = 1.0 / NUMBER_OF_EPISODES
        self.epsilon_min_value = EPSILON_MIN_VALUE
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.best_reward = 0.0
        self.update_network_every = UPDATE_NETWORK_EVERY
        self.env = gym.make(ENVIRONMENT)
        self.number_of_actions = self.env.action_space.n
        self.input_shape = INPUT_SHAPE
        self.dqn_network = DqnNetwork(self.input_shape, self.number_of_actions)
        self.min_replay_memory_size = MIN_REPLAY_MEMORY_SIZE
        self.minibatch_size = MINIBATCH_SIZE

        self.simulate()

    def train(self, is_done):

        if self.replay_memory.get_current_size() < self.min_replay_memory_size:
            return

        minibatch = self.replay_memory.get_random_sample(self.minibatch_size)

        current_states = np.array([transition[CURRENT_STATE] for transition in minibatch])
        future_states = np.array([transition[FUTURE_STATE] for transition in minibatch])

        current_qvalues = self.dqn_network.main_model.predict(current_states)
        future_qvalues = self.dqn_network.target_model.predict(future_states)

        features, labels = [], []

        for i, (current_state, action, reward, future_state, done) in enumerate(minibatch):

            if done:
                q = reward
            else:
                q = reward + self.discount * np.max(future_qvalues[i])

            q_prob = current_qvalues[i]
            q_prob[action] = q
            features.append(current_state)
            labels.append(q_prob)

        self.dqn_network.main_model.fit(np.array(features), np.array(labels), batch_size=self.minibatch_size, verbose=0)

    def update_epsilon(self):

        if self.epsilon > self.epsilon_min_value:
            self.epsilon -= self.epsilon_decay

    def make_action(self, state):

        if np.random.random() > self.epsilon:
            action = np.argmax(self.dqn_network.get_qvalues(state))
        else:
            action = self.env.action_space.sample()
        return action


    def simulate(self):

        for episode in tqdm(range(1, self.number_of_episodes + 1), ascii=True, unit='Episodes'):

            current_reward = 0
            current_state = self.env.reset()
            current_state = prepare_frame(current_state)
            done = False
            while not done:

                action = self.make_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                current_reward += reward
                new_state = prepare_frame(new_state)
                transition = (current_state, action, reward, new_state, done)
                self.replay_memory.add(transition)
                self.train(done)
                current_state = new_state

            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self.dqn_network.main_model.save_weights('best_weights')

            self.update_epsilon()

            if episode % self.update_network_every==0:
                self.dqn_network.update_target_network()

        self.dqn_network.main_model.save_weights('trained_model_weights')