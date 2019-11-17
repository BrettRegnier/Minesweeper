import numpy as np
import os

from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D, Activation, Flatten, MaxPool2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import MaxBoltzmannQPolicy, LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

import Globals


class Broom:
    def __init__(self, env, name='Broom.h5', load=True, memory=50000):
        self._env = env
        self._memory = memory
        self._name = name
        self._model = None
        self._dqn = None
        self._trained = False
        self.CreateModel()

        if load:
            self.Load(name)

    def CreateModel(self):
        nb_actions = self._env._actionSpace.n
        nb_inputs = self._env._observationSpace.shape
        memory = SequentialMemory(limit=self._memory, window_length=1)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=100000)
        self._model = Sequential()
        self._model.add(Flatten(input_shape=nb_inputs))
        self._model.add(Dense(units=1024, activation='relu'))
        self._model.add(Dense(units=1024, activation='relu'))
        self._model.add(Dense(units=1024, activation='relu'))
        self._model.add(Dense(units=1024, activation='relu'))
        self._model.add(Dense(units=nb_actions, activation='linear'))

        self._dqn = DQNAgent(model=self._model, nb_actions=nb_actions, memory=memory,
                             nb_steps_warmup=10000, target_model_update=1, policy=policy)
        self._dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def Load(self, name):
        if not (".h5" in name):
            name += ".h5"

        if os.path.isfile(name):
            self._model.load_weights(name)
            self._trained = True
        else:
            print("No save found with the name", name)

    def Save(self, name):
        if not (".h5" in name):
            name += ".h5"
        self._model.save_weights(name)

    def Train(self, episodes=10):

        history = self._dqn.fit(
            self._env, nb_episodes=episodes, visualize=False, verbose=2)

        # # Plot training points
        # plt.plot(history.history['nb_episode_steps'], linewidth=1.0)
        # plt.title('Boat Fitness')
        # plt.ylabel('Number of steps per episode')
        # plt.xlabel('Episode')
        # plt.show()

        ans = ""
        while (ans != "Y" and ans != "N"):
            print("Do you want to save this model? Y\\N")
            ans = input(">").upper()

        # Save model
        if ans == "Y":
            self.Save(self._name)

    def Test(self, episodes=1):
        if (self._trained == True):
            pred = self._dqn.test(self._env, episodes)
        else:
            print("Please train a model")