import gym
import tensorflow as tf
import numpy as np
from collections import deque
from src import reward_calculator
from src import actions_builder
from retro.scripts import playback_movie
import random
from src.env_creator import wrap_environment
from src.coords_store import CoordsStore
import os


class DQN():
    #actions=actions_builder.build(action_space)
    def __init__(self, observation_space, actions, load_weights=True, gamma=0.995, epsilon=0.2, lr=0.001,
                 keras_verbose=0):
        self.keras_verbose = keras_verbose
        self.ACTIONS = actions
        self.observation_space = observation_space
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))

        self.model = self.build_model(initializer=tf.keras.initializers.Zeros())
        if load_weights and os.path.exists("weights/alvaro_dqn_model.h5"):
            self.model.load_weights("weights/alvaro_dqn_model.h5")

    def _reshape_state(self, state):
        return np.array(state).reshape((1, *self.observation_space.shape))

    def reset(self, env):
        self.prev_action = None
        self.state = env.reset()
        self.action = self.ACTIONS[0]

    def build_model(self, initializer=None):
        frames_in = tf.keras.Input(shape=self.observation_space.shape, name="frames_in")
        # extras_in = tf.keras.Input(shape=(1,), name="extras_in")

        #x = tf.layers.Conv2D(32, (7, 7), input_shape=self.observation_space.shape)(frames_in)
        #x = tf.layers.Conv2D(32, (7, 7))(x)
        #x = tf.layers.AveragePooling2D((7, 7), 7)(x)
        x = tf.layers.Flatten(input_shape=self.observation_space.shape)(frames_in)
        # x = tf.keras.layers.Concatenate()([x, extras_in])
        x = tf.layers.Dense(60, kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.layers.Dense(60, kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.layers.Dense(30, kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.layers.Dense(30, kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.layers.Dense(30, kernel_initializer=initializer)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.layers.Dense(self.action_space.n, kernel_initializer=initializer)(x)

        model = tf.keras.models.Model(inputs=[frames_in], outputs=[x])
        model.summary()

        model.compile(tf.keras.optimizers.Adam(lr=self.lr), tf.keras.losses.mean_squared_error)
        return model

    def step(self, env, human_action=None):
        if human_action is not None:
            self.action = human_action

        new_state, reward, done, info = env.step(self.action)
        new_action = self.select_action(new_state)

        action_state = (self.state, self.action, new_state, reward, done, info, new_action)

        self.state = new_state
        self.action = new_action

        return action_state

    def select_action(self, state):
        probs = self._select_action_probs(state)
        return self.ACTIONS[np.random.choice(range(len(probs)), p=probs)]

    def _select_action_probs(self, state):
        predicted = self.model.predict([self._reshape_state(state)])[0]

        num_actions = self.action_space.n
        probs = np.full(num_actions, self.epsilon/num_actions)
        action = np.argmax(predicted)
        probs[action] = 1 - self.epsilon + self.epsilon / num_actions
        return probs

    def learn_from_memory(self, memory):
        states, actions, new_states, rewards, dones, new_actions = [], [], [], [], [], []
        ret = 0
        for state, action, new_state, reward, done, info, new_action in memory[::-1]:
            ret = reward = reward + ret*self.gamma
            states.append(state)
            actions.append(np.where((self.ACTIONS == np.array(action)).all(axis=1))[0][0])
            new_states.append(new_state)
            rewards.append(self.reward(reward, done, info))
            dones.append(done)
            new_actions.append(np.where((self.ACTIONS == np.array(new_action)).all(axis=1))[0][0])

        states = np.array(states)
        actions = np.array(actions)
        new_states = np.array(new_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        new_actions = np.array(new_actions, dtype=np.int)

        target_prediction = self.model.predict([new_states])
        target_prediction[
            np.arange(len(target_prediction)), new_actions] +=\
            np.array(rewards)

        predictions = self.model.predict([states])
        predictions[
            np.arange(len(predictions)), actions] = target_prediction[np.arange(len(predictions)), new_actions]

        self.model.fit([states], predictions, batch_size=32, shuffle=True, verbose=self.keras_verbose, epochs=1)

    def reward(self, reward, done, info):
        return reward_calculator.reward(reward, done, info)


