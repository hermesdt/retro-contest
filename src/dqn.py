import gym
import tensorflow as tf
import numpy as np
from collections import deque
from src import reward_calculator
from src import actions_builder
from retro.scripts import playback_movie
from baselines.common import atari_wrappers
import random


class DQN():
    def __init__(self, environment, reply_memory_size=10000, gamma=1, epsilon=0.2, lr=0.0001,
                 steps_transfer_weights=1000,
                 steps_learn_from_memory=500,
                 replay_actions=500):
        self.env = environment
        self.replay_memory = deque(maxlen=reply_memory_size)
        self.steps_transfers_weights = steps_transfer_weights
        self._lr = lr
        self._epsilon = epsilon
        self.gamma = gamma
        self.num_steps = 0
        self.prev_info = None
        self.episode_steps = 0
        self.ACTIONS = []
        self._action_space = None
        self.replay_actions = replay_actions
        self.steps_learn_from_memory = steps_learn_from_memory
        self.first_x, self.last_x = None, None
        self.max_x = 0

    def _reshape_state(self, state):
        return np.array(state).reshape((1, *self.observation_space().shape))

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env
        self.episode_steps = 0
        self.state = env.reset()

    @property
    def lr(self):
        # reduce lr based on num_steps
        return self._lr

    @property
    def epsilon(self):
        # reduce epsilon based on num_steps
        return self._epsilon
    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = max(0.01, value)

    def actions(self):
        if self.ACTIONS != []: return self.ACTIONS

        self.ACTIONS = actions_builder.build(self.env.action_space)

        return self.ACTIONS

    def build_model(self, initializer=None):
        model = tf.keras.models.Sequential([
            #tf.layers.Conv2D(32, (5, 5), input_shape=self.observation_space().shape),
            # tf.layers.Conv2D(32, (3, 3)),
            #tf.layers.AveragePooling2D((3, 3), 3),
            # tf.layers.Flatten(),
            # tf.layers.Dense(20, kernel_initializer=initializer, activation=tf.keras.activations.relu),
            # tf.layers.Dense(10, kernel_initializer=initializer, activation=tf.keras.activations.relu),
            # tf.keras.layers.Lambda(lambda data: tf.image.rgb_to_grayscale(data), ),
            tf.layers.Flatten(input_shape=self.observation_space().shape),
            tf.layers.Dense(40, kernel_initializer=initializer, activation=tf.keras.activations.relu),
            tf.layers.Dense(30, kernel_initializer=initializer, activation=tf.keras.activations.relu),
            tf.layers.Dense(20, kernel_initializer=initializer, activation=tf.keras.activations.relu),
            tf.layers.Dense(10, kernel_initializer=initializer, activation=tf.keras.activations.relu),
            tf.layers.Dense(self.action_space().n, kernel_initializer=initializer)
        ])
        model.summary()

        model.compile(tf.keras.optimizers.Adam(lr=self.lr), tf.keras.losses.mean_squared_error)
        return model

    def build_target_model(self):
        return self.build_model(initializer=tf.keras.initializers.Zeros())

    def transfer_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def setup_models(self):
        self.model = self.build_model()
        self.target_model = self.build_target_model()
        return self

    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        if self._action_space: return self._action_space

        self._action_space = gym.spaces.Discrete(len(self.actions()))
        return self._action_space

    def step(self, human_action=None):
        if human_action is None:
            action = self.select_action(self.state)
        else:
            action = human_action

        state, reward, done, info = self.env.step(action)


        self.max_x = max(info["x"], self.max_x)

        if self.first_x is None:
            self.first_x = info["x"]

        if human_action is None and self.episode_steps > 0 and self.episode_steps % 100 == 0:
            self.last_x = info["x"]

            if self.first_x and self.last_x and abs(self.first_x - self.last_x) < 20:
                self.first_x = self.last_x = None
                done = True
                reward = -1
            else:
                self.first_x = self.last_x

        self.remember(self.state, action, state, reward, done, info, self.prev_info)

        self.state = state
        self.prev_info = info

        self.num_steps += 1
        self.episode_steps += 1

        if self.num_steps % self.steps_learn_from_memory == 0:
            self.learn_from_memory()

        if self.num_steps % self.steps_transfers_weights == 0:
            self.transfer_weights()

        return done

    def remember(self, old_state, action, state, reward, done, info, prev_info):
        self.replay_memory.append((old_state, action, state, reward, done, info, prev_info))

    def select_action(self, state):
        probs = self._select_action_probs(state)
        return self.actions()[np.random.choice(range(len(probs)), p=probs)]

    def _select_action_probs(self, state):
        predicted = self.model.predict(self._reshape_state(state))[0]

        num_actions = self.action_space().n
        probs = np.full(num_actions, self.epsilon/num_actions)
        action = np.argmax(predicted)
        probs[action] = 1 - self.epsilon + self.epsilon / num_actions
        return probs

    def learn_from_memory(self):
        mems = random.sample(self.replay_memory, min(len(self.replay_memory), self.replay_actions))

        old_states, actions, states, rewards, dones = [], [], [], [], []
        for old_state, action, state, reward, done, info, prev_info in mems:
            old_states.append(old_state)
            actions.append(action)
            states.append(state)
            rewards.append(self.reward(reward, done, info, prev_info))
            dones.append(done)

        old_states = np.array(old_states)
        actions = np.array(actions)
        states = np.array(states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        target_prediction = self.target_model.predict(states)
        td_targets = np.zeros(target_prediction.shape)
        td_targets[
            np.arange(len(target_prediction)), np.argmax(target_prediction, axis=1)] +=\
            np.array(rewards) + self.gamma

        #dones = np.full(td_targets.shape[0], False)
        #dones[0] = True

        if np.sum(dones) > 0:
            td_targets[dones, np.argmax(target_prediction[dones], axis=1)] = rewards[dones]

        predictions = self.model.predict(old_states)
        predictions[np.arange(len(predictions)), np.argmax(td_targets, axis=1)] = np.max(td_targets, axis=1)

        self.model.fit(old_states, predictions, batch_size=32, shuffle=True, verbose=1)

    def reward(self, reward, done, info, prev_info):
        return reward_calculator.reward(reward, done, info, prev_info)

    def train_from_movie(self, movie_file):
        print("training on", movie_file)
        env, movie, duration, = playback_movie.load_movie(movie_file)
        env = atari_wrappers.WarpFrame(env)
        env = atari_wrappers.ScaledFloatFrame(env)
        env = atari_wrappers.FrameStack(env, 4)
        self.env = env
        self.prev_info = None

        while movie.step():
            keys = []
            for i in range(16):
                keys.append(movie.get_key(i))

            keys = list(map(float, keys))[:12]
            actions = np.where((self.actions() == np.array(keys)).all(axis=1))
            if len(actions) != 1:
                raise ValueError("keys array not present in actions", keys)
            else:
                action = actions[0]

            self.step(action)

        env.close()
        movie.close()
