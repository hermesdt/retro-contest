import gym
import tensorflow as tf
import numpy as np
from collections import deque
from src import reward_calculator
from src import actions_builder
from retro.scripts import playback_movie
from baselines.common import atari_wrappers


class DQN():
    def __init__(self, environment, reply_memory_size=1000, gamma=0.99, epsilon=0.1, lr=0.1, steps_transfer_weights=500):
        self.env = environment
        self.state = self._reshape_state(self.env.reset())
        self.replay_memory = deque(maxlen=reply_memory_size)
        self.steps_transfers_weights = steps_transfer_weights
        self._lr = lr
        self._epsilon = epsilon
        self.gamma = gamma
        self.num_steps = 0
        self.prev_info = None
        self.ACTIONS = []
        self._action_space = None

    def _reshape_state(self, state):
        return np.array(state).reshape((1, *self.observation_space().shape))

    @property
    def lr(self):
        # reduce lr based on num_steps
        return self._lr

    @property
    def epsilon(self):
        # reduce epsilon based on num_steps
        return self._epsilon

    def actions(self):
        if self.ACTIONS != []: return self.ACTIONS

        self.ACTIONS = actions_builder.build(self.env.action_space)

        return self.ACTIONS

    def build_model(self, initializer=None):
        model = tf.keras.models.Sequential([
            #tf.keras.layers.Lambda(lambda data: tf.image.rgb_to_grayscale(data),
            #                       input_shape=self.observation_space().shape),
            # tf.layers.Conv2D(4, (2,2), strides=2, kernel_initializer=initializer),
            tf.layers.Flatten(input_shape=self.observation_space().shape),
            #tf.layers.Dense(100, kernel_initializer=initializer, activation=tf.keras.activations.relu),
            tf.layers.Dense(20, kernel_initializer=initializer, activation=tf.keras.activations.relu),
            tf.layers.Dense(10, kernel_initializer=initializer, activation=tf.keras.activations.relu),
            tf.layers.Dense(self.action_space().n, kernel_initializer=initializer)
        ])

        model.compile(tf.keras.optimizers.RMSprop(lr=self.lr), tf.keras.losses.mean_squared_error)
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

    def step(self):
        action = self.select_action(self.state)
        return self._process_step(*self._take_step(action))

    def select_action(self, state):
        probs = self._select_action_probs(state)
        return self.actions()[np.random.choice(range(len(probs)), p=probs)]

    def _select_action_probs(self, state):
        predicted = self.model.predict(state)[0]

        num_actions = self.action_space().n
        probs = np.full(num_actions, self.epsilon/num_actions)
        action = np.argmax(predicted)
        probs[action] = 1 - self.epsilon + self.epsilon / num_actions
        return probs

    def _take_step(self, action):
        return self.env.step(action)

    def _process_step(self, state, reward, done, info, target=None):
        self.state = self._reshape_state(state)
        reward = self.reward(reward, info, self.prev_info)

        if done:
            target = reward
        else:
            if target is None:
                target = self.td_target(reward, self.state)

        self.update(self.state, target)
        self.prev_info = info

        self.num_steps += 1
        if self.num_steps % self.steps_transfers_weights == 0:
            self.transfer_weights()

        return done

    def reward(self, reward, info, prev_info):
        return reward_calculator.reward(reward, info, prev_info)

    def td_target(self, reward, state):
        target_qs = reward + self.gamma * self.target_model.predict(state)[0]
        target = np.zeros(len(target_qs))
        action_idx = np.argmax(target_qs)
        target[action_idx] = target_qs[action_idx]
        return target

    def update(self, state, target):
        qs = self.model.predict(state)[0]
        qs[np.argmax(target)] = np.max(target)
        self.model.optimizer.lr = self.lr
        self.model.fit(state, qs.reshape(1, len(target)), verbose=0)

    def train_from_movie(self, movie_file):
        print("training on", movie_file)
        emulator, movie, duration, = playback_movie.load_movie(movie_file)
        env = atari_wrappers.WarpFrame(emulator)
        env = atari_wrappers.ScaledFloatFrame(env)
        emulator = atari_wrappers.FrameStack(env, 4)
        emulator.reset()

        while movie.step():
            keys = []
            for i in range(16):
                keys.append(movie.get_key(i))

            print(keys)
            keys = list(map(float, keys))
            display, reward, done, info = emulator.step(keys[:12])

            if keys[:12] not in self.ACTIONS:
                raise Exception("received combination of keys not in current sent of actions")

            action_idx = np.argmax(self.ACTIONS == keys[:12])
            target = np.zeros(self.ACTIONS.shape[0])
            target[action_idx] = 1
            self._process_step(display, reward, done, info, target)

        emulator.close()
        movie.close()