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


class DQN():
    def __init__(self, environment, reply_memory_size=10000, gamma=0.95, epsilon=0.2, lr=0.00001,
                 steps_transfer_weights=1000,
                 steps_learn_from_memory=500,
                 replay_actions=500):
        self.ACTIONS = []
        self.env = environment
        self.replay_memory = deque(maxlen=reply_memory_size)
        self.episode_memory = []
        self.steps_transfers_weights = steps_transfer_weights
        self._lr = lr
        self._epsilon = epsilon
        self.gamma = gamma
        self.num_steps = 0
        self.prev_info = None
        self.episode_steps = 0
        self._action_space = None
        self.replay_actions = replay_actions
        self.steps_learn_from_memory = steps_learn_from_memory
        self.first_x, self.last_x = None, None
        self.max_x = 0
        self.prev_action = None
        self.stucks_store = CoordsStore()

    def _reshape_state(self, state):
        return np.array(state).reshape((1, *self.observation_space().shape))

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env
        self.episode_steps = 0
        self.prev_info = None
        self.episode_memory = []
        self.max_x = 0
        self.state = env.reset()
        self.action = random.choice(self.actions())

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
        self._epsilon = max(0.1, value)

    def actions(self):
        if self.ACTIONS != []: return self.ACTIONS

        self.ACTIONS = actions_builder.build(self.env.action_space)

        return self.ACTIONS

    def build_model(self, initializer=None):
        frames_in = tf.keras.Input(shape=self.observation_space().shape, name="frames_in")
        extras_in = tf.keras.Input(shape=(1,), name="extras_in")

        x = tf.layers.Conv2D(32, (7, 7), input_shape=self.observation_space().shape)(frames_in)
        x = tf.layers.Conv2D(32, (7, 7))(x)
        x = tf.layers.AveragePooling2D((7, 7), 7)(x)
        x = tf.layers.Flatten()(x)
        x = tf.keras.layers.Concatenate()([x, extras_in])
        # x = tf.layers.Dense(20, kernel_initializer=initializer, activation=tf.keras.activations.relu)(x)
        # x = tf.layers.Dense(10, kernel_initializer=initializer, activation=tf.keras.activations.relu)(x)
        # x = tf.keras.layers.Lambda(lambda data: tf.image.rgb_to_grayscale(data), )(x)
        # x = tf.layers.Flatten(input_shape=self.observation_space().shape)(x)
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
        x = tf.layers.Dense(self.action_space().n, kernel_initializer=initializer)(x)

        model = tf.keras.models.Model(inputs=[frames_in, extras_in], outputs=[x])
        model.summary()

        model.compile(tf.keras.optimizers.Adam(lr=self.lr), tf.keras.losses.mean_squared_error)
        return model

    def build_target_model(self):
        return self.build_model(initializer=tf.keras.initializers.Zeros())

    def transfer_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def setup_models(self):
        self.model = self.build_model(initializer=tf.keras.initializers.Zeros())
        self.target_model = self.build_target_model()
        return self

    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        if self._action_space: return self._action_space

        self._action_space = gym.spaces.Discrete(len(self.actions()))
        return self._action_space

    def step(self, human_action=None):
        if human_action is not None:
            self.action = human_action

        new_state, reward, done, info = self.env.step(self.action)
        info['stuck_distance'] = 1/(self.stucks_store.smallest_distance(info["x"], info["y"])+0.01)
        old_state, self.state = self.state, new_state
        old_action, self.action = self.action, self.select_action(new_state, [info["stuck_distance"]])

        self.max_x = max(info["x"], self.max_x)

        if self.first_x is None:
            self.first_x = info["x"]

        if human_action is None and self.episode_steps > 0 and self.episode_steps % 100 == 0:
            self.last_x = info["x"]

            if self.first_x and self.last_x and abs(self.first_x - self.last_x) < 20:
                self.first_x = self.last_x = None
                done = True
                self.stucks_store.add(info["x"], info["y"])
                if reward == 0: reward = -100
            else:
                self.first_x = self.last_x

        self.remember(old_state, old_action, new_state, reward, done, info, self.prev_info, self.action)

        self.prev_info = info
        self.num_steps += 1
        self.episode_steps += 1

        return done

    def remember(self, state, action, new_state, reward, done, info, prev_info, new_action):
        self.episode_memory.append((state, action, new_state, reward, done, info, prev_info, new_action))

    def select_action(self, state, extras):
        probs = self._select_action_probs(state, extras)
        return self.actions()[np.random.choice(range(len(probs)), p=probs)]

    def _select_action_probs(self, state, extras):
        predicted = self.model.predict([self._reshape_state(state), np.array(extras)])[0]

        num_actions = self.action_space().n
        probs = np.full(num_actions, self.epsilon/num_actions)
        action = np.argmax(predicted)
        probs[action] = 1 - self.epsilon + self.epsilon / num_actions
        return probs

    def learn_from_memory(self):
        states, actions, new_states, rewards, dones, new_actions = [], [], [], [], [], []
        stuck_distances = []
        ret = 0
        for state, action, new_state, reward, done, info, prev_info, new_action in self.episode_memory[::-1]:
            ret = reward = reward + ret*self.gamma
            stuck_distances.append(info["stuck_distance"])
            states.append(state)
            actions.append(np.where((self.actions() == np.array(action)).all(axis=1))[0][0])
            new_states.append(new_state)
            rewards.append(self.reward(reward, done, info, prev_info))
            dones.append(done)
            new_actions.append(np.where((self.actions() == np.array(new_action)).all(axis=1))[0][0])

        stuck_distances = np.array(stuck_distances)
        states = np.array(states)
        actions = np.array(actions)
        new_states = np.array(new_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        new_actions = np.array(new_actions, dtype=np.int)

        target_prediction = self.model.predict([new_states, stuck_distances])
        target_prediction[
            np.arange(len(target_prediction)), new_actions] +=\
            np.array(rewards)

        predictions = self.model.predict([states, stuck_distances])
        predictions[
            np.arange(len(predictions)), actions] = target_prediction[np.arange(len(predictions)), new_actions]

        self.model.fit([states, stuck_distances], predictions, batch_size=32, shuffle=True, verbose=1, epochs=1)

    def reward(self, reward, done, info, prev_info):
        return reward_calculator.reward(reward, done, info, prev_info)

    def train_from_movie(self, movie_file):
        print("training on", movie_file)
        env, movie, duration, = playback_movie.load_movie(movie_file)
        self.env = wrap_environment(env)

        while movie.step():
            keys = []
            for i in range(16):
                keys.append(movie.get_key(i))

            keys = list(map(float, keys))[:12]
            actions = np.where((self.actions() == np.array(keys)).all(axis=1))
            if len(actions) != 1:
                raise ValueError("keys array not present in actions", keys)
            else:
                action = self.ACTIONS[actions[0]][0]

            self.step(action)

        env.close()
        movie.close()
