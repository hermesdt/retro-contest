from src.dqn import DQN
from glob import glob
import random, os
import numpy as np
from retro.scripts import playback_movie
from src import actions_builder, env_creator
import math
from src import utils
from collections import deque

logger = utils.get_logger(__name__)

def train_on_random_movie(dqn):
    human_games = glob("human_games/**")
    movie = random.sample(human_games, 1)[0]
    train_from_movie(dqn, movie)

def train_from_movie(dqn, movie_file):
    logger.info("Training on movie {}".format(movie_file))
    env, movie, duration, = playback_movie.load_movie(movie_file)
    env = env_creator.wrap_environment(env)
    dqn.reset(env)
    memory = []
    total_reward = 0
    total_steps = 0

    while movie.step():
        total_steps += 1
        keys = []
        for i in range(16):
            keys.append(movie.get_key(i))

        keys = list(map(float, keys))[:12]
        actions = np.where((dqn.ACTIONS == np.array(keys)).all(axis=1))

        if len(actions) != 1:
            raise ValueError("keys array not present in actions", keys)
        else:
            action = dqn.ACTIONS[actions[0]][0]

            state, action, new_state, reward, done, info, new_action = dqn.step(env, action)
            total_reward += reward

        if len(memory) > 0:
            memory[-1][-1] = action

        memory.append([state, action, new_state, reward, done, info, new_action])

    dqn.learn_from_memory(memory)
    dqn.model.save_weights("weights/alvaro_dqn_model.h5")
    logger.info("Total reward {}, total_steps {}".format(total_reward, total_steps))

    memory.clear()
    del memory
    env.close()
    movie.close()

def enqueue_episode(episodes, total_reward, total_steps, memory):
    if len(episodes) < episodes.maxlen:
        episodes.append((total_reward, total_steps, memory))
        return

    for index, episode in enumerate(episodes):
        reward, steps, memory = episode
        if total_reward > reward:
            episodes[index] = (total_reward, total_steps, memory)
            break


def train_on_env(dqn, env, epochs=1, train_steps=500, render=False,
                 manual_interventions_enabled=True,
                 manual_intervention_epsilon=0.8,
                 manual_intervention_duration=200):

    episodes = deque(maxlen=5)
    for epoch in range(epochs):
        episode_steps = 0
        done = False
        prev_info = None
        total_reward = 0
        memory = []
        first_x, last_x = None, None
        max_x = 0
        pushing_wall = False
        real_rewards = []

        dqn.reset(env)

        while not done:
            max_x = 0
            episode_steps += 1

            #extras = [[(sum(real_rewards[-20:]))/20]]
            extras = [[]]
            state, action, new_state, reward, done, info, new_action, extra_info = dqn.step(env, _extra_info=extras)
            real_rewards.append(reward)

            reward = 0

            max_x = max(max_x, info["x"])
            total_reward += reward

            if render:
                env.render()

            if done:
                sum_last_rewards = sum(real_rewards)
                memory[-1][3] = sum_last_rewards

            if not done:
                if episode_steps % train_steps == 0 and episode_steps > 0:
                    sum_last_rewards = sum(real_rewards[:])
                    logger.info("- trigger online batch training (reward {}, max_x {})".format(sum_last_rewards, max_x))
                    #memory[-1][3] = sum_last_rewards
                    dqn.learn_from_memory(memory[:])

            memory.append([state, action, new_state, reward, done, info, new_action, extra_info])
            prev_info = info

        enqueue_episode(episodes, total_reward, episode_steps, memory)
        #for total_reward, total_steps, memory in episodes:
        #    logger.info("training on memory with reward {}, steps {}".format(total_reward, total_steps))
        #    dqn.learn_from_memory(memory)

        dqn.learn_from_memory(memory)
        #[dqn.learn_from_memory(mem) for mem in full_memories]

        dqn.model.save_weights("weights/alvaro_dqn_model.h5")

        logger.info("Total reward {}, total_steps {}, max_x {}".format(
            sum(real_rewards), episode_steps, max_x))


