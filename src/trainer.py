from src.dqn import DQN
from glob import glob
import random, os
import numpy as np
from retro.scripts import playback_movie
from src import actions_builder, env_creator
import math
from src import utils

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

        memory.append((state, action, new_state, reward, done, info, new_action))

    dqn.learn_from_memory(memory)
    dqn.model.save_weights("weights/alvaro_dqn_model.h5")
    logger.info("Total reward {}, total_steps {}".format(total_reward, total_steps))

    memory.clear()
    del memory
    env.close()
    movie.close()

def train_on_env(dqn, env, epochs=1, train_steps=500, render=False,
                 manual_interventions_enabled=True,
                 manual_intervention_epsilon=0.8,
                 manual_intervention_duration=200):
    for epoch in range(epochs):
        episode_steps = 0
        done = False
        prev_info = None
        total_reward = 0
        memory = []
        initial_epsilon = dqn.epsilon
        epsilon_resetted_at = None
        first_x, last_x = None, None
        manual_interventions = 0

        dqn.reset(env)

        while not done:
            episode_steps += 1
            if epsilon_resetted_at and episode_steps - epsilon_resetted_at >= manual_intervention_duration:
                epsilon_resetted_at = None
                dqn.epsilon = initial_epsilon

            state, action, new_state, reward, done, info, new_action = dqn.step(env)
            total_reward += reward

            if render:
                env.render()

            memory.append((state, action, new_state, reward, done, info, new_action))

            if not done:
                if episode_steps % train_steps == 0 and episode_steps > 0:
                    logger.info("- trigger online batch training (reward {})".format(round(total_reward)))
                    dqn.learn_from_memory(memory)

                # manual intervention
                if epsilon_resetted_at is None and manual_interventions_enabled and episode_steps > 0 and episode_steps % 100 == 0:
                    last_x = info["x"]

                    if first_x and last_x and abs(first_x - last_x) < 20:
                        logger.info("- manual intervention triggered (reward {})".format(round(total_reward)))
                        manual_interventions += 1
                        first_x = last_x = None
                        reward = -10
                        epsilon_resetted_at = episode_steps
                        dqn.epsilon = manual_intervention_epsilon
                    else:
                        first_x = last_x

            prev_info = info

        dqn.learn_from_memory(memory)
        dqn.model.save_weights("weights/alvaro_dqn_model.h5")
        memory.clear()
        dqn.epsilon = initial_epsilon

        logger.info("Total reward {}, total_steps {}, manual interventions {}".format(
            round(total_reward), episode_steps, manual_interventions))
