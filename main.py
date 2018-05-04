from src import dqn
from glob import glob
import os
import random
from src.env_creator import create_environment
from retro import list_games, list_states


env = create_environment(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act3')
dqn = dqn.DQN(env, reply_memory_size=50_000, steps_learn_from_memory=500000, replay_actions=2000, epsilon=0.03,
              gamma=0.995).setup_models()

if os.path.exists("weights/alvaro_dqn_model.h5"):
    dqn.model.load_weights("weights/alvaro_dqn_model.h5")
    dqn.target_model.load_weights("weights/alvaro_dqn_target_model.h5")
env.close()
dqn._env = None

human_games = glob("human_games/*")
def train_on_random_movie(dqn):
    movie = random.sample(human_games, 1)[0]
    dqn.train_from_movie(movie)

def random_state():
    # game = random.choice(list_games())
    game = 'SonicTheHedgehog-Genesis'
    state = random.choice(list_states(game))
    return game, state

def train_on_game(dqn, render=False):
    done = False

    while not done:
        done = dqn.step()
        if render:
            env.render()

if __name__ == "__main__":

    episodes = 0

    while True:
        for i in range(1):
            train_on_random_movie(dqn)
            dqn.learn_from_memory()

        game, state = random_state()

        env = create_environment(game="SonicTheHedgehog-Genesis", state=state)
        dqn.env = env
        dqn._epsilon = 0.2

        for i in range(50):
            dqn.env = env
            train_on_game(dqn, render=False)
            dqn.learn_from_memory()
            dqn._epsilon *= 0.98

            episodes += 1
            print("({}/{}) Episode {}, steps {}, last_x {}, epsilon {}".format(
                game, state, episodes, dqn.episode_steps, dqn.max_x, dqn._epsilon), flush=True)
            dqn.model.save_weights("weights/alvaro_dqn_model.h5")
            dqn.target_model.save_weights("weights/alvaro_dqn_target_model.h5")

        env.close()
