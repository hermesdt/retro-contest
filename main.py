from src import dqn
from glob import glob
import random
from src.env_creator import create_environment


env = create_environment(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act3')
dqn = dqn.DQN(env, reply_memory_size=50_000, steps_learn_from_memory=500000, replay_actions=2000, epsilon=0.2,
              gamma=0.5).setup_models()
#dqn.model.load_weights("weights/alvaro_dqn_model.h5")
#dqn.target_model.load_weights("weights/alvaro_dqn_target_model.h5")
env.close()
dqn._env = None

human_games = glob("human_games/*")
def train_on_random_movie(dqn):
    movie = random.sample(human_games, 1)[0]
    dqn.train_from_movie(movie)

def train_on_game(dqn, render=False, env=None):

    if env is None:
        env = create_environment(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    dqn.env = env
    done = False

    while not done:
        done = dqn.step()
        if render:
            env.render()

    #env.close()
    print("Episode {}, steps {}, last_x {}, epsilon {}".format(i, dqn.episode_steps, dqn.max_x, dqn._epsilon), flush=True)

if __name__ == "__main__":
    from retro import list_games
    print(list_games())
    for i in range(10):
        train_on_random_movie(dqn)
        dqn.learn_from_memory()
    dqn.env = create_environment(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act2')

    for i in range(1000):
        train_on_game(dqn, render=False, env=dqn.env)
        dqn.learn_from_memory()
        train_on_game(dqn, render=False, env=dqn.env)
        dqn.learn_from_memory()
        dqn._epsilon *= 0.98

        print("Episode {}, steps {}, last_x {}, epsilon {}".format(i, dqn.episode_steps, dqn.max_x, dqn._epsilon), flush=True)
        dqn.model.save_weights("weights/alvaro_dqn_model.h5")
        dqn.target_model.save_weights("weights/alvaro_dqn_target_model.h5")

    # train_on_game(dqn, render=True)
    dqn.env.close()
