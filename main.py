from support.retro_contest import local
from baselines.common import atari_wrappers
from src import dqn
from glob import glob
import random

def create_environment():
    env = local.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')

    # see https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L223
    # can't us it because if specific for atari env, should we create or own?
    # env = atari_wrappers.EpisodicLifeEnv(env)
    env = atari_wrappers.WarpFrame(env)
    env = atari_wrappers.ScaledFloatFrame(env)
    # env = atari_wrappers.ClipRewardEnv(env)
    env = atari_wrappers.FrameStack(env, 4)

    return env


env = create_environment()
dqn = dqn.DQN(env, reply_memory_size=50_000, steps_learn_from_memory=500, replay_actions=1000).setup_models()
env.close()

human_games = glob("human_games/*")
def train_on_random_movie(dqn):
    movie = random.sample(human_games, 1)[0]
    dqn.train_from_movie(movie)

def train_on_game(dqn, render=False):
    env = create_environment()
    dqn.env = env
    done = False

    while not done:
        done = dqn.step()
        if render:
            env.render()

    env.close()


if __name__ == "__main__":
    for i in range(5):
        train_on_random_movie(dqn)
        train_on_game(dqn, render=True)
        dqn.model.save_weights("weights/alvaro_dqn_model.h5")
        dqn.target_model.save_weights("weights/alvaro_dqn_target_model.h5")

    train_on_game(dqn, render=True)