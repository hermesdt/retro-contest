from support.retro_contest import local
from baselines.common import atari_wrappers
from src import dqn

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
dqn = dqn.DQN(env).setup_models()

# for j in range(5):
#     print("episode", j)
#     for i in range(1000):
#         if dqn.step():
#             break
#     env.reset()
#
# for i in range(4000):
#     env.render()
#     if dqn.step():
#         break

env.close()
dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0001.bk2")
dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0002.bk2")
dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0003.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0004.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0005.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0006.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0007.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0001.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0002.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0003.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0004.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0005.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0006.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0007.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0001.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0002.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0003.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0004.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0005.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0006.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0007.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0001.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0002.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0003.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0004.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0005.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0006.bk2")
#dqn.train_from_movie("human_games/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0007.bk2")#

env = create_environment()
dqn.env = env
dqn.state = dqn._reshape_state(dqn.env.reset())

for i in range(4000):
  env.render()
  if dqn.step():
      break