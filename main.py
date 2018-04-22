import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), "support"))
from retro_contest import local
from baselines.common import atari_wrappers
from src import dqn

env = local.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')

# see https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L223
# can't us it because if specific for atari env, should we create or own?
# env = atari_wrappers.EpisodicLifeEnv(env)
env = atari_wrappers.WarpFrame(env)
env = atari_wrappers.ScaledFloatFrame(env)
# env = atari_wrappers.ClipRewardEnv(env)
env = atari_wrappers.FrameStack(env, 4)

dqn = dqn.DQN(env).setup_models()

for j in range(5):
    print("episode", j)
    for i in range(1000):
        if dqn.step():
            break
    env.reset()

for i in range(4000):
    env.render()
    if dqn.step():
        break