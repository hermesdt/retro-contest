from support.retro_contest import local
from baselines.common import atari_wrappers

def create_environment(game, state):
    env = local.make(game, state, bk2dir="games")
    env = wrap_environment(env)
    return env

def wrap_environment(env):
    # see https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L223
    # can't us it because if specific for atari env, should we create or own?
    # env = atari_wrappers.EpisodicLifeEnv(env)
    env = atari_wrappers.WarpFrame(env)
    env = atari_wrappers.ScaledFloatFrame(env)
    # env = atari_wrappers.ClipRewardEnv(env)
    env = atari_wrappers.FrameStack(env, 4)

    return env
