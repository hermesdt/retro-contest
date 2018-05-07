import tensorflow as tf
tf.reset_default_graph()

from support.gym_remote import exceptions as gre
from support.gym_remote import client as grc
from src.env_creator import wrap_environment
from src.dqn import DQN
import traceback

from tensorflow.python import pywrap_tensorflow as c_api
print("c_api", c_api)

def train_on_game(dqn):
    done = False

    while not done:
        done = dqn.step()

def main():
    print('connecting to remote environment')
    env = grc.RemoteEnv('tmp/sock')
    env = wrap_environment(env)

    dqn = DQN(env, reply_memory_size=50000, steps_learn_from_memory=500000, replay_actions=2000, epsilon=0.03,
                  gamma=0.995).setup_models()

    dqn.model.load_weights("weights/alvaro_dqn_model.h5")
    dqn.target_model.load_weights("weights/alvaro_dqn_target_model.h5")

    print('starting episode')
    dqn._epsilon = 0.2

    while True:
        train_on_game(dqn)
        dqn.learn_from_memory()
        dqn._epsilon *= 0.98
        print('episode complete')
        dqn.env = env


if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as e:
        import sys
        exc_type, exc_value, tb = sys.exc_info()
        if tb is not None:
            prev = tb
            curr = tb.tb_next
            while curr is not None:
                prev = curr
                print(prev.tb_frame.f_locals)
                curr = curr.tb_next
            print(prev.tb_frame.f_locals)

        traceback.print_tb(e.__traceback__)
        traceback.print_exc()
        print('exception', e)