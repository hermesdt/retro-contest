from src import env_creator, actions_builder
from src.dqn import DQN
from src import trainer
from src import utils
from retro import list_games, list_states
import random

logger = utils.get_logger(__name__)

env = env_creator.create_environment(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
actions = actions_builder.build(env.action_space)
dqn = DQN(env.observation_space, actions, epsilon=0.2, gamma=0.995, lr=0.00001, keras_verbose=0)
env.close()

#trainer.train_on_random_movie(dqn)
#trainer.train_on_random_movie(dqn)

def create_random_env():
    game = 'SonicTheHedgehog-Genesis'
    state = random.choice(list_states(game))
    return env_creator.create_environment(game, state)

for i in range(100):
    for i in range(1):
        trainer.train_on_random_movie(dqn)
    
    # env = create_random_env()
    env = env_creator.create_environment('SonicTheHedgehog-Genesis', "MarbleZone.Act2")
    trainer.train_on_env(dqn, env, epochs=50, render=True,
                         train_steps=200,
                         manual_interventions_enabled=False,
                         manual_intervention_epsilon=0.8,
                         manual_intervention_duration=200)
    env.close()
