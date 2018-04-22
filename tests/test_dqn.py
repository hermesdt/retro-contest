import unittest
from unittest.mock import Mock
from src.dqn import DQN
from gym import spaces
import numpy as np
import tensorflow as tf

class TestDqn(unittest.TestCase):
    def setUp(self):
        observation_space = spaces.Box(0, 255, shape=(224, 320, 3), dtype=np.uint8)
        environment = Mock(action_space=spaces.MultiBinary(12),
                           observation_space=observation_space,
                           reset=lambda: observation_space.sample())
        self.dqn = DQN(environment)

    def test_observation_space(self):
        self.assertEqual(self.dqn.observation_space().shape, (224, 320, 3))

    def test_action_space(self):
        self.assertEqual(self.dqn.action_space().n, 12)

    def test_reward(self):
        self.assertEqual(self.dqn.reward(10, None, None), 10)

    def test_build_model(self):
        model = self.dqn.build_model()
        self.assertEqual(model.layers[-1].output.shape.as_list(), [None, 12])

    def test_setup_models(self):
        self.dqn.build_model = Mock()
        self.dqn.setup_models()

        call_one, call_two = self.dqn.build_model.call_args_list
        self.assertEqual(call_one, (()))
        self.assertIsInstance(call_two[1]['initializer'], tf.keras.initializers.Zeros)

    def test_transfer_weights(self):
        self.dqn.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, input_shape=[10], kernel_initializer="ones"),
            tf.keras.layers.Dense(1, kernel_initializer="ones")
        ])
        self.dqn.model.compile(optimizer="sgd", loss="mse")

        self.dqn.target_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, input_shape=[10], kernel_initializer="zeros"),
            tf.keras.layers.Dense(1, kernel_initializer="zeros")
        ])
        self.dqn.target_model.compile(optimizer="sgd", loss="mse")

        w1, b1, w2, b2 = self.dqn.target_model.get_weights()
        self.assertEqual(np.sum(w1), 0)
        self.assertEqual(np.sum(b1), 0)
        self.assertEqual(np.sum(w2), 0)
        self.assertEqual(np.sum(b2), 0)

        self.dqn.transfer_weights()
        w1, b1, w2, b2 = self.dqn.target_model.get_weights()
        self.assertEqual(np.sum(w1), 100)
        self.assertEqual(np.sum(b1), 0)
        self.assertEqual(np.sum(w2), 10)
        self.assertEqual(np.sum(b2), 0)

    def test_actions(self):
        actions = self.dqn.actions()
        self.assertEqual(actions.tolist(), [
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
        ])

    def test_step(self): pass

    def test_select_action(self): pass

    def test__select_action_probs(self):
        self.dqn.setup_models()
        state = self.dqn.observation_space().sample()
        probs = np.round(self.dqn._select_action_probs(self.dqn._reshape_state(state)), 2)

        np.testing.assert_allclose(probs, [0.91, 0.01, 0.01 , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ])

    def test_td_target(self):
        self.dqn.setup_models()
        w = self.dqn.target_model.get_weights()
        target = self.dqn.td_target(0.9, self.dqn._reshape_state(self.dqn.observation_space().sample() + 1))
        target = np.round(target, 2)
        np.testing.assert_equal(target, [0.9, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
