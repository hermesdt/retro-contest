import unittest
from src import actions_builder
from gym.spaces import MultiBinary
import numpy as np

class TestActionsBuilder(unittest.TestCase):
    def test_build_simplified(self):
        actions = actions_builder.build_simplified(MultiBinary(12))
        np.testing.assert_equal(actions, [
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.]
        ])