import numpy as np
import itertools

def build_full(action_space):
    actions = []
    for i in range(action_space.n):
        action_array = np.zeros(action_space.n)
        action_array[i] = 1
        actions.append(action_array)

    return np.asarray(actions)

def build_simplified(action_space):
    """
    Return all combinations of actions 0, 6, 7, 8 (no-np, left, right, jump)
    """
    actions = [6, 7, 8]

    multi_binary_actions = []
    for combination in itertools.chain.from_iterable(itertools.combinations(actions, i) for i in range(len(actions) + 1)):
        multi_binary_action = np.zeros(action_space.n)
        multi_binary_action[list(combination)] = 1
        multi_binary_actions.append(multi_binary_action)

    multi_binary_actions.append(np.zeros(action_space.n))

    return np.array(multi_binary_actions)

def build(action_space):
    return build_simplified(action_space)