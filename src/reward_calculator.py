def reward(reward, info, prev_info):
    if prev_info is None:
        return reward

    if prev_info['x'] < info['x']:
        reward = 10
    else:
        reward = -1

    return reward