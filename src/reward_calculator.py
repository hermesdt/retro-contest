def reward(reward, done, info, prev_info):
    if done:
        reward

    if reward:
        return reward

    if prev_info and info["x"] > prev_info["x"]:
        return 0.001

    return -0.01