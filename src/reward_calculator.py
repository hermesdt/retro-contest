def reward(reward, done, info, prev_info):
    # if done:
    #     reward
#
    # if reward:
    #     return reward
#
    if prev_info:
        if info["lives"] < prev_info["lives"]:
            return -0.1

        if info["x"] > prev_info["x"] + 1:
            return 0.01

    return -0.001#
