END_OF_GAME = -100

def reward(reward, done, info, prev_info):
    if prev_info and info["lives"] < prev_info["lives"]:
        return -10
    return reward
