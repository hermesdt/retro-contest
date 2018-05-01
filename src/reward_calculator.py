END_OF_GAME = -100

def reward(reward, done, info, prev_info):
    if prev_info:
        if info["lives"] < prev_info["lives"]:
            return END_OF_GAME

    return reward
