#!/usr/bin/env python
from __future__ import print_function
import time
import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), "..", "support"))
from retro_contest import local
import numpy as np

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

env = local.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', bk2dir="human_games")

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decisi
# on SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

pressed = set()
def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    pressed.add(a)
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    pressed.remove(a)
    if human_agent_action == a:
        human_agent_action = 0

env.reset()
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        action = np.zeros(env.action_space.n)
        for key in pressed:
            action[key] = 1
        obser, r, done, info = env.step(action)
        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done:
            break
        if human_wants_restart:
            break
        while human_sets_pause:
            env.render()
            time.sleep(0.05)
        time.sleep(0.05)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break

