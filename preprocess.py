import cv2
import numpy as np

import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp
import gym

class FrameProcessor(object) :
  def __init__(self, config):
    self.config = config
    self.im_size = self.config['im_size'] #add more options later

  def process_frame(self, frame):
    if frame is not None:
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
      frame = cv2.resize(frame, (self.im_size, self.im_size))[None, :, :] / 255.

      #frame = np.swapaxes(frame, 0, 2)

      return frame
    else:
      return np.zeros((1, self.im_size, self.im_size))

#maniuplates reward function during the training stage
class CustomRewardProcessor(Wrapper):

  def __init__(self, env, config, monitor=None):
    self.config = config
    self.im_size = self.config['im_size']
    self.win_reward = self.config['win_reward']
    self.milestone_reward_scaler = self.config['milestone_reward_scaler']

    self.frame_processor = FrameProcessor(config)

    super(CustomRewardProcessor, self).__init__(env)
    self.observation_space = Box(low=0, high=255, shape=(1, self.im_size, self.im_size))
    self.curr_score = 0
    if monitor:
      self.monitor = monitor
    else:
      self.monitor = None

  def step(self, action):
    state, reward, done, info = self.env.step(action)
    if self.monitor:
      self.monitor.record(state)

    state = self.frame_processor.process_frame(state)
    reward += (info["score"] - self.curr_score) * self.milestone_reward_scaler # 1/40
    self.curr_score = info["score"]
    if done:
      if info["flag_get"]:
        reward += self.win_reward
      else:
        reward -= self.win_reward
    return state, reward / 10., done, info

  def reset(self):
    self.curr_score = 0
    return self.frame_processor.process_frame(self.env.reset())


class RepeatActionAndFrame(Wrapper):
  def __init__(self, env , repeat, clip_rewards=False, no_ops=0, fire_first=False ):
    super(RepeatActionAndFrame,self).__init__(env=env)

    self.env = env
    self.shape = self.env.observation_space.low.shape
    self.repeat = repeat
    self.frame_buffer = np.zeros_like((2,self.shape))

    self.clip_rewards = clip_rewards
    self.no_ops = no_ops
    self.fire_first = fire_first
    self.curr_score = 0

  def step(self, action):
    total_reward = 0.0
    done = False

    for i in range(self.repeat):
      obs, reward, done, info = self.env.step(action)

      reward += (info["score"] - self.curr_score) / 40.


      self.curr_score = info["score"]

      if done:
        if info["flag_get"]:
          reward += 100000
        else:
          reward -= 50

      #print(reward)

      no_ops = np.random.randint(0, self.no_ops) + 1 if self.no_ops > 0 else 0
      for _ in range(no_ops):
        _, _, done, _ = self.env.step(0)
        if done:
          break

      if self.fire_first:
        assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
        obs_, _, _, _ = self.env.step(1)

      if self.clip_rewards :
        reward = np.clip(np.array([reward]), -50, 50)[0]

      total_reward += reward

      idx = i % 2 #why?
      self.frame_buffer[idx] = obs

      if done : break

    #we just want max pixel vals
    max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])

    return max_frame, total_reward, done, info

#we don't record action for every frame, instead we collect a batch of skip numbers then act upon that
class FrameSkippingProcessor(Wrapper):
  def __init__(self, env, config):
    self.config = config
    self.skip = config['frame_skip']
    super(FrameSkippingProcessor, self).__init__(env)
    self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))

  def step(self, action):
    total_reward = 0
    states = []
    state, reward, done, info = self.env.step(action)
    for i in range(self.skip):
      if not done:
        state, reward, done, info = self.env.step(action)
        total_reward += reward
        states.append(state)
      else:
        states.append(state)
    states = np.concatenate(states, 0)[None, :, :, :]
    states = states.swapaxes(1, 3)


    return states.astype(np.float32), reward, done, info

  def reset(self):
    state = self.env.reset()
    states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]

    states = states.swapaxes(1,3)

    return states.astype(np.float32)