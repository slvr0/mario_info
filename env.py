
import gym

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from preprocess import CustomRewardProcessor, FrameSkippingProcessor, RepeatActionAndFrame

from gym import Wrapper
from nes_py.wrappers import JoypadSpace

class MarioEnvironment(object) :
  def __init__(self, config,):
    self.config = config
    self.world = self.config['world']
    self.level = self.config['level']

    self.record_path = self.config['record_path']
    self.action_type = config['action_type'] #default right

  def initiate(self):
    self.env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(self.world, self.level))

    if self.action_type == "right":
      actions = RIGHT_ONLY
    elif self.action_type == "simple":
      actions = SIMPLE_MOVEMENT
    else:
      actions = COMPLEX_MOVEMENT

    #add config option to filter preprocessors we want

    self.env = JoypadSpace(self.env, actions)
    self.env = RepeatActionAndFrame(self.env, 4)
    self.env = CustomRewardProcessor(self.env, self.config)
    self.env = FrameSkippingProcessor(self.env, self.config)

    #add monitor preprocessor for automatic recording

    self.input_shape = self.env.observation_space.shape
    self.output_shape = len(actions)

    return self.env, self.input_shape, self.output_shape

def start_env() :
  env = gym.make('CartPole-v0')
  input = env.observation_space.shape[0]
  output = env.action_space.n

  return env, input, output