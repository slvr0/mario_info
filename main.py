#usr/bin/python3.6

from env import start_env
from agent import DqnAgent

from tensorflow.python.framework.ops import disable_eager_execution

from env import MarioEnvironment

disable_eager_execution()

config = \
    {   'world': 1,
        'level': 1,
        'lr': 1e-4, #learning rate
        'num_process': 6, #threads
        'gamma': .9, #gamma tau beta explains how the agent balances recent training data against historic as well as balance actor value vs critic
        'tau': 1.0,
        'beta': 0.01,
        'im_size': 84,
        'frame_skip': 4,
        'episode_steps': 2000,
        'nr_episodes': 20000,
        'save_net_interval': 10,
        'log_path': 'logs',
        'record_path' : 'logs/videos',
        'action_type' : 'right', #defines how marios dimensions of freedom, cross jumps/duck/backwards move etc
        'win_reward' : 50,
        'milestone_reward_scaler' : 40,
        'batch_size' : 2,
        'test_mode' : False
    }

if __name__ == '__main__':

    menv = MarioEnvironment(config)

    env, input_dims, output_dims = menv.initiate()

    # env, input_dims , output_dims = start_env()

    dqn = DqnAgent(env, input_dims, output_dims)

    render = False
    test = False
    dqn.simulate_game(render=render, test=test)



