import gym
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution2D, Flatten, LSTM, MaxPooling2D
from keras.optimizers import Adam

from collections import deque
from keras.callbacks import TensorBoard

import os
from time import time


class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


class DqnAgent(object) :
    def __init__(self, env, input_dims , output_dims):
        self.env = env
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.tensorboard_callback = TensorBoard(log_dir='logs')
        self.mem = Memory(max_size=10000)

        self.eps_start = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.0005
        self.gamma = 0.99

        self.actor_fp = 'models/actor0'
        self.critic_fp = 'models/critic0'

        hidden_size = 16

        if os.path.isdir(self.actor_fp):
            print("loading actor net")
            self.actor_net = self.load_network()
        else:
            self.actor_net = self.build_cnn_net()

        optimizer = Adam(lr=1e-4)
        self.actor_net.compile(loss='mse', optimizer=optimizer)

        self.batch_size = 32
        self.pretrain_len = self.batch_size
        self.step = 0

    def save_network(self):
        self.actor_net.save(self.actor_fp)

    def load_network(self):
        return load_model(self.actor_fp)

    def save_state(self, state_set) :
        self.mem.add(state_set)

    def choose_move_eps_greed(self, state):

        eps = self.eps_min + (self.eps_start - self.eps_min) * np.exp(-self.eps_decay * self.step)

        if eps > np.random.rand():
            action = self.env.action_space.sample()

        else:
            q = self.actor_net.predict(state)[0]
            action = np.argmax(q)

        return action, eps

    def pretrain(self):

        state = self.env.reset()



        for i in range(self.pretrain_len):

            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)

            if done:
                next_state = np.zeros(state.shape)

                self.mem.add((state, action, reward, next_state))

                self.env.reset()

                state, reward, done, _ = self.env.step(self.env.action_space.sample())

            else:
                self.mem.add((state, action, reward, next_state))
                state = next_state

    def test(self, render=False):
        done = False
        state = self.env.reset()

        score = 0
        while not done:

            #state = np.reshape(state, [1, self.input_dims])
            logits = self.actor_net.predict(state)[0]

            action = np.argmax(logits)

            new_state, reward, done, info = self.env.step(action)
            score += reward

            state = new_state

            if render: self.env.render()

        print("test run scored : {}".format(score))

    def train_network(self):

        if self.mem.size() < self.batch_size:
            return

        inputs = np.zeros((self.batch_size, 84, 84, 4))
        targets = np.zeros((self.batch_size, self.output_dims))

        t0 = time()
        minibatch = self.mem.sample(self.batch_size)
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
            inputs[i:i + 1] = state_b
            target = reward_b

            try :
                if not (next_state_b[0, 0, : , 0] == np.zeros(84)).all(axis=0):
                    target_Q = self.actor_net.predict(next_state_b)[0]
                    target = reward_b + self.gamma * np.amax(target_Q)

                targets[i] = self.actor_net.predict(state_b)
                targets[i][action_b] = target
            except Exception as p :
                print(p)
                print(state_b.shape, next_state_b.shape)

        # print(time() -t0)
        self.actor_net.fit(inputs, targets, epochs=1, verbose=0)

    def simulate_game(self, render=False, test=False):
        game_id = 1
        self.step = 0

        self.pretrain()

        n_maxscores = 0

        while True:
            if test:  # just directly test output , net predicts with 100 % argmax right away
                self.test(render)
                continue

            state = self.env.reset()

            score_sum = []

            score = 0

            t = 0
            max_ep_steps = 200
            while t < max_ep_steps:
                self.step += 1
                t += 1

                action, eps = self.choose_move_eps_greed(state)

                new_state, reward, done, info = self.env.step(action)

                score += reward

                if done :
                    t = max_ep_steps
                    new_state = np.zeros((1, 84, 84, 4))

                    print('Episode: {}'.format(game_id),
                          'Total reward: {}'.format(score),
                          'Explore P: {:.4f}'.format(eps))

                if render:
                    self.env.render()

                self.save_state((state, action, reward, new_state))

                self.train_network()

                state = new_state

            score_sum.append(score)

            #self.save_network()

            #whats the info when mario reaches flag? this is the eqvivalent
            if game_id % 50 == 0:
                print("saving network, checkpoint")
                self.save_network()
                print("performing test run...")
                self.test()

            if score < 200.0:
                n_maxscores = 0
            else :
                n_maxscores += 1

            if n_maxscores > 5:
                print("saving network, winning at game")
                self.save_network()

            game_id += 1

    def build_cartpole_net(self, hidden_layer = 16):

        model = Sequential()

        model.add(Dense(hidden_layer, activation='relu', input_dim=self.input_dims))
        model.add(Dense(hidden_layer, activation='relu'))
        model.add(Dense(self.output_dims, activation='linear'))

        optimizer = Adam(lr=1e-4)
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def build_cnn_net(self):
        model = Sequential()

        model.add(Convolution2D(16, (3, 3), strides=(1, 1), activation='relu', input_shape=(84, 84, 4)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.output_dims, activation='linear'))

        return model

        #optimizer = Adam(lr=1e-4)
        #model.compile(loss='mse', optimizer=optimizer)

        return model

