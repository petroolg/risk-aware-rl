import numpy as np
import tensorflow as tf

import tensorflow as tf
import logging
import os
import matplotlib

matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import numpy as np
import sklearn

from simple_car_game import *
from pdb import set_trace

GAMMA = 0.9


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class SGDRegressor_occupancy:
    def __init__(self, xd, restore=True):

        self.save_path = 'graphs/graph_supervised/graph.ckpt'

        self.states = tf.placeholder(tf.float32, shape=[None, xd - 2], name='states')
        self.actions = tf.placeholder(tf.float32, shape=[None, 9], name='actions')
        self.action_ind = tf.placeholder(tf.int32, shape=[None, 2], name='act_inds')
        self.weight_ind = tf.placeholder(tf.int32, shape=[None, 2], name='weiht_inds')

        with tf.variable_scope('policy'):
            pi0 = tf.layers.dense(self.states, 64, activation=tf.nn.sigmoid, use_bias=True)
            pi1 = tf.layers.dense(pi0, 32, activation=tf.nn.sigmoid, use_bias=True)
            pi2 = tf.layers.dense(pi1, 16, activation=tf.nn.sigmoid, use_bias=True)
            pi = tf.layers.dense(pi2, 9, activation=tf.nn.softmax, use_bias=True)
            self.pi = tf.gather_nd(pi, self.action_ind)

        self.weights = tf.constant(np.array([[1, 1, 1, 1, 0.1, 1, 1, 1, 1]]).astype('float32'))
        self.weights_picked = tf.gather_nd(self.weights, self.weight_ind)

        self.smce = tf.losses.mean_squared_error(tf.gather_nd(self.actions, self.action_ind), self.pi,
                                                 weights=self.weights_picked)

        self.cost = self.smce

        # ops we want to call later
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train = optimizer.minimize(self.cost)

        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        if restore:
            self.saver.restore(self.session, self.save_path)
        else:
            self.session.run(init)

        # self.session.graph.finalize()

    def partial_fit(self, sa_pairs, actions):
        action_inds = [game.actios.index(s[-2:].tolist()) for s in sa_pairs]
        action_inds = np.vstack((np.arange(0, len(sa_pairs), 1), action_inds))
        inds = action_inds.T
        weight_inds = [game.actios.index(s[-2:].tolist()) for s in sa_pairs]
        weight_inds = np.vstack((np.zeros((1, len(sa_pairs))), weight_inds))
        weight_inds = weight_inds.T
        self.session.run(self.train,
                         feed_dict={self.actions: actions, self.states: sa_pairs[:, :-2], self.action_ind: inds,
                                    self.weight_ind: weight_inds})

    def predict(self, sa_pairs):
        action_inds = [game.actios.index(s[-2:].tolist()) for s in sa_pairs]
        action_inds = np.vstack((np.arange(0, len(sa_pairs), 1), action_inds))
        inds = action_inds.T
        weight_inds = [game.actios.index(s[-2:].tolist()) for s in sa_pairs]
        weight_inds = np.vstack((np.zeros((1, len(sa_pairs))), weight_inds))
        weight_inds = weight_inds.T
        return self.session.run(self.pi, feed_dict={self.states: sa_pairs[:, :-2], self.action_ind: inds,
                                                    self.weight_ind: weight_inds})

    def save_sess(self):
        self.saver.save(self.session, self.save_path)

    def look_incide(self, sa_pairs, actions):
        action_inds = [game.actios.index(s[-2:].tolist()) for s in sa_pairs]
        action_inds = np.vstack((np.arange(0, len(sa_pairs), 1), action_inds))
        inds = action_inds.T

        weight_inds = [game.actios.index(s[-2:].tolist()) for s in sa_pairs]
        weight_inds = np.vstack((np.zeros((1, len(sa_pairs))), weight_inds))
        weight_inds = weight_inds.T

        pi = self.session.run(self.pi, feed_dict={self.states: sa_pairs[:, :-2], self.action_ind: inds})
        weights = self.session.run(self.weights_picked,
                                   feed_dict={self.states: sa_pairs[:, :-2], self.weight_ind: weight_inds})
        softm = self.session.run(self.smce,
                                 feed_dict={self.actions: actions, self.states: sa_pairs[:, :-2], self.action_ind: inds,
                                            self.weight_ind: weight_inds})
        c0 = self.session.run(self.cost,
                              feed_dict={self.actions: actions, self.states: sa_pairs[:, :-2], self.action_ind: inds,
                                         self.weight_ind: weight_inds})
        comy = np.multiply(weights, softm)
        comy_mean = np.mean(comy)


def sample_trajectories(game, model):
    trajectories = []

    while len(trajectories) < 20:
        game.init_game(seed=None)
        # s_a_pairs = []
        total_rew = 0
        while not game.game_over:
            st = game.state

            # print(np.repeat(st,len(game.actios),axis=0))
            # print(game.actios)

            action_probs = model.predict(np.hstack((np.repeat([st], len(game.actios), axis=0), game.actios)))
            idx = np.random.choice(range(9), p=action_probs.ravel())
            d_v = game.actios[idx]
            print(np.sum(action_probs), d_v)
            # s_a_pairs.append(np.hstack((st, d_v)))
            rew = game.move(d_v)
            total_rew += rew
            time.sleep(0.1)
            # print('\rVelocity: {}'.format(game.car.v), end='')
        # trajectories.append(s_a_pairs)
    # return trajectories


def one_hot(actions, game):
    new_y = np.zeros((len(actions), 9))
    for i, a in enumerate(actions):
        idx = game.actios.index(a.tolist())
        new_y[i, idx] = 1.0

    return new_y


if __name__ == '__main__':
    expert_traj = []

    traj_path = 'trajectories_all/trajectories_rand_big/'

    game = Road_game()

    for i, t in enumerate(os.listdir(traj_path)):
        raw_traj = np.load(traj_path + t)
        expert_traj.append(raw_traj)

    expert_traj = np.array(expert_traj)
    expert_tuples = np.vstack(expert_traj)
    X = expert_tuples[:, :-2]
    y = expert_tuples[:, -2:]
    y = one_hot(y, game)
    model = SGDRegressor_occupancy(expert_traj[0].shape[1])

    logging.basicConfig(filename='images/road_game.log', level=logging.DEBUG)

    N_iterations = 2000

    for i in range(N_iterations):
        print('{}/{}'.format(i, N_iterations))

        logging.debug('======Iteration #{}======'.format(i))
        model.partial_fit(expert_tuples, y)
        # model.look_incide(expert_tuples, y)

    model.save_sess()

    sample_trajectories(game, model)
