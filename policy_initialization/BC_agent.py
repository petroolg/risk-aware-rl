import argparse
import os

import tensorflow as tf
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MDP"))

from environment import *

GAMMA = 0.9
N_ITERATIONS = 2000
N_EVAL_GAMES = 200


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class SGDRegressor:
    def __init__(self, xd, restore=False):

        self.save_path = 'models/model_BC/model.ckpt'

        self.states = tf.placeholder(tf.float32, shape=[None, xd], name='states')
        self.actions = tf.placeholder(tf.float32, shape=[None, 9], name='actions')

        with tf.variable_scope('policy'):
            pi0 = tf.layers.dense(self.states, 64, activation=tf.nn.sigmoid, use_bias=True)
            pi1 = tf.layers.dense(pi0, 32, activation=tf.nn.sigmoid, use_bias=True)
            self.pi = tf.layers.dense(pi1, 9, activation=tf.nn.softmax, use_bias=True)

        self.weights = tf.constant(np.array([[1, 1, 1, 1, 0.1, 1, 1, 1, 1]]).astype('float32'))

        weights = tf.reduce_sum(self.weights * self.actions, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.actions, logits=self.pi)
        weighted_losses = unweighted_losses * weights
        self.loss = tf.reduce_mean(weighted_losses)

        # ops we want to call later
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train = optimizer.minimize(self.loss)

        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        if restore:
            self.saver.restore(self.session, self.save_path)
        else:
            self.session.run(init)

        self.session.graph.finalize()

    def partial_fit(self, states, actions):
        self.session.run(self.train, feed_dict={self.actions: actions, self.states: states})

    def predict(self, states):
        return self.session.run(self.pi, feed_dict={self.states: states})

    def compute_loss(self, states, actions):
        return self.session.run(self.loss, feed_dict={self.actions: actions, self.states: states})

    def save_sess(self):
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        self.saver.save(self.session, self.save_path)


def evaluate_performance(model):
    print("Evaluating performance.")
    stats = []
    for j in range(N_EVAL_GAMES):
        game.init_game(seed=None)
        s_a_pairs = []
        total_rew = 0
        print('\rValidating strategy: {}/{}'.format(j, N_EVAL_GAMES), end="")
        while not game.game_over:
            st = game.state
            action_probs = model.predict(np.atleast_2d(st))
            idx = np.random.choice(range(9), p=action_probs.ravel())
            d_v = game.actios[idx]
            s_a_pairs.append(np.hstack((st, d_v)))
            rew = game.move(d_v)
            total_rew += rew
        stats.append((total_rew, len(s_a_pairs), int(game.collision())))

    stats = np.array(stats)

    return_ = stats[:, 0]
    return_per_step = stats[:, 0] / stats[:, 1]
    episode_length = stats[:, 1]
    collisions = stats[:, 2]

    print("Performance metrics for the BC agent:")
    print(f"Return mean: {np.mean(return_)}, median: {np.median(return_)}")
    print(f"Return per step mean: {np.mean(return_per_step)}, median: {np.median(return_per_step)}")
    print(f"Episode length: {np.mean(episode_length)}, median: {np.median(episode_length)}")
    print(f"Collision rate: {sum(collisions) / len(collisions)}")

    with open('results/BC_stats.txt', 'w') as file:
        file.writelines([", ".join([str(i) for i in t]) + "\n" for t in stats])


def one_hot(actions, game):
    new_y = np.zeros((len(actions), 9))
    for i, a in enumerate(actions):
        idx = game.actios.index(a.tolist())
        new_y[i, idx] = 1.0

    return new_y


def run_experiment(model, X, y):
    for i in range(N_ITERATIONS):
        if i % 100 == 0:
            loss = model.compute_loss(X, y)
            print('Iteration {}/{}, Loss: {}'.format(i, N_ITERATIONS, loss))
            model.save_sess()
        model.partial_fit(X, y)


def prepare_data(traj_path):
    expert_traj = []
    for i, t in enumerate(os.listdir(traj_path)):
        raw_traj = np.load(os.path.join(traj_path, t))
        expert_traj.append(raw_traj)
    expert_traj = np.array(expert_traj)
    expert_tuples = np.vstack(expert_traj)
    X = expert_tuples[:, :-2]
    y = expert_tuples[:, -2:]
    y = one_hot(y, game)

    return X, y

def prepare_experiment():
    folders = ['results']

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Behavioral Cloning experiment parameters')
    parser.add_argument("--tp", dest='traj_path', type=str, required=True, default="trajectories",
                        help='Path to the directory with expert\'s trajectories.')

    args = parser.parse_args()
    traj_path = args.traj_path

    prepare_experiment()
    game = Road_game(n_steps=50)

    X, y = prepare_data(traj_path)
    model = SGDRegressor(X.shape[1])

    run_experiment(model, X, y)
    evaluate_performance(model)