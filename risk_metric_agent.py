import tensorflow as tf
import numpy as np
from simple_car_game import *
from pdb import set_trace
from imitation_agent import run_avg

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import time

class Q_approx:

    def __init__(self, xd):

        self.sa_pairs = tf.placeholder(tf.float64, (None, xd), name='sa_pairs')
        self.target = tf.placeholder(tf.float64, (None, 1), name='target')

        self.weight1 = tf.Variable(tf.random_normal((xd,20),stddev=0.5,dtype=tf.float64))
        self.weight2 = tf.Variable(tf.random_normal((20, 10),stddev=0.5,dtype=tf.float64))
        self.weight3 = tf.Variable(tf.random_normal((10, 1),stddev=0.5,dtype=tf.float64))
        self.bias1 = tf.Variable(tf.random_normal((1, 20),stddev=0.5,dtype=tf.float64))
        self.bias2 = tf.Variable(tf.random_normal((1,10),stddev=0.5,dtype=tf.float64))
        self.bias3 = tf.Variable(tf.random_normal((1,),stddev=0.5,dtype=tf.float64))

        self.Q0 = tf.nn.relu(tf.matmul(self.sa_pairs, self.weight1) + self.bias1)
        self.Q1 = tf.nn.relu(tf.matmul(self.Q0, self.weight2) + self.bias2)
        self.Q = (tf.matmul(self.Q1, self.weight3) + self.bias3)

        self.cost = tf.reduce_mean(tf.squared_difference(self.Q, self.target))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = self.optimizer.minimize(self.cost)

        tf.summary.histogram('weight1', self.weight1)
        tf.summary.histogram('weight2', self.weight2)
        tf.summary.histogram('weight3', self.weight3)
        tf.summary.histogram('bias1', self.bias1)
        tf.summary.histogram('bias2', self.bias2)
        tf.summary.histogram('bias3', self.bias3)
        tf.summary.scalar('cost', self.cost)

        self.sess = tf.Session()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('graphs/risk_metric/run{}_{}'.format(str(datetime.date(datetime.now())), str(datetime.time(datetime.now()))[:8]), self.sess.graph)


        init = tf.global_variables_initializer()
        self.sess.run(init)

    def fit(self, sa_pairs, target):
        # q = self.sess.run(self.Q, feed_dict={self.sa_pairs:np.atleast_2d(sa_pairs),  self.target: np.atleast_2d(target)})
        # w1 = self.sess.run(self.weight1)
        # w2 = self.sess.run(self.weight2)
        # b1 = self.sess.run(self.bias1)
        # # b2 = self.sess.run(self.bias2)
        # grad = self.sess.run(self.optimizer.compute_gradients(self.cost, tf.trainable_variables()), feed_dict = {
        #     self.sa_pairs: np.atleast_2d(sa_pairs), self.target: np.atleast_2d(target)})
        #
        # d = self.sess.run(tf.square(self.Q - self.target), feed_dict={self.sa_pairs:np.atleast_2d(sa_pairs),  self.target: np.atleast_2d(target)})
        # prediction = self.sess.run(self.Q, feed_dict={self.sa_pairs:np.atleast_2d(sa_pairs)})

        summary, _ = self.sess.run([self.merged, self.train_op], feed_dict={self.sa_pairs: np.atleast_2d(sa_pairs), self.target: np.atleast_2d(target)})
        self.train_writer.add_summary(summary)

    def predict(self, sa_pairs):
        # q = self.sess.run(self.Q0, feed_dict={self.sa_pairs:np.atleast_2d(sa_pairs)})
        return self.sess.run(self.Q, feed_dict={self.sa_pairs:np.atleast_2d(sa_pairs)})

def softmax(vec):
    return np.exp(vec) / np.sum(np.exp(vec))

def play_game(game, model):

    sa_pairs, targets = [], []
    total_rews = []

    for i in range(10):
        game.init_game(seed=None)
        np.random.seed(None)
        total_rew = 0
        while not game.game_over:
            s_a = np.hstack((np.tile([game.state.copy()], (9,1)), np.eye(9)))
            action_probs = model.predict(s_a)
            idx = np.random.choice(range(9), p=softmax(action_probs).ravel())
            # idx = np.argmax(action_probs)
            # print(softmax(action_probs).ravel())
            d_v = game.actios[idx]

            act_one_hot = np.zeros((9,))
            act_one_hot[game.actios.index(d_v)] = 1.0
            cur_sa = np.hstack((game.state.copy(), act_one_hot))
            rew = game.move(d_v)
            total_rew += rew

            s_a_next = np.hstack((np.tile([game.state.copy()], (9, 1)), np.eye(9)))
            s_a_next = model.predict(s_a_next)
            # print(s_a_next)
            trgt = rew+0.9*np.max(s_a_next)
            sa_pairs.append(cur_sa)
            targets.append(trgt)
        total_rews.append(total_rew)

        summary = tf.Summary()
        summary.value.add(tag='Total reward', simple_value=total_rew)
        model.train_writer.add_summary(summary)
        model.train_writer.flush()

    game.init_game(seed=None)
    np.random.seed(None)
    total_rew = 0
    s_a_next = np.hstack((np.tile([game.state.copy()], (9, 1)), np.eye(9)))
    print(model.predict(s_a_next))

    model.fit(sa_pairs, np.array(targets)[np.newaxis].T)

    return total_rews



if __name__ == '__main__':
    game = Road_game()
    model = Q_approx(190)
    tot_rew = []

    for i in range(10000):
        total_rew = play_game(game, model)
        tot_rew += total_rew
        print(i)
        # plt.plot(run_avg(tot_rew,bin=300))
        # plt.savefig('fig.png')
        # plt.close()



