import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from simple_car_game import *
from model import Model
# import matplotlib
# matplotlib.use("Agg")
# from matplotlib import pyplot as plt
import time
import pickle
import scipy.interpolate
import multiprocessing
import argparse

GAMMA = 0.7

# Based on Policy Gradients with variance Related Risk Criteria

class PolicyGradientVariance:

    def __init__(self, xd, summary_name=''):
        with tf.variable_scope('policy_train', reuse=tf.AUTO_REUSE):
            self.save_path = 'graphs/policy_gradient/model/' + summary_name + '/model.ckpt'

            self.alpha = tf.constant(0.1)
            self.beta = tf.constant(0.01)
            self.lam = tf.constant(0.5)
            self.tau = tf.constant(2.0) # temperature for softmax policy

            self.states = tf.placeholder(tf.float32, (None, xd), name='states')
            self.actions = tf.placeholder(tf.int32, (None, 2), name='actions')
            self.reward = tf.placeholder(tf.float32, (), name='reward')
            # self.b = tf.placeholder(tf.float32, (), name='variance_bound')
            self.b = tf.constant(100.0)

            self.gain = tf.Variable(0.0, name='reward')
            self.variance = tf.Variable(0.0, name='variance')

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.variable_scope('policy_function'):
                self.policy0 = tf.layers.dense(self.states, 60, activation=tf.nn.leaky_relu, name='policy0')
                self.policy1 = tf.layers.dense(self.policy0, 30, activation=tf.nn.leaky_relu, name='policy1')
                self.policy2 = tf.layers.dense(self.policy1, 9, activation=tf.nn.leaky_relu, name='policy')
                self.policy = tf.divide(tf.exp(tf.divide(self.policy2, self.tau)), tf.reshape(tf.reduce_sum(tf.exp(tf.divide(self.policy2, self.tau)), axis=1), (-1,1)))

            with tf.name_scope('reward-update'):
                self.reward_loss = tf.scalar_mul(0.5, tf.square(tf.subtract(self.reward, self.gain)))
                self.reward_train_op = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.reward_loss,
                                                                                                 var_list=[self.gain])

            with tf.name_scope('variance-update'):
                self.variance_loss = tf.scalar_mul(0.5, tf.square(tf.subtract(tf.subtract(tf.square(self.reward), tf.square(self.gain)), self.variance)))
                self.variance_train_op = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.variance_loss,
                                                                                                   var_list=[
                                                                                                       self.variance])

            self.vars = tf.trainable_variables(scope='policy_train/policy_function')

            with tf.name_scope('policy-update'):
                self.probs = tf.gather_nd(self.policy, self.actions)
                self.Z = tf.log(self.probs)
                self.g_prime = tf.square(tf.maximum(0.0, self.variance - self.b))
                self.m = tf.square(self.reward) - tf.scalar_mul(2.0, self.gain)
                self.mult = tf.subtract(self.reward, self.lam * self.g_prime * self.m)
                self.policy_loss = tf.multiply(self.reward, self.Z)

                self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.beta)
                self.policy_train_op = self.policy_optimizer.minimize(self.policy_loss, var_list=self.vars,global_step=self.global_step)

            for var in self.vars:
                tf.summary.histogram(var.name, var)
            tf.summary.histogram("activations", self.policy)
            tf.summary.scalar('reward', self.gain)
            tf.summary.scalar('variance', self.variance)
            # tf.summary.scalar('policy_cost', self.policy_loss)
            tf.summary.scalar('variance_cost', self.variance_loss)
            tf.summary.scalar('reward_cost', self.reward_loss)

            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            summary_name = 'run{}_{}'.format(str(datetime.date(datetime.now())),
                                             str(datetime.time(datetime.now()))[:8]) + summary_name
            self.train_writer = tf.summary.FileWriter('graphs/policy_gradient/' + summary_name, self.sess.graph)
            self.saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            self.sess.run(init)

            # self.saver.restore(self.sess, self.save_path)
            # self.sess.graph.finalize()

    def fit(self, states, actions,  R):

        action_inds = np.vstack((np.zeros(len(actions)), actions)).T
        r=0
        discounted_rewards = np.zeros_like(R)
        for t in reversed(range(len(R))):
            # future discounted reward from now on
            r = R[t] + GAMMA * r
            discounted_rewards[t] = r

        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/(np.std(discounted_rewards) + 0.001)

        for state, action, reward in zip(states, action_inds, discounted_rewards):
            # policy, probs, z, g_prime, m, mult, cost = self.sess.run([self.policy, self.probs, self.Z, self.g_prime, self.m, self.mult, self.policy_loss],
            #                                                          feed_dict={self.states: np.atleast_2d(state),
            #                                                                     self.actions: np.atleast_2d(action),
            #                                                                     self.reward: reward})
            #
            # grad = self.sess.run(self.policy_optimizer.compute_gradients(loss=self.policy_loss, var_list=self.vars),
            #                      feed_dict={self.states: np.atleast_2d(state),
            #                                 self.actions: np.atleast_2d(action),
            #                                 self.reward: reward})


            self.sess.run([self.reward_train_op, self.variance_train_op, self.policy_train_op],
                          feed_dict={self.states: np.atleast_2d(state),
                                     self.actions: np.atleast_2d(action),
                                     self.reward: reward})

    def predict(self, states):
        move_idx = self.sess.run(self.policy, feed_dict={self.states: np.atleast_2d(states)})
        return move_idx

    def add_summary(self, states, actions, R, collision):

        action_inds = np.vstack((np.arange(0, len(actions), 1), actions)).T

        global_step, summary = self.sess.run([self.global_step, self.merged],
                                             feed_dict={self.states: np.atleast_2d(states),
                                                        self.reward: R,
                                                        self.actions: action_inds})

        self.train_writer.add_summary(summary, global_step=global_step)

        summary = tf.Summary()
        summary.value.add(tag='Total reward', simple_value=R)
        summary.value.add(tag='Collisions', simple_value=collision)
        self.train_writer.add_summary(summary, global_step=global_step)

    def save_session(self):
        self.saver.save(self.sess, self.save_path)


def play_game(game, model, **kwargs):
    states, actions = [], []
    total_rew, collision = 0, False

    game.init_game(seed=kwargs.get('seed', None))
    total_rew = [0]

    T = 0

    while not game.game_over:
        states.append(game.state.copy())
        action_probs = model.predict(game.state.copy())
        idx = np.random.choice(range(9), p=action_probs.ravel())
        actions.append(idx)
        print(("{:.2f} "*9).format(*action_probs.ravel()))
        d_v = game.actios[idx]
        rew = game.move(d_v)
        total_rew.append(total_rew[-1]+rew)
        collision = game.collision()

    model.fit(states, actions, total_rew[1:])
    model.add_summary(states, actions, total_rew[-1], collision)



def perform_experiment(kwargs):
    print(kwargs)

    game = Road_game()
    # game = TwentyStateGame(0.0)


    print('Executing run #{} for hyperparams {}'.format(kwargs['j'], kwargs))
    summary_name = '{}'.format(kwargs)
    model = PolicyGradientVariance(188, summary_name=summary_name)

    learning_steps = 10000

    for i in range(learning_steps):
        total_rew = play_game(game, model, **kwargs)
        print('\r{}/{}'.format(i, learning_steps), end='')
        if i % 10 == 0:
            model.save_session()



if __name__ == '__main__':
    hyper_b = np.linspace(2.0, 100.0, 3)
    hyperparams = [{'b': b, 'j': j} for j, b in enumerate(hyper_b)]

    # pool = multiprocessing.Pool(len(hyperparams))
    # pool.map_async(perform_experiment, hyperparams)
    #
    # pool.close()
    # pool.join()

    perform_experiment(hyperparams[0])

