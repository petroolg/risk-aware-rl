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

GAMMA = 0.95
state_size = 188
action_size = 9


# Based on Policy Gradients with variance Related Risk Criteria

class PolicyGradient:

    def __init__(self, summary_name='', risk_metric=None):
        with tf.variable_scope('policy_train', reuse=tf.AUTO_REUSE):
            alpha = 0.5
            beta = 0.001
            self.b = 5.0
            self.lambda_ = 0.01

            self.risk_metric = risk_metric

            self.states = tf.placeholder(tf.float32, (None, state_size), name='states')
            self.actions = tf.placeholder(tf.int32, (None, action_size), name='actions')
            self.reward = tf.placeholder(tf.float32, (None,), name='reward')
            self.sample_reward = tf.placeholder(tf.float32, (), name='sample_reward')
            self.sample_variance = tf.placeholder(tf.float32, (), name='sample_variance')

            self.gain = tf.Variable(0.0, name='reward')
            self.variance = tf.Variable(0.0, name='variance')

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.variable_scope('policy_function'):
                self.policy0 = tf.layers.dense(self.states, 300, activation=tf.nn.leaky_relu, name='policy0',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                               bias_initializer=tf.contrib.layers.xavier_initializer())
                self.policy1 = tf.layers.dense(self.policy0, 100, activation=tf.nn.leaky_relu, name='policy1',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                               bias_initializer=tf.contrib.layers.xavier_initializer())
                self.policy = tf.layers.dense(self.policy1, action_size, activation=tf.nn.softmax, name='policy',
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              bias_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope('reward-update'):
                self.reward_loss = tf.losses.mean_squared_error(self.gain, self.sample_reward)
                self.reward_train_op = tf.train.AdamOptimizer(learning_rate=alpha).minimize(self.reward_loss)
            with tf.name_scope('variance-update'):
                self.variance_loss = tf.losses.mean_squared_error(self.variance, self.sample_variance)
                self.variance_train_op = tf.train.AdamOptimizer(learning_rate=alpha).minimize(self.variance_loss)

            self.vars = tf.trainable_variables(scope='policy_train/policy_function')

            with tf.name_scope('policy-update'):
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.policy, labels=self.actions)

                self.policy_loss = tf.reduce_mean(neg_log_prob * self.reward)

                self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=beta)
                self.policy_train_op = self.policy_optimizer.minimize(self.policy_loss, var_list=self.vars,
                                                                      global_step=self.global_step)

            for var in self.vars:
                tf.summary.histogram(var.name, var)
            tf.summary.histogram("activations", self.policy)
            tf.summary.scalar('reward', self.gain)
            tf.summary.scalar('variance', self.variance)
            tf.summary.scalar('policy_cost', self.policy_loss)

            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            summary_name = 'run{}_{}'.format(str(datetime.date(datetime.now())),
                                             str(datetime.time(datetime.now()))[:8]) + summary_name
            self.save_path = 'graphs/policy_gradient/model/' + summary_name + '/model.ckpt'
            self.train_writer = tf.summary.FileWriter('graphs/policy_gradient/' + summary_name, self.sess.graph)
            self.saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            self.sess.run(init)

            # self.saver.restore(self.sess, self.save_path)
            # self.sess.graph.finalize()

    def fit(self, states, actions, rewards):

        cumulated_rewards = [PolicyGradient.compute_values(rew) for rew in rewards]

        total_rewards = np.array([rew[0] for rew in cumulated_rewards])
        sample_mean_reward, sample_variance = np.mean(total_rewards), np.mean((total_rewards-np.mean(total_rewards))**2)
        self.sess.run([self.reward_train_op, self.variance_train_op], feed_dict={self.sample_reward: sample_mean_reward,self.sample_variance: sample_variance})

        cumulated_rewards = np.hstack(cumulated_rewards)

        if self.risk_metric == 'variance':

            penalized_rewards = np.array([r - self.lambda_*np.maximum(0.0, sample_variance-self.b)*abs(r-sample_mean_reward) for r in cumulated_rewards])

            normalized_rewards = (penalized_rewards - np.mean(penalized_rewards))/(np.std(penalized_rewards) + 0.001)
            self.sess.run(self.policy_train_op, feed_dict={self.states: states,
                                                           self.actions: actions,
                                                           self.reward: normalized_rewards})
        else:
            # print(cumulated_rewards, np.mean(cumulated_rewards), np.std(cumulated_rewards))
            normalized_rewards = (cumulated_rewards - np.mean(cumulated_rewards))/(np.std(cumulated_rewards) + 0.001)
            self.sess.run(self.policy_train_op, feed_dict={self.states: states,
                                                           self.actions: actions,
                                                           self.reward: normalized_rewards})

    def predict(self, states):
        return self.sess.run(self.policy, feed_dict={self.states: np.atleast_2d(states)})

    def add_summary(self, states, actions, rewards, collision):
        cumulated_rewards = np.hstack([PolicyGradient.compute_values(rew) for rew in rewards])
        normalized_rewards = (cumulated_rewards - np.mean(cumulated_rewards)) / np.std(cumulated_rewards + 0.001)
        global_step, summary = self.sess.run([self.global_step, self.merged],
                                             feed_dict={self.states: np.atleast_2d(states),
                                                        self.reward: normalized_rewards,
                                                        self.actions: actions})

        self.train_writer.add_summary(summary, global_step=global_step)

        summary = tf.Summary()
        summary.value.add(tag='Total reward', simple_value=np.mean([sum(r) for r in rewards]))
        summary.value.add(tag='Collisions', simple_value=np.mean(collision))
        self.train_writer.add_summary(summary, global_step=global_step)

    def save_session(self):
        self.saver.save(self.sess, self.save_path)

    @staticmethod
    def compute_values(R):
        r = 0
        discounted_rewards = np.zeros_like(R)
        for t in reversed(range(len(R))):
            r = R[t] + GAMMA * r
            discounted_rewards[t] = r

        return discounted_rewards


def play_game(game, model, **kwargs):
    states, actions, rewards, collisions = [], [], [], []
    n_episodes = 20
    for _ in range(n_episodes):
        episode_states, episode_actions, episode_rewards = [], [], []

        game.init_game(seed=kwargs.get('seed', None))

        state = game.state.copy()
        while not game.game_over:
            action_probability_distribution = model.predict(state)

            action = np.random.choice(range(action_size),
                                      p=action_probability_distribution.ravel())

            reward = game.move(game.actios[action])
            new_state = game.state.copy()
            episode_states.append(state)
            action_ = np.zeros(action_size)
            action_[action] = 1
            episode_actions.append(action_)
            episode_rewards.append(reward)
            state = new_state
        collisions.append(game.collision())
        states.append(episode_states)
        actions.append(episode_actions)
        rewards.append(episode_rewards)

    states = np.vstack(states)
    actions = np.vstack(actions)

    model.fit(states, actions, rewards)
    model.add_summary(states, actions, rewards, collisions)


def perform_experiment(kwargs):
    print(kwargs)

    game = Road_game()
    # game = TwentyStateGame(0.0)

    print('Executing run #{} for hyperparams {}'.format(kwargs['j'], kwargs))
    summary_name = '{}'.format(kwargs)
    model = PolicyGradient(summary_name=summary_name, risk_metric=kwargs.get('risk_metric', None))

    learning_steps = 10000

    for i in range(learning_steps):
        total_rew = play_game(game, model, **kwargs)
        print('\r{}/{}'.format(i, learning_steps), end='')
        if i % 10 == 0:
            model.save_session()


if __name__ == '__main__':
    hyper_b = np.linspace(2.0, 100.0, 3)
    hyperparams = [{'b': b, 'j': j, 'risk_metric':'variance'} for j, b in enumerate(hyper_b)]

    pool = multiprocessing.Pool(len(hyperparams))
    pool.map_async(perform_experiment, hyperparams)

    pool.close()
    pool.join()

    # perform_experiment(hyperparams[0])
