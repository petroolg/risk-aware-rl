import argparse
import sys
import numpy as np
import pickle
from datetime import datetime

import tensorflow as tf
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MDP"))

from environment import Road_game
from model import Model

GAMMA = 0.9


class QApprox:

    def __init__(self, xd, risk_metric=None, summary_name=''):
        with tf.variable_scope('Q_train', reuse=tf.AUTO_REUSE):

            self.restore_path = "graphs/risk_metric/model/run2018-10-07_16:43:58{'j': 1, 'type': 'pessimistic'}/model.ckpt"

            self.sa_pairs = tf.placeholder(tf.float64, (None, xd), name='sa_pairs')
            self.target = tf.placeholder(tf.float64, (None, 1), name='target')

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.variable_scope('Q_function'):
                self.Q0 = tf.layers.dense(self.sa_pairs, 128, activation=tf.nn.leaky_relu, name='Q0',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.contrib.layers.xavier_initializer()
                                          )
                self.Q1 = tf.layers.dense(self.Q0, 64, activation=tf.nn.leaky_relu, name='Q1',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.contrib.layers.xavier_initializer()
                                          )
                self.Q = tf.layers.dense(self.Q1, 1, name='Q',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         bias_initializer=tf.contrib.layers.xavier_initializer()
                                         )

            self.cost = tf.reduce_mean(tf.squared_difference(self.Q, self.target))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
            self.train_op = self.optimizer.minimize(self.cost, global_step=self.global_step)

            vars = tf.trainable_variables(scope='Q_function')

            for var in vars:
                tf.summary.histogram(var.name, var)
            tf.summary.histogram("activations", self.Q)
            tf.summary.scalar('cost', self.cost)

            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            summary_name = 'run{}_{}'.format(str(datetime.now().date()), str(datetime.now().time())[:8]) + summary_name
            self.save_path = 'graphs/q_learning/' + summary_name + '/model.ckpt'
            self.train_writer = tf.summary.FileWriter('graphs/q_learning/' + summary_name, self.sess.graph)
            self.saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            self.sess.run(init)
            # self.saver.restore(self.sess, self.restore_path)
            # self.sess.graph.finalize()

            if risk_metric == 'entropy':
                self.risk_metric = self.entropy_risk
            elif risk_metric == 'cvar':
                self.risk_metric = self.c_value_at_risk
            elif risk_metric == 'mean':
                self.risk_metric = self.mean

    def fit(self, sa_pairs, target):
        self.sess.run(self.train_op, feed_dict={self.sa_pairs: np.atleast_2d(sa_pairs),
                                                self.target: np.atleast_2d(target)})

    def predict(self, sa_pairs, game, transition_model, with_risk=False, **kwargs):
        Q, step = self.sess.run([self.Q, self.global_step], feed_dict={self.sa_pairs: np.atleast_2d(sa_pairs)})

        if kwargs.get('risk_metric', None) and with_risk:
            risk = np.array([self.risk_metric(game,
                                              transition_model,
                                              sa_pair[:-9],
                                              list(sa_pair[-9:]).index(1),
                                              **kwargs)
                             for sa_pair in sa_pairs])

            return np.exp(-step / 200) * kwargs['p'] * risk + (1 - kwargs['p'] * np.exp(-step / 200)) * Q.ravel()
            # print(('{:.2f} '*9).format(*risk))
            # return risk
        else:
            return Q.ravel()

    def add_summary(self, sa_pairs, target, total_rews, collisions):

        global_step, summary = self.sess.run([self.global_step, self.merged],
                                             feed_dict={self.sa_pairs: np.atleast_2d(sa_pairs),
                                                        self.target: np.atleast_2d(target)})
        self.train_writer.add_summary(summary, global_step=global_step)

        summary = tf.Summary()
        summary.value.add(tag='Total reward', simple_value=np.mean(total_rews))
        summary.value.add(tag='Collisions', simple_value=np.mean(collisions))
        self.train_writer.add_summary(summary, global_step=global_step)

    def save_session(self):
        self.saver.save(self.sess, self.save_path)

    def entropy_risk(self, game, trans_model, s, a, **kwargs):

        prob, rews = self.multi_step_distr(s, a, trans_model, game, GAMMA, kwargs['n_steps'])

        h = -prob.dot(np.log(prob.T))
        e_r = prob.dot(rews.T)

        risk = kwargs['l'] * h - (1 - kwargs['l']) * e_r / 0.5

        return -float(risk)

    def c_value_at_risk(self, game, trans_model, s, a, **kwargs):
        prob, rews = self.multi_step_distr(s, a, trans_model, game, GAMMA, kwargs['n_steps'])

        if len(prob.ravel()) == 1:
            return float(rews)
        cumulative_dist = np.cumsum(prob, axis=1)
        value_at_risk = rews[cumulative_dist >= kwargs['alpha']][0]

        c_var = prob[rews <= value_at_risk].dot(rews[rews <= value_at_risk])

        return float(c_var)

    def mean(self, game, trans_model, s, a, **kwargs):
        prob, rews = self.multi_step_distr(s, a, trans_model, game, GAMMA, kwargs['n_steps'])

        e_r = prob.dot(rews.T)

        return float(e_r)

    def multi_step_distr(self, s, a, trans_model, game, gamma, n_steps):
        p, r = self.multi_step_distr_recursion(s, a, trans_model, game, gamma, 1, n_steps)

        prob, rews = [], []

        for rew in np.sort(np.unique(r)):
            rews.append(rew)
            prob.append(np.sum(p[r == rew]))

        return np.atleast_2d(prob), np.atleast_2d(rews)

    def multi_step_distr_recursion(self, s, a, trans_model, game, gamma, n, n_steps):

        mtx, s_primes = trans_model.get_distr(s, a, game)

        if n == n_steps:
            if len(mtx) == 0:
                return np.array([[1.0]]), np.array([[game.reward_dict['default']]])
            else:

                return mtx[:, 0], mtx[:, 1]
        else:
            next_p, next_r = np.array([[]]), np.array([[]])
            if len(mtx) == 0:
                next_p_s = [[1.0]]
                next_r_s = [[game.reward_dict['default']]]

                next_p = np.append(next_p, next_p_s)
                next_r = np.append(next_r, next_r_s)
            else:
                for i, s_prime in enumerate(s_primes):
                    s_a_next = np.hstack((np.tile([np.frombuffer(s_prime, dtype=int)], (9, 1)), np.eye(9)))
                    s_a_next = self.predict(s_a_next, game, transition_model, with_risk=True)
                    a_prime = np.argmax(s_a_next)

                    p, r = self.multi_step_distr_recursion(s_prime, a_prime, transition_model, game, gamma, n + 1,
                                                           n_steps)

                    next_r_s = mtx[i, 1] + gamma * r
                    next_p_s = mtx[i, 0] * p
                    next_p = np.append(next_p, next_p_s)
                    next_r = np.append(next_r, next_r_s)

        return next_p, next_r


def softmax(vec, tau=0.2):
    return np.exp(vec / tau) / np.sum(np.exp(vec / tau))


def play_game(game, model, transition_model, **kwargs):
    sa_pairs, targets = [], []
    total_rews, collisions = [], []

    for i in range(10):
        game.init_game()
        total_rew = 0
        while not game.game_over:
            s_a = np.hstack((np.tile([game.state.copy()], (9, 1)), np.eye(9)))
            action_probs = model.predict(s_a, game, transition_model, with_risk=True, **kwargs)
            # print(action_probs)

            idx = np.random.choice(range(9), p=softmax(action_probs).ravel())
            d_v = game.actios[idx]

            act_one_hot = np.zeros((9,))
            act_one_hot[game.actios.index(d_v)] = 1.0
            cur_sa = np.hstack((game.state.copy(), act_one_hot))
            rew = game.move(d_v)
            total_rew += rew

            s_a_next = np.hstack((np.tile([game.state.copy()], (9, 1)), np.eye(9)))
            s_a_next = model.predict(s_a_next, game, transition_model, with_risk=True, **kwargs)
            # print(s_a_next)
            if not game.game_over:
                trgt = rew + GAMMA * np.max(s_a_next)
            else:
                trgt = rew
            sa_pairs.append(cur_sa)
            targets.append(trgt)
            if kwargs.get("learn_model"):
                transition_model.add_prob(cur_sa[:-9], idx, game.state.copy(), game.event())

        total_rews.append(total_rew)
        collisions.append(game.collision())
        model.fit(sa_pairs, np.array(targets)[np.newaxis].T)

    model.add_summary(sa_pairs, np.array(targets)[np.newaxis].T, total_rews, collisions)

    return total_rews


def perform_experiment(kwargs, transition_model):
    print(kwargs)

    game = Road_game(n_steps=30)

    risk_metric = kwargs.get('risk_metric', None)
    print('Executing for hyperparameters {}'.format(kwargs))

    summary_name = '{}'.format(kwargs)
    model = QApprox(197, risk_metric=risk_metric, summary_name=summary_name)
    learning_steps = 1000

    for i in range(learning_steps):

        play_game(game, model, transition_model, **kwargs)

        print('\r{}/{}'.format(i, learning_steps), end='')

        if i % 10 == 0 and i > 0:
            model.save_session()
            if kwargs.get("learn_model"):
                with open(kwargs.get("path_to_model"), 'wb') as file:
                    pickle.dump(transition_model, file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("-m", dest='mode', type=str, default='model_free', choices=["model_free", "model_based"],
                        help='Mode of Q-learning: [model_free, model_based]')
    parser.add_argument("-mp", dest='path_to_model', type=str, default='trans_model.pckl',
                        help="Path to empiric model of environment.")
    parser.add_argument("-rm", dest='risk_metric', type=str, default=None, required=False,
                        choices=["entropy", "cvar", "mean"],
                        help='Risk metric to use for model-based Q-learning: [entropy, cvar]')
    parser.add_argument("-lm", dest='learn_model', type=bool, default=True, required=False, choices=[True, False],
                        help='Save transitions to model: [True, False]')
    parser.add_argument("--lambda", dest="risk_mixing_proportion", type=float, required=False, default=0.5,
                        help="Risk mixing proportion for model-based risk-aware Q-learning.")
    parser.add_argument("--alpha", dest="cvar_parameter", type=float, required=False, default=0.2,
                        help="CVaR parameter alpha.")
    parser.add_argument("--beta", dest="entropy_parameter", type=float, required=False, default=0.3,
                        help="Entropy-based risk metric parameter beta.")
    parser.add_argument("--n_steps", dest="n_steps", type=int, required=False, default=3,
                        help="Number of steps for return distribution estimation.")

    args = parser.parse_args()
    model_based = True if args.mode == "model_based" else False
    path_to_model = args.path_to_model
    risk_metric = args.risk_metric
    learn_model = args.learn_model

    # Validate input parameters
    if not model_based and risk_metric:
        raise ValueError("Either choose model-based mode or leave risk metric unspecified. "
                         "Risk-aware Q-learning run in model-based mode only.")

    # Load transition model if any
    if (model_based or learn_model) and os.path.exists(path_to_model):
        transition_model = pickle.load(open(path_to_model, 'rb'))
    else:
        transition_model = Model()

    # Prepare parameters for experiments
    if not model_based:
        hyperparams = {'model_based': False,
                       'learn_model': learn_model,
                       'path_to_model': path_to_model}

    else:

        if risk_metric == 'entropy':
            hyperparams = {'model_based': True,
                           'p': args.risk_mixing_proportion,
                           'l': args.entropy_parameter,
                           'n_steps': args.n_steps,
                           'risk_metric': 'entropy',
                           'learn_model': learn_model,
                           'path_to_model': path_to_model}

        elif risk_metric == 'cvar':
            hyperparams = {'model_based': True,
                           'p': args.risk_mixing_proportion,
                           'alpha': args.cvar_parameter,
                           'n_steps': args.n_steps,
                           'risk_metric': 'cvar',
                           'learn_model': learn_model,
                           'path_to_model': path_to_model}

        else:
            hyperparams = {'model_based': True,
                           'p': 1.0,
                           'n_steps': args.n_steps,
                           'risk_metric': 'mean',
                           'learn_model': learn_model,
                           'path_to_model': path_to_model}

    perform_experiment(hyperparams, transition_model)
