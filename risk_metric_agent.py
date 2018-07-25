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

global transition_model

GAMMA = 0.95


class QApprox:

    def __init__(self, xd, summary_name=''):
        with tf.variable_scope('Q_train', reuse=tf.AUTO_REUSE):
            self.save_path = 'graphs/risk_metric/model/' + summary_name + '/model.ckpt'

            self.sa_pairs = tf.placeholder(tf.float64, (None, xd), name='sa_pairs')
            self.target = tf.placeholder(tf.float64, (None, 1), name='target')

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.variable_scope('Q_function'):
                self.Q0 = tf.layers.dense(self.sa_pairs, 60, activation=tf.nn.leaky_relu, name='Q0')
                self.Q1 = tf.layers.dense(self.Q0, 30, activation=tf.nn.leaky_relu, name='Q1')
                self.Q = tf.layers.dense(self.Q1, 1, name='Q')

            self.cost = tf.reduce_mean(tf.squared_difference(self.Q, self.target))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            self.train_op = self.optimizer.minimize(self.cost, global_step=self.global_step)

            vars = tf.trainable_variables(scope='Q_function')

            for var in vars:
                tf.summary.histogram(var.name, var)
            tf.summary.histogram("activations", self.Q)
            tf.summary.scalar('cost', self.cost)

            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            summary_name = 'run{}_{}'.format(str(datetime.date(datetime.now())),
                                             str(datetime.time(datetime.now()))[:8]) + summary_name
            self.train_writer = tf.summary.FileWriter('graphs/risk_metric/' + summary_name, self.sess.graph)
            self.saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            self.sess.run(init)
            # self.saver.restore(self.sess, self.save_path)
            # self.sess.graph.finalize()

    def fit(self, sa_pairs, target):
        global_step, summary, _ = self.sess.run([self.global_step, self.merged, self.train_op],
                                                feed_dict={self.sa_pairs: np.atleast_2d(sa_pairs),
                                                           self.target: np.atleast_2d(target)})

    def predict(self, sa_pairs, game, transition_model, **kwargs):
        Q = self.sess.run(self.Q, feed_dict={self.sa_pairs: np.atleast_2d(sa_pairs)})

        if kwargs.get('risk_metric', None):
            risk = np.array([self.entropy_risk(game, transition_model, sa_pair[:-9], list(sa_pair[-9:]).index(1),
                                               **kwargs)
                             for sa_pair in sa_pairs])

            return kwargs['p'] * risk + (1 - kwargs['p']) * Q.ravel()
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

    def mean_deviation_risk(self, game, trans_model, s, a, **kwargs):
        prob, rews = self.multi_step_distr(s, a, trans_model, game, GAMMA, kwargs['n_steps'])

        ER = prob.dot(rews.T)
        risk = ER + kwargs['b'] * prob.dot((rews - ER) ** kwargs['p']) ** (1 / kwargs['p'])

        return float(risk)

    def mean_semi_deviation_risk(self, game, trans_model, s, a, **kwargs):
        prob, rews = self.multi_step_distr(s, a, trans_model, game, GAMMA, kwargs['n_steps'])

        ER = prob.dot(rews.T)
        devs = np.array([d if d > 0 else d for d in rews - ER])
        risk = ER + kwargs['b'] * prob.dot(devs ** kwargs['p']) ** (1 / kwargs['p'])

        return float(risk)

    def c_value_at_risk(self, game, trans_model, s, a, **kwargs):
        prob, rews = self.multi_step_distr(s, a, trans_model, game, GAMMA, kwargs['n_steps'])
        distribution = scipy.interpolate.interp1d(np.cumsum(prob), rews, bounds_error=False,
                                                  fill_value='extrapolate')
        value_at_risk = distribution(kwargs['alpha'])
        c_var = prob[rews <= value_at_risk].dot(rews[rews <= value_at_risk])

        return c_var

    def multi_step_distr(self, s, a, trans_model, game, gamma, n_steps):
        p, r = self.multi_step_distr_recursion(s, a, trans_model, game, gamma, 0, n_steps)

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
                    s_a_next = self.predict(s_a_next, game, transition_model)
                    a_prime = np.argmax(s_a_next)

                    p, r = self.multi_step_distr_recursion(s_prime, a_prime, transition_model, game, gamma, n + 1,
                                                           n_steps)

                    next_r_s = mtx[i, 1] + gamma * r
                    next_p_s = mtx[i, 0] * p
                    next_p = np.append(next_p, next_p_s)
                    next_r = np.append(next_r, next_r_s)

        return next_p, next_r


def softmax(vec):
    return np.exp(vec) / np.sum(np.exp(vec))


def play_game(game, model, transition_model, **kwargs):
    sa_pairs, targets = [], []
    total_rews, collisions = [], []

    for i in range(1):
        game.init_game(seed=kwargs.get('seed', None))
        total_rew = 0
        while not game.game_over:
            s_a = np.hstack((np.tile([game.state.copy()], (9, 1)), np.eye(9)))

            # [[1, 1],
            #  [1, 0],
            #  [1, -1],
            #  [0, 1],
            #  [0, 0],
            #  [0, -1],
            #  [-1, 1],
            #  [-1, 0],
            #  [-1, -1]]

            action_probs = model.predict(s_a, game, transition_model, **kwargs)

            idx = np.random.choice(range(9), p=softmax(action_probs).ravel())
            d_v = game.actios[idx]

            act_one_hot = np.zeros((9,))
            act_one_hot[game.actios.index(d_v)] = 1.0
            cur_sa = np.hstack((game.state.copy(), act_one_hot))
            rew = game.move(d_v)
            total_rew += rew

            s_a_next = np.hstack((np.tile([game.state.copy()], (9, 1)), np.eye(9)))
            s_a_next = model.predict(s_a_next, game, transition_model, **kwargs)
            # print(s_a_next)
            trgt = rew + GAMMA * np.max(s_a_next)
            sa_pairs.append(cur_sa)
            targets.append(trgt)
            transition_model.add_prob(cur_sa[:-9], idx, game.state.copy(), game.event())
            # time.sleep(0.15)
        total_rews.append(total_rew)
        collisions.append(game.collision())

    model.fit(sa_pairs, np.array(targets)[np.newaxis].T)
    model.add_summary(sa_pairs, np.array(targets)[np.newaxis].T, total_rews, collisions)

    return total_rews


def perform_experiment(kwargs):
    print(kwargs)

    global transition_model

    game = Road_game()
    print('Executing run #{} for hyperparams p={}, lambda={}'.format(kwargs['j'], kwargs['p'], kwargs['l']))
    model = QApprox(190, summary_name='{}_{}_p_{:.2f}_lambda_{:.2f}'.format(kwargs['risk_metric'], kwargs['n_steps'],
                                                                            kwargs['p'], kwargs['l']))
    seeds = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    learning_steps = 8000

    for i in range(learning_steps):
        seed = np.random.choice(seeds)
        kwargs['seed'] = seed
        total_rew = play_game(game, model, transition_model, **kwargs)
        print('\r{}/{}'.format(i, learning_steps), end='')
        if i % 10 == 0:
            model.save_session()


if __name__ == '__main__':
    model_name = 'trans_model_safe.pckl'
    transition_model = pickle.load(open(model_name, 'rb'))

    hyper_p = np.linspace(0.1, 0.6, 3)
    hyper_lambda = np.linspace(0.05, 0.95, 3)
    risk_metrics = ['entropy_risk', 'mean_deviation', 'cvar']
    hyperparams = list(zip(list(np.tile(hyper_p, len(hyper_lambda))), list(np.repeat(hyper_lambda, len(hyper_p)))))
    hyperparams = [{'p': p_l[0], 'l': p_l[1], 'j': j, 'n_steps': 2, 'risk_metric': risk_metrics[0]} for j, p_l in
                   enumerate(hyperparams)]

    # pool = multiprocessing.Pool(9)
    # pool.map_async(perform_experiment, hyperparams)
    #
    # pool.close()
    # pool.join()

    perform_experiment(hyperparams[0])
