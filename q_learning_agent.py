import argparse

import tensorflow as tf

from simple_car_game import *

# import matplotlib
# matplotlib.use("Agg")
# from matplotlib import pyplot as plt

global transition_model

GAMMA = 0.9
model_name = 'trans_model_safe.pckl'

class QApprox:

    def __init__(self, xd, risk_metric=None, summary_name=''):
        with tf.variable_scope('Q_train', reuse=tf.AUTO_REUSE):
            self.save_path = 'graphs/risk_metric/model/' + summary_name + '/model.ckpt'

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

            if risk_metric == 'entropy':
                self.risk_metric = self.entropy_risk
            elif risk_metric == 'mean_deviation':
                self.risk_metric = self.mean_deviation_risk
            elif risk_metric == 'cvar':
                self.risk_metric = self.c_value_at_risk

    def fit(self, sa_pairs, target):
        self.sess.run(self.train_op,feed_dict={self.sa_pairs: np.atleast_2d(sa_pairs),
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

            return np.exp(-step / 3000) * kwargs['p'] * risk + (1 - kwargs['p'] * np.exp(-step / 3000)) * Q.ravel()
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
        risk = ER - kwargs['b'] * prob.dot((rews - ER).T ** kwargs['ps']) ** (1 / kwargs['ps'])

        return float(risk)

    def mean_semi_deviation_risk(self, game, trans_model, s, a, **kwargs):
        prob, rews = self.multi_step_distr(s, a, trans_model, game, GAMMA, kwargs['n_steps'])

        ER = prob.dot(rews.T)
        devs = np.array([d if d > 0 else d for d in rews - ER])
        risk = ER - kwargs['b'] * prob.dot(devs.T ** kwargs['ps']) ** (1 / kwargs['ps'])

        return float(risk)

    def c_value_at_risk(self, game, trans_model, s, a, **kwargs):
        prob, rews = self.multi_step_distr(s, a, trans_model, game, GAMMA, kwargs['n_steps'])

        if len(prob.ravel()) == 1:
            return float(rews)
        cumulative_dist = np.cumsum(prob, axis=1)
        value_at_risk = rews[cumulative_dist >= kwargs['alpha']][0]

        c_var = prob[rews <= value_at_risk].dot(rews[rews <= value_at_risk])

        return float(c_var)

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
                    s_a_next = self.predict(s_a_next, game, transition_model, with_risk=True)
                    a_prime = np.argmax(s_a_next)

                    p, r = self.multi_step_distr_recursion(s_prime, a_prime, transition_model, game, gamma, n + 1,
                                                           n_steps)

                    next_r_s = mtx[i, 1] + gamma * r
                    next_p_s = mtx[i, 0] * p
                    next_p = np.append(next_p, next_p_s)
                    next_r = np.append(next_r, next_r_s)

        return next_p, next_r


def softmax(vec, tau=0.01):
    return np.exp(vec/tau) / np.sum(np.exp(vec/tau))


def play_game(game, model, transition_model, **kwargs):
    sa_pairs, targets = [], []
    total_rews, collisions = [], []

    for i in range(10):
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

            action_probs = model.predict(s_a, game, transition_model, with_risk=True, **kwargs)

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

    risk_metric = kwargs.get('risk_metric', None)
    print('Executing run #{} for hyperparams {}'.format(kwargs['j'], kwargs))
    summary_name = '{}'.format(kwargs)
    model = QApprox(197, risk_metric=risk_metric, summary_name=summary_name)
    seeds = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    learning_steps = 1000

    for i in range(learning_steps):
        seed = np.random.choice(seeds)
        kwargs['seed'] = seed
        total_rew = play_game(game, model, transition_model, **kwargs)

        print('\r{}/{}'.format(i, learning_steps), end='')

        if i % 50 == 0:
            model.save_session()
            # with open(model_name, 'wb') as file:
            #     pickle.dump(transition_model, file)
            # with open(model_name + '.bk', 'wb') as file:
            #     pickle.dump(transition_model, file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Experiment name')
    parser.add_argument('experiment_name', type=str, default=None,
                        help='EXPERIMENT NAME: [entropy, cvar, mean_dev]')

    args = parser.parse_args()
    name = args.experiment_name


    transition_model = pickle.load(open(model_name, 'rb'))

    if name == 'entropy':
        hyper_p = np.linspace(0.1, 0.6, 3)
        hyper_lambda = np.linspace(0.05, 0.6, 3)
        hyperparams = list(zip(list(np.tile(hyper_p, len(hyper_lambda))), list(np.repeat(hyper_lambda, len(hyper_p)))))
        hyperparams = [{'p': p_l[0], 'l': p_l[1], 'j': j, 'n_steps': 2, 'risk_metric': 'entropy'} for j, p_l in
                       enumerate(hyperparams)]
    # completed
    elif name == 'cvar':
        hyper_p = np.linspace(0.3, 0.6, 3)
        hyper_alpha = np.linspace(0.4, 0.8, 3)
        hyperparams = list(zip(list(np.tile(hyper_p, len(hyper_alpha))), list(np.repeat(hyper_alpha, len(hyper_p)))))
        hyperparams = [{'p': p_a[0], 'alpha': p_a[1], 'j': j, 'n_steps': 2, 'risk_metric': 'cvar'} for j, p_a in
                       enumerate(hyperparams)]


    elif name == 'mean_deviation':
        hyper_p = np.linspace(0.6, 1.0, 3)
        hyper_b = np.linspace(0.1, 3.0, 3)
        hyperparams = list(zip(list(np.tile(hyper_p, len(hyper_b))), list(np.repeat(hyper_b, len(hyper_p)))))
        hyperparams = [{'p': p_b[0], 'ps': 2, 'b': p_b[1], 'j': j, 'n_steps': 4, 'risk_metric': 'mean_deviation'} for
                       j, p_b in
                       enumerate(hyperparams)]

    elif name == 'model_based':
        hyperparams = [{'j': i, 'type': 'model_based'} for i in range(10)]

    else:
        hyperparams = [{'j': i} for i in range(10)]

    pool = multiprocessing.Pool(len(hyperparams))
    pool.map_async(perform_experiment, hyperparams)

    pool.close()
    pool.join()

    # perform_experiment(hyperparams[0])
