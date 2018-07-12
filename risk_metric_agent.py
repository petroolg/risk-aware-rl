import tensorflow as tf
from simple_car_game import *
from model import Model, hash18
# import matplotlib
# matplotlib.use("Agg")
# from matplotlib import pyplot as plt
import time
import pickle

def risk_adjusted_utility(trans_model, s, a, l):
    mtx = np.atleast_2d(list(trans_model.get(s, {}).get(a, {}).values())).astype(float)
    if len(mtx[0]) == 0:
        return 50
    mtx[:, 0] = mtx[:, 0] / np.sum(mtx[:, 0])
    H = -mtx[:,0].T.dot(np.log(mtx[:,0]))
    ERS = {}
    for key in trans_model.get(s,{}).keys():
        m = np.atleast_2d(list(trans_model.get(s,{}).get(key,{}).values())).astype(float)
        m[:, 0] = m[:,0]/np.sum(m[:,0])
        ERS[key] = m[:,0].T.dot(m[:,1])
    # return l*H - (1-l)*ERS[a]/np.max(list(ERS.values()))
    return l * H - (1 - l) * ERS[a] / 0.1

class Q_approx:

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

            summary_name = 'run{}_{}'.format(str(datetime.date(datetime.now())), str(datetime.time(datetime.now()))[:8]) + summary_name

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


    def predict(self, sa_pairs):
        return self.sess.run(self.Q, feed_dict={self.sa_pairs:np.atleast_2d(sa_pairs)})

    def predict_risk_adjusted_utility(self, sa_pairs, transition_model, p, lam):
        Q = self.sess.run(self.Q, feed_dict={self.sa_pairs:np.atleast_2d(sa_pairs)})
        risk = np.array([risk_adjusted_utility(transition_model,
                                               sa_pair[:-9],
                                               list(sa_pair[-9:]).index(1),
                                               lam)
                         for sa_pair in sa_pairs])
        return p*(1-risk) + (1-p)*Q.ravel()

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


def softmax(vec):
    return np.exp(vec) / np.sum(np.exp(vec))

def play_game(game, model, transition_model:Model, seed, risk_averse, p, l):

    sa_pairs, targets = [], []
    total_rews, collisions = [], []

    for i in range(1):
        game.init_game(seed=seed)
        total_rew = 0
        while not game.game_over:
            s_a = np.hstack((np.tile([game.state.copy()], (9,1)), np.eye(9)))

            if risk_averse:
                action_probs = model.predict_risk_adjusted_utility(s_a, transition_model, p, l)
            else:
                action_probs = model.predict(s_a)

            idx = np.random.choice(range(9), p=softmax(action_probs).ravel())
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
            transition_model.add_prob(cur_sa[:-9], idx, game.state.copy(), rew)
            # time.sleep(0.15)
        total_rews.append(total_rew)
        collisions.append(game.collision())

    model.fit(sa_pairs, np.array(targets)[np.newaxis].T)
    model.add_summary(sa_pairs, np.array(targets)[np.newaxis].T, total_rews, collisions)

    return total_rews



if __name__ == '__main__':
    game = Road_game()
    transition_model = pickle.load(open('trans_model.pckl', 'rb'))
    # transition_model = Model()

    seeds = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    hyper_p = np.linspace(0.001, 0.2, 5)
    hyper_lambda = np.linspace(0.05, 0.95, 5)
    hyperparams = list(zip(list(np.tile(hyper_p, len(hyper_lambda))), list(np.repeat(hyper_lambda, len(hyper_p)))))

    for j, params in enumerate(hyperparams[6:]):
        p, l = params
        print('Executing run #{} for hyperparams p={}, lambda={}'.format(j, p, l))
        model = Q_approx(190, summary_name='p_{:.2f}_lambda_{:.2f}'.format(p, l))

        for i in range(3000):
            seed = np.random.choice(seeds)
            total_rew = play_game(game, model, transition_model, seed, risk_averse=True, p=p, l=l)
            print('\r{}/3000'.format(i), end='')
            if i%500==0:
                with open('trans_model.pckl', 'wb') as file:
                    pickle.dump(transition_model,file)
                with open('trans_model.pckl.bk', 'wb') as file:
                    pickle.dump(transition_model,file)
                model.save_session()
        tf.reset_default_graph()