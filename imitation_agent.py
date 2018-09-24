import tensorflow as tf
import logging
import os
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import sklearn

from simple_car_game import *
from pdb import set_trace

GAMMA = 0.9

graph_names = ['Mean D-score for agent', 'Mean D-score for expert', 'Policy cost', 'Discriminator cost']
graphs = [[], [], [], []]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class SGDRegressor_occupancy:
    def __init__(self, xd):
        self.save_path = 'graphs/graph_imit/graph.ckpt'
        self.restore_path = 'graphs/graph_imit/graph.ckpt'

        self.states = tf.placeholder(tf.float32, shape=[None, xd - 2], name='states')
        self.action_ind = tf.placeholder(tf.int32, shape=[None, 2], name='act_inds')
        self.sa_pairs = tf.placeholder(tf.float32, shape=[None, xd], name='sa_pairs')

        self.expert_occ_measure = tf.placeholder(tf.float32, shape=(None), name='EOM')
        self.agent_occ_measure = tf.placeholder(tf.float32, shape=(None), name='OM')
        self.occ_measure_for_Q = tf.placeholder(tf.float32, shape=(None, None), name='QOM')

        self.expert_inds = tf.placeholder(tf.int32, shape=[None, 2], name='expert_inds')
        self.agent_inds = tf.placeholder(tf.int32, shape=[None, 2], name='agent_inds')

        Dw0 = tf.layers.dense(self.sa_pairs, 32, activation=tf.nn.sigmoid, use_bias=True, name='Dw0', reuse=None)
        Dw1 = tf.layers.dense(Dw0, 16, activation=tf.nn.sigmoid, use_bias=True, name='Dw1', reuse=None)
        # Dw2 = tf.layers.dense(Dw1, 16, activation=tf.nn.sigmoid, use_bias=True, name='Dw2', reuse=None)
        Dw = tf.layers.dense(Dw0, 1, activation=tf.nn.sigmoid, use_bias=True, name='Dw', reuse=None)

        self.Dw_agent = tf.reshape(tf.gather_nd(Dw, self.agent_inds), [-1, 1])
        self.Dw_expert = tf.reshape(tf.gather_nd(Dw, self.expert_inds), [-1, 1])

        self.cost_D = tf.matmul(self.agent_occ_measure, tf.log(self.Dw_agent)) + tf.matmul(self.expert_occ_measure,
                                                                                           tf.log(1.0 - self.Dw_expert))
        # self.cost_D = tf.reduce_mean(tf.log(1.0-self.Dw_agent)) + tf.reduce_mean(tf.log(self.Dw_expert))

        with tf.variable_scope('policy'):
            pi0 = tf.layers.dense(self.states, 32, activation=tf.nn.sigmoid, use_bias=True)
            pi1 = tf.layers.dense(pi0, 16, activation=tf.nn.sigmoid, use_bias=True)
            # pi2 = tf.layers.dense(pi1, 16, activation=tf.nn.sigmoid, use_bias=True)
            pi = tf.layers.dense(pi1, 9, activation=tf.nn.softmax, use_bias=True)
            self.pi = tf.gather_nd(pi, self.action_ind)

        # prediction and cost of policy
        self.H = -tf.reduce_sum(tf.multiply(self.pi, tf.log(self.pi)))
        # self.Q = tf.matmul(self.occ_measure_for_Q,tf.log(self.Dw))
        self.Q = tf.log(self.Dw_agent)
        self.piQ = tf.multiply(tf.reshape(tf.log(self.pi), [-1, 1]), self.Q)

        self.cost_pi = tf.matmul(self.agent_occ_measure, self.piQ)  # - lambda_*self.H
        # self.cost_pi = tf.reduce_mean(self.piQ)

        # ops we want to call later
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_D = optimizer.minimize(-self.cost_D)

        policy_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy')
        # print(policy_train_vars)
        self.train_pi = optimizer2.minimize(self.cost_pi, var_list=policy_train_vars)

        # self.grad_D = optimizer.compute_gradients(self.cost_D)
        # self.grad_pi = optimizer2.compute_gradients(self.cost_pi, var_list=[self.theta0])

        # start the session and initialize params
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy'))
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(init)
        # saver.restore(self.session, self.restore_path)
        self.session.graph.finalize()

    def partial_fit_D(self, aOM, eOM, sa_pairs_agent, sa_pairs_expert):
        sa_pairs = np.vstack((sa_pairs_agent, sa_pairs_expert))

        agent_inds = np.vstack((np.arange(0, len(sa_pairs_agent), 1), np.zeros((1, len(sa_pairs_agent))))).T
        expert_inds = np.vstack((np.arange(len(sa_pairs_agent), len(sa_pairs_agent) + len(sa_pairs_expert), 1),
                                 np.zeros((1, len(sa_pairs_expert))))).T

        self.session.run(self.train_D, feed_dict={self.agent_occ_measure: np.atleast_2d(aOM),
                                                  self.expert_occ_measure: np.atleast_2d(eOM),
                                                  self.sa_pairs: sa_pairs,
                                                  self.agent_inds: agent_inds,
                                                  self.expert_inds: expert_inds})

    def partial_fit_policy(self, aOM, sa_pairs_agent):
        action_inds = [game.actios.index(s[-2:].tolist()) for s in sa_pairs_agent]
        action_inds = np.vstack((np.arange(0, len(sa_pairs_agent), 1), action_inds))
        action_inds = action_inds.T
        agent_inds = np.vstack((np.arange(0, len(sa_pairs_agent), 1), np.zeros((1, len(sa_pairs_agent))))).T
        self.session.run(self.train_pi, feed_dict={self.agent_occ_measure: np.atleast_2d(aOM),
                                                   self.sa_pairs: sa_pairs_agent,
                                                   self.states: sa_pairs_agent[:, :-2],
                                                   self.action_ind: action_inds,
                                                   self.agent_inds: agent_inds})

    def predict_action_prob(self, state):
        # print('Predict action prob: ', state_action.shape)
        action_inds = [game.actios.index(s[-2:].tolist()) for s in state]
        action_inds = np.vstack((np.arange(0, len(state), 1), action_inds))
        action_inds = action_inds.T
        policy = self.session.run(self.pi, feed_dict={self.states: state[:, :-2],
                                                      self.action_ind: action_inds})
        return policy

    def comp_Dw_agent(self, sa_pairs):
        # print('Predict action prob: ', state_action.shape)
        agent_inds = np.vstack((np.arange(0, len(sa_pairs), 1), np.zeros((1, len(sa_pairs))))).T
        return self.session.run(self.Dw_agent, feed_dict={self.sa_pairs: sa_pairs,
                                                          self.agent_inds: agent_inds,
                                                          })

    def comp_Dw_expert(self, sa_pairs):
        # print('Predict action prob: ', state_action.shape)
        expert_inds = np.vstack((np.arange(0, len(sa_pairs), 1), np.zeros((1, len(sa_pairs))))).T
        return self.session.run(self.Dw_expert, feed_dict={self.sa_pairs: sa_pairs,
                                                           self.expert_inds: expert_inds})

    def save_sess(self):
        self.saver.save(self.session, self.save_path)

    def policy_loss(self, aOM, sa_pairs_agent):
        action_inds = [game.actios.index(s[-2:].tolist()) for s in sa_pairs_agent]
        action_inds = np.vstack((np.arange(0, len(sa_pairs_agent), 1), action_inds))
        action_inds = action_inds.T

        agent_inds = np.vstack((np.arange(0, len(sa_pairs_agent), 1), np.zeros((1, len(sa_pairs_agent))))).T
        #
        # pi = self.session.run(self.pi, feed_dict={self.states: sa_pairs_agent[:,:-2], self.action_ind:action_inds})
        #
        # Dw = self.session.run(self.Dw_agent, feed_dict={self.sa_pairs_agent: sa_pairs_agent, self.agent_occ_measure:aOM})
        #
        # Q = self.session.run(self.Q,
        #                  feed_dict={self.agent_occ_measure: np.atleast_2d(aOM), self.sa_pairs_agent: sa_pairs_agent,
        #                             self.states: sa_pairs_agent[:, :-2], self.action_ind: action_inds})
        #
        # piQ = self.session.run(self.piQ,
        #                      feed_dict={self.agent_occ_measure: np.atleast_2d(aOM), self.sa_pairs_agent: sa_pairs_agent,
        #                                 self.states: sa_pairs_agent[:, :-2], self.action_ind: action_inds})

        cost_pi = self.session.run(self.cost_pi,
                                   feed_dict={self.agent_occ_measure: np.atleast_2d(aOM),
                                              self.sa_pairs: sa_pairs_agent,
                                              self.states: sa_pairs_agent[:, :-2],
                                              self.action_ind: action_inds,
                                              self.agent_inds: agent_inds})

        return cost_pi

    def disc_loss(self, aOM, eOM, sa_pairs_agent, sa_pairs_expert):
        sa_pairs = np.vstack((sa_pairs_agent, sa_pairs_expert))

        agent_inds = np.vstack((np.arange(0, len(sa_pairs_agent), 1), np.zeros((1, len(sa_pairs_agent))))).T
        expert_inds = np.vstack((np.arange(len(sa_pairs_agent), len(sa_pairs_agent) + len(sa_pairs_expert), 1),
                                 np.zeros((1, len(sa_pairs_expert))))).T

        return self.session.run(self.cost_D, feed_dict={self.agent_occ_measure: np.atleast_2d(aOM),
                                                        self.expert_occ_measure: np.atleast_2d(eOM),
                                                        self.sa_pairs: sa_pairs,
                                                        self.agent_inds: agent_inds,
                                                        self.expert_inds: expert_inds})


def predict_sa_prob(state_action, traj):
    n_taken, n_diff = 0, 0
    for trajectory in traj:
        for sa in trajectory:
            if (sa == state_action).all():
                n_taken += 1
            elif (sa[:-4] == state_action[:-4]).all():
                n_diff += 1
    if (n_taken + n_diff) == 0:
        return 0
    else:
        return n_taken / (n_taken + n_diff)


def occupancy_measure(traj):
    unique_states = np.unique(np.vstack(traj)[:, :-2], axis=0).tolist()
    unique_actions = np.unique(np.vstack(traj)[:, -2:], axis=0).tolist()

    max_d = max([np.array(t).shape[0] for t in traj])

    sa_x_traj = np.zeros((max_d, len(unique_states)))
    sa_a_traj = np.zeros((len(unique_states), len(unique_actions)))

    for t in traj:

        ind_s = list(map(lambda b: unique_states.index(b), np.array(t)[:, :-2].tolist()))
        ind_a = list(map(lambda b: unique_actions.index(b), np.array(t)[:, -2:].tolist()))

        for i, sa in enumerate(t):
            sa_x_traj[i, ind_s[i]] += 1.0
            sa_a_traj[ind_s[i], ind_a[i]] += 1.0

    # set_trace()
    state_probs = np.nan_to_num(sa_x_traj.T / sa_x_traj.sum(axis=1))
    act_probs = np.nan_to_num(sa_a_traj.T / sa_a_traj.sum(axis=1))
    prob = np.dot(np.array([GAMMA ** i for i in range(max_d)]), state_probs.T)
    prob = np.repeat(prob, len(unique_actions)) * np.reshape(act_probs, (len(unique_states) * len(unique_actions)),
                                                             order='F')
    non_zero_prob = np.nonzero(prob)[0]

    ind_s = non_zero_prob // len(unique_actions)
    ind_a = non_zero_prob % len(unique_actions)

    return prob[non_zero_prob], np.array(unique_states)[ind_s], np.array(unique_actions)[ind_a]


def sample_trajectories(game, model):
    trajectories = []

    while len(trajectories) < 20:
        game.init_game(seed=None)

        s_a_pairs = []
        total_rew = 0
        t = 0
        while not game.game_over and t < 400:
            # rew, zk = game.auto_move(zk)
            st = game.state

            s_a = np.hstack((np.tile([st], (9, 1)), game.actios))
            action_probs = model.predict_action_prob(s_a)
            idx = np.random.choice(range(9), p=action_probs.ravel())
            # idx = np.argmax(action_probs.ravel())
            d_v = game.actios[idx]
            # print(action_probs, d_v)
            s_a_pairs.append(np.hstack((st, d_v)))
            rew = game.move(d_v)
            total_rew += rew
            # time.sleep(0.15)
            # print('\rVelocity: {}'.format(game.car.v), end='')
            t += 1
        trajectories.append(s_a_pairs)
    return trajectories


def delete_imgs(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def measure_perf(model, expert_traj, agent_trajs, eOM, aOM):
    agent_score = np.mean(model.comp_Dw_agent(agent_trajs[::5]))
    expert_score = np.mean(model.comp_Dw_expert(expert_traj[::5]))
    logging.debug('Mean D for expert_traj: ' + str(expert_score))
    logging.debug('Mean D for agent_traj: ' + str(agent_score))
    graphs[0].append(agent_score)
    graphs[1].append(expert_score)

    policy_loss = model.policy_loss(aOM, agent_trajs)
    disc_loss = model.disc_loss(aOM, eOM, agent_trajs, expert_traj)

    graphs[2].append(float(policy_loss))
    graphs[3].append(float(disc_loss))


def plot_graphs(path):
    try:
        if os.path.isfile(path):
            os.unlink(path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

    plt.figure(figsize=(20, 20))
    for i, graph_name in enumerate(zip(graphs, graph_names)):
        plt.subplot(221 + i)
        plt.plot(run_avg(graph_name[0]))
        plt.title(graph_name[1])
        if i < 2:
            plt.ylim((0.0, 1.0))
        plt.grid()
    plt.savefig(path)
    plt.close()


def run_avg(lst, bin=20):
    out = []
    l = len(lst)
    for i in range(l):
        out.append(np.mean(lst[max(0, i - bin):min(l - 1, i + bin)]))
    return out

def sample_trajectories_end(game, model):
    stats = []

    for j in range(200):
        game.init_game(seed=None)
        s_a_pairs = []
        total_rew = 0
        print('{}/{}'.format(j, 200))
        while not game.game_over:
            st = game.state

            # print(np.repeat(st,len(game.actios),axis=0))
            # print(game.actios)

            action_probs = model.predict_action_prob(np.hstack((np.repeat([st], len(game.actios), axis=0), game.actios)))
            idx = np.random.choice(range(9), p=action_probs.ravel())
            d_v = game.actios[idx]
            s_a_pairs.append(np.hstack((st, d_v)))
            rew = game.move(d_v)
            total_rew += rew
        stats.append((total_rew, len(s_a_pairs), int(game.collision())))

    with open('graphs/graph_imit/stats.txt', 'w') as file:
            file.writelines([", ".join([str(i) for i in t]) + "\n" for t in stats])

if __name__ == '__main__':
    expert_traj = []
    trajectories_path = 'trajectories_all/trajectories6x30/'

    for i, t in enumerate(os.listdir(trajectories_path)):
        raw_traj = np.load(trajectories_path + t)
        expert_traj.append(raw_traj)

    expert_traj = np.array(expert_traj)

    eOM, euS, euA = occupancy_measure(expert_traj)
    np.save('occ_measures/eOM.npy', eOM)
    np.save('occ_measures/euS.npy', euS)
    np.save('occ_measures/euA.npy', euA)

    # eOM, euS, euA = np.load('occ_measures/eOM.npy'), np.load('occ_measures/euS.npy'), np.load('occ_measures/euA.npy')

    model = SGDRegressor_occupancy(expert_traj[0].shape[1])

    delete_imgs('images/')
    logging.basicConfig(filename='images/road_game.log', level=logging.DEBUG)

    N_iterations = 10000

    game = Road_game()
    for i in range(N_iterations):
        print('{}/{}'.format(i, N_iterations))

        logging.debug('======Iteration #{}======'.format(i))

        agent_trajs = sample_trajectories(game, model)

        # Update the Discriminator parameters from wi to wi+1 with the gradient
        aOM, auS, auA = occupancy_measure(agent_trajs)
        model.partial_fit_D(aOM, eOM, np.hstack((auS, auA)), np.hstack((euS, euA)))
        #     # print(model.comp_Dw())

        if i % 10 == 0 and i > 0:
            plot_graphs('images/stat_graph.png')
            model.save_sess()

        model.partial_fit_policy(aOM, np.hstack((auS, auA)))

        measure_perf(model, np.hstack((euS, euA)), np.hstack((auS, auA)), eOM, aOM)
        # print(graphs)

    sample_trajectories_end(game, model)
