import numpy as np
from matplotlib import pyplot as plt

# Based on Policy Gradients with variance Related Risk Criteria

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

class TwentyStateGame:

    def __init__(self, var_bound):
        self.state_length = 22
        self.state = np.zeros(self.state_length)  # discrete states
        self.position = 0
        self.speed = 0
        self.theta = np.random.random_sample((self.state_length,3))/self.state_length  # parameters
        self.actions = np.array([-1, 0, 1])
        self.pitfall = False

        self.G = 0.0  # estimation of total reward
        self.Var = 0.0  # estimation of variance of reward
        self.var_bound = var_bound  # threshold of variance
        self.alpha_step = 0.01  # step of gradient ascent
        self.beta_step = 0.01  # step of gradient ascent
        self.lam = 0.1  # penalization, related to the approximation of COP solution, equations (9) and (10)
        self.eps = 0.0001  # epsilon for epsilon-constrained softmax policy

    # Mean reward, variance and parameter update function, equations (13)
    def update(self, R, zk, type='Var', update_theta=True):
        self.G += self.alpha_step * (R - self.G)
        self.Var += self.alpha_step * (R * R - self.G * self.G - self.Var)
        if update_theta:
            g_prime = 0.0 if (self.Var - self.var_bound) < 0.0 else (self.Var - self.var_bound)**2
            if type == 'Var':
                self.theta += self.beta_step * (R - self.lam * g_prime * (R * R - 2.0 * self.G)) * zk
            if type == 'Sharpe':
                self.theta += self.beta_step / np.sqrt(self.Var) * \
                              (R - (self.G * R * R - 2.0 * self.G * self.G * R) / (2 * self.Var)) * zk

    @property
    def is_pitfall(self):
        return self.state[-1]

    # p_a represents a probability to be in the 2nd state
    def move(self, a):
        # self.dist = p_a.copy()
        # a = np.random.choice(self.actions, p=p_a)

        prev_state, prev_pos = self.state.copy(), self.position

        self.update_game(a)

        if prev_pos <= 9 and self.position >= 9 and self.is_pitfall:
            self.game_over = True
            return -3.0
        if self.position >= 19:
            self.game_over = True
            return 0
        else:
            return -0.1

    def update_game(self, a):
        self.state[self.position] = 0.0
        self.state[-2] = 0.0

        self.speed = clamp(self.speed + a, 0, 3)
        self.position = clamp(self.position + self.speed, 0, 19)

        self.state[self.position] = 1.0
        self.state[-2] = self.speed
        self.spawn_pitfall()

    # initialization function, chooses state randomly
    def zero_state(self):
        self.state = np.zeros(self.state_length)  # discrete states
        self.position = 0
        self.speed = 0
        self.state[0] = 1.0
        self.state[-2] = 0.0
        self.game_over = False
        self.spawn_pitfall()

    def spawn_pitfall(self):
        if self.pitfall and np.random.random() > 0.5:
            self.pitfall = False
        elif np.random.random() > 0.8:
            self.pitfall = True
        self.state[-1] = int(self.pitfall)

    # function returns probability of being in th 2nd state using softmax policy
    def sample_action(self, temperature=2.0):
        # http://incompleteideas.net/sutton/book/ebook/node17.html
        lst = self.state.dot(self.theta) / temperature
        # print(lst)
        e_lst = np.exp(lst)
        return e_lst / np.sum(e_lst)

    # gradient of log-likelihood used for computing zk
    def log_like(self, temperature=2.0):
        lst = self.state.dot(self.theta) / temperature
        e_lst = np.exp(lst)
        d_lst = np.repeat([[(e_lst[1] + e_lst[2]), (e_lst[0] + e_lst[2]), (e_lst[0] + e_lst[1])]], self.state_length,
                          axis=0)
        d_lst = np.multiply(self.state[np.newaxis].T, d_lst / np.sum(e_lst))

        return d_lst

    # function plays one game, computes total reward and zk along trajectory
    def play_one(self, T):
        total_rew, zk = 0.0, 0.0
        self.zero_state()
        for i in range(T):
            p_a = self.sample_action()
            zk += self.log_like()
            rew = self.move(p_a)
            total_rew += rew
            if self.game_over:
                break
        return total_rew, zk


# function plotting running average of vector vec, n specifies width of window
def plot_run_avg(vec, n, **kwargs):
    p = []
    for i in range(len(vec)):
        p.append(np.mean(vec[max(0, i - int(n/2)) : min(i+int(n/2), len(vec)-1)]))
    plt.plot(p, **kwargs)

if __name__ == '__main__':
    # TODO: make beta_step and alpha_step fulfil condition from paper
    N_games_learn = 5000 # number of games to play for learning
    N_games_test = 200  # number of games to play for data gathering
    length_of_game = 40  # number of steps in one game

    variance_bounds = [0.0, 100.0]  # variance bounds
    colors = ['red', 'blue']

    tr_plot, theta_plot, Var_plot, G_plot = [], [], [], []  # data for plots

    for v in variance_bounds:
        game = TwentyStateGame(v)  # instance of a game
        total_rews, theta, Var, G = [], [], [], []
        for i in range(N_games_learn):
            ut = False  # parameter which specifies if theta is updated in that iteration
            if i == N_games_learn/2:  # in the middle of the game make lambda to be almost 1.0
                # this is related to the approximation of COP solution approximation
                # equations (9) and (10) and 7 lines of text under
                game.lam = 0.99
            if i % 50 == 0:
                ut = True  # theta gets updated every 20. iteration
                theta.append(game.theta.copy())  # gathering data for graph
            Var.append(game.Var)  # gathering data for graph
            G.append(game.G)  # gathering data for graph
            total_rew, zk = game.play_one(length_of_game)
            game.update(total_rew, zk, update_theta=ut)  # finally update everything

        for i in range(N_games_test):
            total_rew, _ = game.play_one(length_of_game)  # gathering data for graph without update
            total_rews.append(total_rew)  # gathering data for graph

        tr_plot.append(total_rews)  # gathering data for graph
        theta_plot.append(theta)  # gathering data for graph
        Var_plot.append(Var)  # gathering data for graph
        G_plot.append(G)  # gathering data for graph

    plt.figure()
    for rew, v in zip(tr_plot,variance_bounds):
        plt.hist(rew, label='Var bound %f'%v)
    plt.legend()
    plt.title('Total rewards')
    # plt.show()

    plt.figure()
    for theta, v, c in zip(theta_plot,variance_bounds, colors):
        th = plt.plot(np.arange(N_games_learn/50), np.array(theta)[:,0],
                 np.arange(N_games_learn/50), np.array(theta)[:, 1],
                 np.arange(N_games_learn / 50), np.array(theta)[:, 2],
                 color=c)
        plt.legend([th], ['Var bound %f' % v])
    plt.title('Patameters theta')
    # plt.show()

    plt.figure()
    for rew, v in zip(Var_plot,variance_bounds):
        plot_run_avg(rew, 100, label='Var bound %f'%v)
    plt.legend()
    plt.title('Variance')
    # plt.show()

    plt.figure()
    for rew, v in zip(G_plot,variance_bounds):
        plot_run_avg(rew, 200, label='Var bound %f'%v)
    plt.legend()
    plt.title('Mean reward')
    plt.show()