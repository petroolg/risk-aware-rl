import numpy as np
from matplotlib import pyplot as plt

# Based on Policy Gradients with variance Related Risk Criteria

class TwoStateGame:

    def __init__(self, var_bound):
        self.state = np.array([0.0, 1.0])  # two discrete states
        # [0.0, 1.0] means agent is in th 2nd state, [1.0, 0.0] - 1st state
        self.theta = np.array([[0.0],[0.0]])  # parameters

        self.G = 0.0  # estimation of total reward
        self.Var = 0.0  # estimation of variance of reward
        self.var_bound = var_bound  # threshold of variance
        self.alpha_step = 0.1  # step of gradient ascent
        self.beta_step = 0.1  # step of gradient ascent
        self.lam = 0.1  # penalization, related to the approximation of COP solution, equations (9) and (10)
        self.eps = 0.0001  # epsilon for epsilon-constrained softmax policy

    # Mean reward, variance and parameter update function, equations (13)
    def update(self, R, zk, type='Var', update_theta=True):
        self.G += self.alpha_step * (R - self.G)
        self.Var += self.alpha_step * (R * R - self.G * self.G - self.Var)
        if update_theta:
            g_prime = 0.0 if (self.Var - self.var_bound) < 0.0 else 2.0 * (self.Var - self.var_bound)
            if type == 'Var':
                self.theta += self.beta_step * (R - self.lam * g_prime * (R * R - 2.0 * self.G)) * zk[np.newaxis].T
            if type == 'Sharpe':
                self.theta += self.beta_step / np.sqrt(self.Var) * \
                              (R - (self.G * R * R - 2.0 * self.G * self.G * R) / (2 * self.Var)) * zk[np.newaxis].T

    # p_a represents a probability to be in the 2nd state
    def move(self, p_a):
        a = np.random.rand()
        if a < p_a:
            self.state = np.array([0.0, 1.0])
            # reward sampled from normal distribution with 0.7 mean and 0.3 variance
            # higher return, higher risk
            return np.random.normal(0.7, 0.3, 1)[0]
        else:
            self.state = np.array([1.0, 0.0])
            # reward sampled from normal distribution with 0.5 mean and 0.1 variance
            # lower return, lower risk
            return np.random.normal(0.5, 0.1, 1)[0]

    # initialization function, chooses state randomly
    def zero_state(self):
        choice = np.random.choice([0, 1])
        self.state = np.zeros(2)
        self.state[choice] = 1.0

    # function returns probability of being in th 2nd state using softmax policy
    def sample_action(self):
        """epsilon-constrained softmax policy"""
        return self.eps + (1.0 - 2.0*self.eps)/(1.0 + np.exp(-self.state.dot(self.theta)))

    # gradient of log-likelihood used for computing zk
    def log_like(self):
        v = (self.eps*self.state)/((self.eps-1.0)*np.exp(self.state.dot(self.theta)) - self.eps) + \
        self.state/(np.exp(self.state.dot(self.theta)) + 1.0)
        return v

    # function plays one game, computes total reward and zk along trajectory
    def play_one(self, T):
        total_rew, zk = 0.0, 0.0
        self.zero_state()
        for i in range(T):
            p_a = self.sample_action()
            zk += self.log_like()
            rew = self.move(p_a)
            total_rew += rew
        return total_rew, zk


# function plotting running average of vector vec, n specifies width of window
def plot_run_avg(vec, n, **kwargs):
    p = []
    for i in range(len(vec)):
        p.append(np.mean(vec[max(0, i - n/2) : min(i+n/2, len(vec)-1)]))
    plt.plot(p, **kwargs)

if __name__ == '__main__':
    # TODO: make beta_step and alpha_step fulfil condition from paper
    N_games_learn = 500 # number of games to play for learning
    N_games_test = 500  # number of games to play for data gathering
    length_of_game = 10  # number of steps in one game

    variance_bounds = [100.0, 0.0]  # variance bounds

    tr_plot, theta_plot, Var_plot, G_plot = [], [], [], []  # data for plots

    for v in variance_bounds:
        game = TwoStateGame(v)  # instance of a game
        total_rews, theta, Var, G = [], [], [], []
        for i in range(N_games_learn):
            ut = False  # parameter which specifies if theta is updated in that iteration
            if i == N_games_learn/2:  # in the middle of the game make lambda to be almost 1.0
                # this is related to the approximation of COP solution approximation
                # equations (9) and (10) and 7 lines of text under
                game.lam = 0.99
            if i % 20 == 0:
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

    for rew, v in zip(tr_plot,variance_bounds):
        plt.hist(rew, label='Var bound %f'%v)
    plt.legend()
    plt.title('Total rewards')
    plt.show()

    for theta, v in zip(theta_plot,variance_bounds):
        plt.plot(np.arange(N_games_learn/20), np.array(theta)[:,0],
                 np.arange(N_games_learn/20), np.array(theta)[:, 1],
                 label='Var bound %f'%v)
    plt.legend()
    plt.title('Patameters theta')
    plt.show()

    for rew, v in zip(Var_plot,variance_bounds):
        plot_run_avg(rew, 100, label='Var bound %f'%v)
    plt.legend()
    plt.title('Variance')
    plt.show()

    for rew, v in zip(G_plot,variance_bounds):
        plot_run_avg(rew, 200, label='Var bound %f'%v)
    plt.legend()
    plt.title('Mean reward')
    plt.show()