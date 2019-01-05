import numpy as np
from matplotlib import pyplot as plt
from simple_game import plot_run_avg

class StockExchange:

    def __init__(self, N, r_l, p_switch, r_nl_high, r_nl_low, inv_fraction, var_bound=0.0):
        self.N = N  # number of steps of maturity
        self.state = np.zeros(N+2, dtype=float)  # state
        self.state[0] = 100.0  # all money are invested in liquid assets
        self.p_switch = p_switch  # probability of transition between r_nl_high and r_nl_low
        self.r_l = r_l  # interest rate of liquid asset
        self.r_nl_high = r_nl_high  # high interest rate of non-liquid asset
        self.r_nl_low = r_nl_low  # low interest rate of non-liquid asset
        self.r_nl = r_nl_low  # current interest rate of non-liquid asset
        self.alpha = inv_fraction  # fixed fraction of portfolio to invest
        self.p_risk = 0.2  # risk of not being paid (non-liquid asset)

        self.theta = (np.random.random_sample(N+2)[np.newaxis].T-0.5)/np.sqrt(N+2)
        # self.theta = np.zeros(N+2)[np.newaxis].T

        self.G = 0.0  # estimation of total reward
        self.Var = 0.0  # estimation of variance of reward
        self.var_bound = var_bound  # threshold of variance
        self.alpha_step = 0.01  # step of gradient ascent
        self.beta_step = 0.01  # step of gradient ascent
        self.lam = 0.1  # penalization, related to the approximation of COP solution

        self.eps = 0.0001  # epsilon for epsilon-constrained softmax policy

    def sample_action(self):
        """epsilon-constrained policy"""
        return self.eps + (1.0 - 2.0*self.eps)/(1.0 + np.exp(-self.state.dot(self.theta)))

    def zero_state(self):
        """init state to all zeroes and one in the beginning"""
        self.state = np.zeros(self.N + 2, dtype=float)
        self.state[0] = 100.0

    def update(self, R, zk, type='Var', update_theta=True):
        self.G += self.alpha_step * (R - self.G)
        self.Var += self.alpha_step * (R * R - self.G * self.G - self.Var)
        if update_theta:
            g_prime = 0.0 if (self.Var - self.var_bound) < 0.0 else 2.0 * (self.Var - self.var_bound)
            if type == 'Var':
                self.theta += self.beta_step * (R - self.lam * g_prime * (R * R - 2.0 * self.G)) * zk.T
            if type == 'Sharpe':
                self.theta += self.beta_step/np.sqrt(self.Var) * (R - (self.G*R*R - 2.0*self.G*self.G*R)/(2 * self.Var)) * zk.T
        # print('Variance', self.Var)
        # print('Total gain',self.G)
        # print('Theta', np.mean(self.theta))

    def invest(self, prob):
        """step in game"""
        default = np.random.random()  # probability of default in country
        if default > self.p_risk:  # there's no default, non-liquid assets will be paid
            rew = self.state[1] * self.r_nl
            self.state[0] += self.state[1]
        else:
            rew = 0.0  # there's a default, non-liquid assets will not be paid
            self.state[0] += self.state[1]
        self.state[1:-2] = self.state[2:-1]
        self.state[-2] = 0.0
        a = np.random.random()
        if self.state[0] >= self.alpha:
            if a < prob:
                    self.state[-2] = self.alpha*100.0
                    self.state[0] -= self.alpha*100.0
            else:
                rew += self.r_l * self.state[0]
        return rew

    def advance(self):
        p = np.random.random()
        if p < self.p_switch:
            self.r_nl = self.r_nl_low if self.r_nl == self.r_nl_high else self.r_nl_high


    def log_like(self):
        v = (self.eps*self.state)/((self.eps-1.0)*np.exp(np.multiply(self.state,self.theta.T)) - self.eps) + \
        self.state/(np.exp(np.multiply(self.state,self.theta.T)) +1.0)
        return v.copy()

    def play_one(self, T):
        total_rew, zk = 0.0, 0.0
        nl_int_rates = [self.r_nl]
        self.zero_state()
        for i in range(T):
            self.state[-1] = self.r_nl - np.mean(nl_int_rates)
            p_a = self.sample_action()
            zk += self.log_like()
            rew = self.invest(p_a)
            total_rew += rew
            self.advance()
            nl_int_rates.append(self.r_nl)
        return total_rew, zk

if __name__ == '__main__':
    # TODO: make beta_step and alpha_step fulfil condition from paper
    N_games_learn = 3000
    N_games_test = 100
    length_of_game = 50
    variance_bounds = [0.0, 1000.0]
    tr_plot, theta_plot, Var_plot, G_plot = [], [], [], []  # data for plots
    #  N, r_l, p_switch, r_nl_high, r_nl_low, inv_fraction
    game_params = [10, 0.002, 0.1, 0.3, 0.01, 0.2]
    for v in variance_bounds:
        game = StockExchange(*game_params, var_bound=v)
        total_rews, theta, Var, G = [], [], [], []
        for i in range(N_games_learn):
            ut = False
            if i == N_games_learn/2:
                game.lam = 0.99
            if i% 20 == 0:
                ut = True
                theta.append(game.theta.copy())
            Var.append(game.Var)  # gathering data for graph
            G.append(game.G)  # gathering data for graph
            total_rew, zk = game.play_one(length_of_game)
            game.update(total_rew, zk, update_theta=ut)


        for i in range(N_games_test):
            total_rew, zk = game.play_one(length_of_game)
            total_rews.append(total_rew)

        tr_plot.append(total_rews)  # gathering data for graph
        theta_plot.append(theta)  # gathering data for graph
        Var_plot.append(Var)  # gathering data for graph
        G_plot.append(G)


    for rew, v in zip(tr_plot,variance_bounds):
        plt.hist(rew, label='Var bound %f'%v)
    plt.legend()
    plt.title('Total rewards')
    plt.show()

    for theta, v in zip(theta_plot,variance_bounds):
        for i in range(np.array(theta).shape[1]):
            plt.plot(np.arange(N_games_learn/20), np.array(theta)[:,i],
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