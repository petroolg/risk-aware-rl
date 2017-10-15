import time
import numpy as np
import pygame
from matplotlib import pyplot as plt
from pygame import QUIT, K_LEFT, K_RIGHT, K_DOWN, K_UP
from pygame.colordict import THECOLORS

# Based on Policy Gradients with variance Related Risk Criteria

ROAD_W = 30
ROAD_H = 100

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

class car:

    def __init__(self, pos, size, v):
        self.pos = np.array(pos)
        self.size = size
        self.v = v
        self.color = THECOLORS.get('blue')


    def coord(self, y):
        x0 = self.pos[0] - self.size[0]/2
        y0 = self.pos[1] - y + ROAD_H/2
        return int(x0), int(y0)

    def draw(self, display, y):
        x, y = self.coord(y)
        pygame.draw.rect(display, self.color, (x,y,self.size[0],self.size[1]))

# function plotting running average of vector vec, n specifies width of window
def plot_run_avg(vec, n, **kwargs):
    p = []
    vec = np.array(vec)
    for i in range(len(vec)):
        p.append(np.mean(vec[int(max(0, i - n/2)) : int(min(i+n/2, len(vec)-1))]))
    plt.plot(p, **kwargs)

class road:
    def __init__(self, width, length, line1_vel, line2_vel):
        self.pole = np.zeros((length, width))
        self.width = width
        self.length = length
        self.l2c = int(self.width/4) + 1
        self.l1c = int(3 * self.width / 4) + 1
        self.l2v = line2_vel
        self.l1v = line1_vel
        self.vel = np.linspace(self.l1v, self.l2v, self.l1c - self.l2c)


class Road_game:
    def __init__(self, var_bound=0.0):

        self.road = road(ROAD_W, ROAD_H, 5, 8)
        self.car_size = [ROAD_W/5, 2*ROAD_W/5]

        self.actions = np.array([-1, 0, 1])

        self.goal = 270

        self.state_lenth = self.road.pole.shape[0]*self.road.pole.shape[1]
        self.theta = (np.random.sample((self.state_lenth, len(self.actions))) - 0.5)/np.sqrt(self.state_lenth)

        self.G = 0.0  # estimation of total reward
        self.Var = 0.0  # estimation of variance of reward
        self.var_bound = var_bound  # threshold of variance
        self.alpha_step = 0.005  # step of gradient ascent
        self.beta_step = 0.005  # step of gradient ascent
        self.lam = 0.1  # penalization, related to the approximation of COP solution, equations (9) and (10)

        pygame.init()
        self.DISPLAY = pygame.display.set_mode((ROAD_W+150, max(150, ROAD_H)), 0, 32)

    @property
    def state(self):
        return self.road.pole.reshape(self.state_lenth)

    def collision(self):
        x, y = self.car.pos - [self.car_size[0]/2, 0]
        a, b = self.car.pos + [self.car_size[0]/2, self.car_size[1]]
        for v in self.ov:
            x1, y1 = v.pos - [self.car_size[0] / 2, 0]
            a1, b1 = v.pos + [self.car_size[0] / 2, self.car_size[1]]
            if not (a < x1 or a1 < x or b < y1 or b1 < y):
                print('collision')
                return True
        return False

    def goal_reached(self):
        if self.car.pos[1] >= self.goal:
            print('goal reached')
            return True
        return False

        # Mean reward, variance and parameter update function, equations (13)
    def update(self, R, zk, type='Var', update_theta=True):
        self.G += self.alpha_step * (R - self.G)
        self.Var += self.alpha_step * (R * R - self.G * self.G - self.Var)
        print('Variance', self.Var)
        print('ZK:', np.min(zk), np.max(zk))
        if update_theta:
            g_prime = 0.0 if (self.Var - self.var_bound) < 0.0 else 2.0 * (self.Var - self.var_bound)
            if type == 'Var':
                print('Update:', (R - self.lam * g_prime * (R * R - 2.0 * self.G)))
                self.theta += self.beta_step * (R - self.lam * g_prime * (R * R - 2.0 * self.G)) * zk
            if type == 'Sharpe':
                self.theta += self.beta_step / np.sqrt(self.Var) * \
                              (R - (self.G * R * R - 2.0 * self.G * self.G * R) / (2 * self.Var)) * zk

    def auto_move(self, zk):
        p_a = self.sample_action()
        zk += self.log_like()
        rew = self.move(p_a)
        return rew, zk

    def process_keys(self):
        keys = pygame.key.get_pressed()
        p_a = np.zeros(3)
        if keys[K_RIGHT]:
            p_a[2] = 1.0
        elif keys[K_LEFT]:
            p_a[0] = 1.0
        else:
            p_a[1] = 1.0
        rew = self.move(p_a)
        return rew, 0.0

    # function plays one game, computes total reward and zk along trajectory
    def play_one(self, T):
        total_rew, zk = 0.0, 0.0
        self.init_game()
        i = 0
        # plt.imshow(self.road.pole)
        # plt.waitforbuttonpress
        while i < T and not self.game_over:
            rew, zk = self.auto_move(zk)
            # rew, zk = self.process_keys()
            total_rew += rew
            # time.sleep(0.2)
            i+=1
            # print(self.car.pos, i)
            # plt.imshow(self.road.pole)
            # plt.waitforbuttonpress()
        return total_rew, zk

    # function returns probability of being in th 2nd state using softmax policy
    def sample_action(self, temperature=20.0):
        # http://incompleteideas.net/sutton/book/ebook/node17.html
        lst = self.state.dot(self.theta)/temperature
        # print(lst)
        e_lst = np.exp(lst)
        return e_lst / np.sum(e_lst)

    # gradient of log-likelihood used for computing zk
    def log_like(self, temperature=2.0):
        lst = self.state.dot(self.theta) / temperature
        e_lst = np.exp(lst)
        d_lst = np.repeat([[(e_lst[1]+e_lst[2]), (e_lst[0]+e_lst[2]), (e_lst[0]+e_lst[1])]], self.state_lenth, axis=0)
        d_lst = np.multiply(self.state[np.newaxis].T, d_lst/np.sum(e_lst))
        
        return d_lst

    # initialization function, chooses state randomly
    def init_game(self):
        self.game_over = False
        self.ov = []

        pos = np.random.choice(np.arange(0,self.road.width,3))
        pos = np.random.choice(np.arange(0,self.road.width,3))
        for i in range(3):
            self.ov.append(car([self.road.l2c, pos], self.car_size,  self.road.l2v))
            pos += np.random.choice(np.arange(int(self.road.width/10 + self.car_size[1]),self.road.width*2,3))

        pos = [self.road.l1c, int(self.road.length/2 + 2*self.car_size[1])]
        self.ov.append(car(pos, self.car_size, self.road.l1v))
        pos = [self.road.l1c, int(self.road.length/2 - 3*self.car_size[1])]
        self.ov.append(car(pos, self.car_size, self.road.l1v))

        pos = [self.road.l1c, int(self.road.length/2)]
        self.car = car(pos, self.car_size, self.road.l1v)
        self.camera = [self.car.pos[1]]
        self.render()

    def draw_dist(self):
        if hasattr(self, 'dist'):
            try:
                pygame.draw.rect(self.DISPLAY, THECOLORS.get('blue'), (ROAD_W + 20, 20+100*(1-self.dist[0]), 30, 20+100*self.dist[0]))
                pygame.draw.rect(self.DISPLAY, THECOLORS.get('blue'), (ROAD_W + 60, 20+100*(1-self.dist[1]), 30, 20+100*self.dist[1]))
                pygame.draw.rect(self.DISPLAY, THECOLORS.get('blue'), (ROAD_W + 100, 20+100*(1-self.dist[2]), 30, 20+100*self.dist[2]))
            except TypeError:
                print((ROAD_W + 20, 20+100*(1-self.dist[0]), 30, 20+100*self.dist[0]))
                print((ROAD_W + 60, 20+100*(1-self.dist[1]), 30, 20+100*self.dist[1]))
                print((ROAD_W + 100, 20+100*(1-self.dist[2]), 30, 20+100*self.dist[2]))

    def render(self):
        self.road.pole = np.zeros_like(self.road.pole)
        w = int(self.car_size[0])
        h = int(self.car_size[1])
        for v in self.ov:
            x,y = v.coord(self.car.pos[1])
            self.road.pole[y:y+h, x:x+w] = 1.0
        x, y = self.car.coord(self.car.pos[1])
        self.road.pole[y:y + h, x:x + w] = 1.0

        self.DISPLAY.fill(THECOLORS.get('white'))
        pygame.draw.rect(self.DISPLAY, THECOLORS.get('grey'), (0, 0, ROAD_W, ROAD_H))

        for v in self.ov:
            v.draw(self.DISPLAY, self.car.pos[1])
        self.car.draw(self.DISPLAY, self.car.pos[1])

        self.draw_dist()

        pygame.display.update()

        quit_ = False
        for event in pygame.event.get():
            if event.type == QUIT:
                quit_ = True

        return quit_


    # p_a represents a probability to be in the 2nd state
    def move(self, p_a):

        self.dist = p_a.copy()
        a = np.random.choice(self.actions, p=p_a)

        if (self.car.pos[0] > self.road.l2c and a == -1) or (self.car.pos[0] <= self.road.l1c and a == 1):
            self.car.pos[0] += 3*a

        self.car.v = self.road.vel[clamp(self.road.l1c - self.car.pos[0], 0, len(self.road.vel)-1)]

        self.update_game()

        if self.collision():
            self.game_over=True
            return -3
        if self.goal_reached():
            self.game_over=True
            return 3
        return -0.01

    def update_game(self):
        for v in self.ov:
            v.pos[1] += v.v
        self.car.pos[1] += self.car.v
        self.render()

if __name__ == '__main__':
    # TODO: make beta_step and alpha_step fulfil condition from paper
    N_games_learn = 500 # number of games to play for learning
    N_games_test = 200 # number of games to play for data gathering
    length_of_game = 40  # number of steps in one game
    theta_update_step = 75

    variance_bounds = [100.0,0.0]  # variance bounds

    tr_plot, theta_plot, Var_plot, G_plot = [], [], [], []  # data for plots

    for v in variance_bounds:
        game = Road_game(v)  # instance of a game
        total_rews, theta, Var, G = [], [], [], []
        for i in range(N_games_learn):
            ut = False  # parameter which specifies if theta is updated in that iteration
            if i == int(2*N_games_learn/3):  # in the middle of the game make lambda to be almost 1.0
                # this is related to the approximation of COP solution approximation
                # equations (9) and (10) and 7 lines of text under
                game.lam = 0.99
            if i % theta_update_step == 0 and i != 0:
                ut = True  # theta gets updated every 20. iteration
                # theta.append(game.theta.copy())  # gathering data for graph
            Var.append(game.Var)  # gathering data for graph
            G.append(game.G)  # gathering data for graph
            total_rew, zk = game.play_one(length_of_game)
            print('Total reward, iteration:', total_rew, i)
            game.update(total_rew, zk, update_theta=ut)  # finally update everything

        for i in range(N_games_test):
            total_rew, _ = game.play_one(length_of_game)  # gathering data for graph without update
            total_rews.append(total_rew)  # gathering data for graph

        tr_plot.append(total_rews)  # gathering data for graph
        # theta_plot.append(theta)  # gathering data for graph
        Var_plot.append(Var)  # gathering data for graph
        G_plot.append(G)  # gathering data for graph

    plt.figure()
    for rew, v in zip(tr_plot,variance_bounds):
        plt.hist(rew, label='Var bound %f'%v)
    plt.legend()
    plt.title('Total rewards')

    # plt.figure()
    # for theta, v in zip(theta_plot,variance_bounds):
    #     plt.plot(np.arange(N_games_learn / theta_update_step - 1), np.array(theta)[:, 0],
    #              np.arange(N_games_learn / theta_update_step - 1), np.array(theta)[:, 1],
    #              np.arange(N_games_learn / theta_update_step - 1), np.array(theta)[:, 2],
    #              label='Var bound %f'%v)
    # plt.legend()
    # plt.title('Patameters theta')

    plt.figure()
    for rew, v in zip(Var_plot,variance_bounds):
        plot_run_avg(rew, 100, label='Var bound %f'%v)
    plt.legend()
    plt.title('Variance')

    plt.figure()
    for rew, v in zip(G_plot,variance_bounds):
        plot_run_avg(rew, 200, label='Var bound %f'%v)
    plt.legend()
    plt.title('Mean reward')
    plt.show()