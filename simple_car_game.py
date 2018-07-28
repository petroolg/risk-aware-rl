import time
import numpy as np
import pygame
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import animation
from pygame import QUIT, K_LEFT, K_RIGHT, K_DOWN, K_UP
from pygame.colordict import THECOLORS
from datetime import datetime
from model import Model
import sys
import os
import pickle
import json

os.environ["SDL_VIDEODRIVER"] = "dummy"

ROAD_W = 6  # min value 6
ROAD_H = 30
ZOOM = 10


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


class Car:

    def __init__(self, pos, size, v, color='blue'):
        self.pos = np.array(pos)
        self.size = size
        self.v = v
        self.safety_border = (1, 1)
        self.color = THECOLORS.get(color)

    def coord(self, y):
        x0 = self.pos[0] - self.size[0] / 2
        y0 = ROAD_H - (self.pos[1] - y + int(ROAD_H / 4))
        return int(x0), int(y0)

    def draw_safe(self, display, y):
        x, y = self.coord(y)
        pygame.draw.rect(display, THECOLORS.get('lightpink'), ((x - self.safety_border[0]) * ZOOM,
                                                               (y - self.safety_border[1]) * ZOOM,
                                                               (self.size[0] + self.safety_border[0] + 1) * ZOOM,
                                                               (self.size[1] + self.safety_border[1] + 1) * ZOOM))

    def draw(self, display, y):
        x, y = self.coord(y)
        pygame.draw.rect(display, self.color, (x * ZOOM, y * ZOOM, self.size[0] * ZOOM, self.size[1] * ZOOM))


class road:
    def __init__(self, width, length, line1_vel, line2_vel):
        self.pole = np.zeros((length, width))
        self.width = width
        self.length = length
        self.l2c = int(self.width / 6) + 1
        self.l1c = int(5 * self.width / 6)
        self.l2v = line2_vel
        self.l1v = line1_vel
        self.vel = np.linspace(self.l1v, self.l2v, self.l1c - self.l2c)


class Road_game:
    def __init__(self):

        self.road = road(ROAD_W, ROAD_H, 6, 8)
        self.car_size = [ROAD_W // 5, 2 * ROAD_W // 5]

        self.step = 0
        self.goal = 70

        self.n_cars_behind_l1 = 0
        self.n_cars_behind_l2 = 0

        self.min_speed = 4
        self.max_speed = 11

        self.state_length = self.road.pole.shape[0] * self.road.pole.shape[1]

        pygame.init()
        self.DISPLAY = pygame.display.set_mode((ROAD_W * ZOOM, ROAD_H * ZOOM), 0, 32)
        pygame.display.set_caption('risk-aware-rl')

        self.car = None
        self.game_over = False
        self.camera = None

        self.safety_border = (1, 1)

        self.actios = [[1, 1],
                       [1, 0],
                       [1, -1],
                       [0, 1],
                       [0, 0],
                       [0, -1],
                       [-1, 1],
                       [-1, 0],
                       [-1, -1]]

        self.reward_dict = json.load(open('reward.json', 'r'))

    @property
    def state(self):
        # vel = np.zeros(self.max_speed-self.min_speed+1)
        # vel[int(self.car.v - self.min_speed)] = 1.0
        return np.hstack((self.road.pole.reshape(self.state_length), self.car.v))

    def collision(self):
        x, y = self.car.pos - [self.car_size[0] / 2, 0]
        a, b = self.car.pos + [self.car_size[0] / 2, self.car_size[1]]
        for v in self.ov:
            x1, y1 = v.pos - [self.car_size[0] / 2, 0]
            a1, b1 = v.pos + [self.car_size[0] / 2, self.car_size[1]]
            if not (a < x1 or a1 < x or b < y1 or b1 < y):
                # print('collision')
                return True
        return False

    def safe_collision(self):
        x, y = self.car.pos - [(self.car_size[0] + self.car.safety_border[0] * 2) / 2, 0]
        a, b = self.car.pos + [(self.car_size[0] + self.car.safety_border[0] * 2) / 2,
                               self.car_size[1] + self.car.safety_border[1] * 2]
        for v in self.ov:
            x1, y1 = v.pos - [(self.car_size[0] + self.car.safety_border[0] * 2) / 2, 0]
            a1, b1 = v.pos + [(self.car_size[0] + self.car.safety_border[0] * 2) / 2,
                              self.car_size[1] + self.car.safety_border[1] * 2]
            if not (a <= x1 or a1 <= x or b <= y1 or b1 <= y):
                # print('safe collision')
                return True
        return False

    def goal_reached(self):
        if self.step >= self.goal:
            return True
        return False

    def auto_move(self, zk):
        p_a = self.sample_action()
        rew = self.move(p_a)
        return rew, zk

    def process_keys(self):
        keys = pygame.key.get_pressed()
        d_v = np.zeros(2)  # direction and velocity

        if keys[K_RIGHT]:
            d_v[0] = 1.0
        elif keys[K_LEFT]:
            d_v[0] = -1.0
        if keys[K_UP]:
            d_v[1] = 1
        elif keys[K_DOWN]:
            d_v[1] = -1
        return d_v

    # function plays one game, computes total reward and zk along trajectory
    def play_one(self, seed=None, save=True):
        total_rew = []
        self.init_game(seed=seed)

        s_a_pairs = []

        while not self.game_over:
            # rew, zk = self.auto_move(zk)
            st = self.state
            d_v = self.process_keys()
            s_a_pairs.append(np.hstack((st, d_v)))
            rew = self.move(d_v)
            total_rew.append(rew)
            # print('\rVelocity: {} Reward: {} Step {}'.format(self.car.v, sum(total_rew), self.step), end='')
            time.sleep(0.15)
        if not self.collision() and save:
            np.save('trajectories_all/trajectories60x30/traj_{}.npy'.format(int(datetime.now().timestamp())), s_a_pairs)
            return sum(total_rew), len(s_a_pairs), True
        elif len(s_a_pairs) > 30 and save:
            np.save('trajectories_all/trajectories60x30/traj_{}.npy'.format(int(datetime.now().timestamp())),
                    s_a_pairs[:-10])
            return sum(total_rew[:-10]), len(s_a_pairs) - 10, True
        return sum(total_rew), len(s_a_pairs), False

    # function plays one game, computes total reward and zk along trajectory
    def play_one_learn_model(self, model: Model, seed=None, save=False):
        total_rew = 0.0
        self.init_game(seed=seed)

        s_a_pairs = []

        while not self.game_over:
            # rew, zk = self.auto_move(zk)
            st = self.state.copy()
            # d_v = self.process_keys()
            idx = np.random.choice(range(9))
            d_v = self.actios[idx]
            s_a_pairs.append(np.hstack((st, d_v)))
            rew = self.move(d_v)
            model.add_prob(st, self.actios.index(list(d_v)), self.state.copy(), self.event())
            total_rew += rew
            print('\rVelocity: {}'.format(self.car.v), end='')
            time.sleep(0.15)
        if not self.collision() and save:
            np.save('trajectories_all/trajectories60x30safety/traj_{}.npy'.format(int(datetime.now().timestamp())),
                    s_a_pairs)
        elif np.array(s_a_pairs).shape[0] > 30 and save:
            np.save('trajectories_all/trajectories60x30safety/traj_{}.npy'.format(int(datetime.now().timestamp())),
                    s_a_pairs[:-10])
        return total_rew

    # initialization function, chooses state randomly
    def init_game(self, seed):
        self.game_over = False
        self.ov = []
        self.step = 0

        np.random.seed(seed)

        self.car = Car([self.road.l1c, 0], self.car_size, self.road.l1v, color='green')
        self.camera = [self.car.pos[1]]
        self.n_cars_behind_l1 = 0
        self.n_cars_behind_l2 = 0

        pos1, pos2 = -2000, -2000
        car_dists = np.arange(int(self.road.width / 10 + self.car_size[1]), self.road.width * 7, 7)
        for i in range(120):
            self.ov.append(Car([self.road.l2c, pos1], self.car_size, self.road.l2v))
            if not int(-3 * self.car_size[1]) < pos2 < int(3 * self.car_size[1]):
                self.ov.append(Car([self.road.l1c, pos2], self.car_size, self.road.l1v))
            self.n_cars_behind_l1 += int(pos2 < 0)
            self.n_cars_behind_l2 += int(pos1 < 0)
            pos1 += np.random.choice(car_dists)
            pos2 += np.random.choice(car_dists)

        self.render()
        np.random.seed(None)

    def render(self):

        self.DISPLAY.fill(THECOLORS.get('white'))
        pygame.draw.rect(self.DISPLAY, THECOLORS.get('grey'), (0, 0, ROAD_W * ZOOM, ROAD_H * ZOOM))

        for v in self.ov:
            v.draw_safe(self.DISPLAY, self.car.pos[1])

        for v in self.ov:
            v.draw(self.DISPLAY, self.car.pos[1])

        self.car.draw_safe(self.DISPLAY, self.car.pos[1])
        self.car.draw(self.DISPLAY, self.car.pos[1])

        pygame.display.update()

        bluepx = pygame.surfarray.pixels_blue(self.DISPLAY)
        bluepx = np.array(bluepx).astype(int)
        redpx = pygame.surfarray.pixels_red(self.DISPLAY)
        redpx = np.array(redpx).astype(int)
        greenpx = pygame.surfarray.pixels_green(self.DISPLAY)
        greenpx = np.array(greenpx).astype(int)
        greenpx[greenpx != 255] = 0
        greenpx[greenpx == 255] = 1
        bluepx[bluepx != 255] = 0
        bluepx[bluepx == 255] = 1
        redpx[redpx != 255] = 0
        redpx[redpx == 255] = 1
        self.road.pole = 2 * bluepx[::ZOOM, ::ZOOM] + 2 * greenpx[::ZOOM, ::ZOOM] + redpx[::ZOOM, ::ZOOM] - 1

        # plt.imshow(self.road.pole.reshape(ROAD_W, ROAD_H))
        # plt.colorbar()
        # plt.show()

        quit_ = False
        for event in pygame.event.get():
            if event.type == QUIT:
                quit_ = True

        return quit_

    def event(self):

        events = ()

        if self.collision():
            self.game_over = True
            return ('collision',)

        if self.safe_collision():
            events += ('safe_collision',)

        if self.goal_reached():
            self.game_over = True

        n_cars_behind_l1 = 0
        n_cars_behind_l2 = 0
        for car in self.ov:
            n_cars_behind_l1 += int(car.pos[1] < self.car.pos[1]) * int(car.pos[0] == self.road.l1c)
            n_cars_behind_l2 += int(car.pos[1] < self.car.pos[1]) * int(car.pos[0] == self.road.l2c)

        reward1 = (n_cars_behind_l2 - self.n_cars_behind_l2) * 5
        reward2 = (n_cars_behind_l1 - self.n_cars_behind_l1)

        self.n_cars_behind_l1 = n_cars_behind_l1
        self.n_cars_behind_l2 = n_cars_behind_l2

        if reward1 > 0:
            events += ('overtake_fast',)
        elif reward1 < 0:
            events += ('pass_fast',)

        if reward2 > 0:
            events += ('overtake_slow',)
        elif reward2 < 0:
            events += ('pass_slow',)

        if not events:
            events = ('nil',)

        return events

    def reward(self, events=None):
        r = 0
        if not events:
            for e in self.event():
                r += self.reward_dict[e]
        else:
            for e in events:
                r += self.reward_dict[e]
        return r

    def move(self, d_v):

        self.step += 1

        self.dist = d_v.copy()

        if (self.car.pos[0] > self.road.l2c and d_v[0] == -1) or (self.car.pos[0] < self.road.l1c and d_v[0] == 1):
            self.car.pos[0] += d_v[0]

        # self.car.v = self.road.vel[clamp(self.road.l1c - self.car.pos[0], 0, len(self.road.vel)-1)]
        if (self.car.v < self.max_speed and d_v[1] > 0) or (self.car.v > self.min_speed and d_v[1] < 0):
            self.car.v += d_v[1]

        self.update_game()

        return self.reward()

    def update_game(self):
        for v in self.ov:
            v.pos[1] += v.v
        self.car.pos[1] += self.car.v
        self.render()

    def quit(self):
        pygame.quit()


def manual_control(seeds):
    game = Road_game()  # instance of a game
    while True:
        seed = np.random.choice(seeds)
        print(seed)
        total_rew, len_traj, saved = game.play_one(seed=seed)
        print('\n{:.2f}\n'.format(total_rew))
        if saved:
            if os.path.isfile('trajectories_all/trajectories60x30/meta.npy'):
                meta = np.load('trajectories_all/trajectories60x30/meta.npy')
                meta = np.append(meta, [[total_rew, len_traj]], axis=0)
                np.save('trajectories_all/trajectories60x30/meta.npy', meta)
            else:
                np.save('trajectories_all/trajectories60x30/meta.npy', np.atleast_2d([total_rew, len_traj]))


def replay_game(traj):
    fig = plt.figure()

    def f(i):
        return traj[i, :-3].reshape((ROAD_W, ROAD_H)).T

    ims = [] # type list
    for i in range(len(traj)):
        im = plt.imshow(f(i), animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=10000, repeat=False)

    plt.show()


def learn_model(seeds):
    game = Road_game()  # instance of a game
    # model = pickle.load(open('trans_model_safe.pckl', 'rb'))
    model = Model()

    while True:
        # seed = np.random.choice(seeds)
        # print(seed)
        total_rew = game.play_one_learn_model(model, seed=None)
        print('\n{:.2f}\n'.format(total_rew))
        with open('trans_model_safe.pckl', 'wb') as file:
            pickle.dump(model, file)


if __name__ == '__main__':
    seeds = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    # for traj in os.listdir('trajectories_all/trajectories60x30'):
    #     replay_game(np.load('trajectories_all/trajectories60x30/' + traj))
    manual_control(seeds)
    # learn_model(seeds)
