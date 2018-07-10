import time
import numpy as np
import pygame
# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
from pygame import QUIT, K_LEFT, K_RIGHT, K_DOWN, K_UP
from pygame.colordict import THECOLORS
from datetime import datetime
import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"

# Based on Policy Gradients with variance Related Risk Criteria

ROAD_W = 6 # min value 6
ROAD_H = 30
ZOOM = 5

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
        y0 = ROAD_H - (self.pos[1] - y + int(ROAD_H/4))
        return int(x0), int(y0)

    def draw(self, display, y):
        x, y = self.coord(y)
        pygame.draw.rect(display, self.color, (x*ZOOM,y*ZOOM,self.size[0]*ZOOM,self.size[1]*ZOOM))

# # function plotting running average of vector vec, n specifies width of window
# def plot_run_avg(vec, n, **kwargs):
#     p = []
#     vec = np.array(vec)
#     for i in range(len(vec)):
#         p.append(np.mean(vec[int(max(0, i - n/2)) : int(min(i+n/2, len(vec)-1))]))
#     plt.plot(p, **kwargs)

class road:
    def __init__(self, width, length, line1_vel, line2_vel):
        self.pole = np.zeros((length, width))
        self.width = width
        self.length = length
        self.l2c = int(self.width/6) + 1
        self.l1c = int(5 * self.width / 6)
        self.l2v = line2_vel
        self.l1v = line1_vel
        self.vel = np.linspace(self.l1v, self.l2v, self.l1c - self.l2c)


class Road_game:
    def __init__(self):

        self.road = road(ROAD_W, ROAD_H, 6, 8)
        self.car_size = [ROAD_W/5, 2*ROAD_W/5]

        self.goal = 100

        self.state_length = self.road.pole.shape[0] * self.road.pole.shape[1]

        pygame.init()
        self.DISPLAY = pygame.display.set_mode((ROAD_W*ZOOM, ROAD_H*ZOOM), 0, 32)
        pygame.display.set_caption('risk-aware-rl')

        self.car = None
        self.game_over = False
        self.camera = None

        self.actios = [[1,1],
                       [1,0],
                       [1,-1],
                       [0,1],
                       [0,0],
                       [0,-1],
                       [-1,1],
                       [-1,0],
                       [-1,-1]]


    @property
    def state(self):
        return np.hstack((self.road.pole.reshape(self.state_length), self.car.v))

    def collision(self):
        x, y = self.car.pos - [self.car_size[0]/2, 0]
        a, b = self.car.pos + [self.car_size[0]/2, self.car_size[1]]
        for v in self.ov:
            x1, y1 = v.pos - [self.car_size[0] / 2, 0]
            a1, b1 = v.pos + [self.car_size[0] / 2, self.car_size[1]]
            if not (a < x1 or a1 < x or b < y1 or b1 < y):
                # print('collision')
                return True
        return False

    def goal_reached(self):
        if self.car.pos[1] >= self.goal:
            # print('goal reached')
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
    def play_one(self, seed=None, save=False):
        total_rew = 0.0
        self.init_game(seed=seed)

        s_a_pairs = []

        while not self.game_over:
            # rew, zk = self.auto_move(zk)
            st = self.state
            d_v = self.process_keys()
            s_a_pairs.append(np.hstack((st,d_v)))
            rew = self.move(d_v)
            total_rew += rew
            print('\rVelocity: {}'.format(self.car.v), end='')
            time.sleep(0.1)
        if not self.collision() and save:
            np.save('trajectories_all/trajectories_rand_big/traj_{}.npy'.format(int(datetime.now().timestamp())), s_a_pairs)
        elif np.array(s_a_pairs).shape[0]>80 and save:
            np.save('trajectories_all/trajectories_rand_big/traj_{}.npy'.format(int(datetime.now().timestamp())), s_a_pairs[:-10])
        return total_rew

    def replay(self, traj):
        self.init_game()
        i = 0
        for t in traj:
            rew = self.move(t[-2:])
            print('\rVelocity: {}'.format(self.car.v), end='')
            time.sleep(0.15)
            i+=1


    # initialization function, chooses state randomly
    def init_game(self, seed):
        self.game_over = False
        self.ov = []

        np.random.seed(seed)

        pos1, pos2 = -2000, -2000
        car_dists = np.arange(int(self.road.width/10 + self.car_size[1]),self.road.width*7,7)
        for i in range(120):
            self.ov.append(car([self.road.l2c, pos1], self.car_size,  self.road.l2v))
            if not int(self.road.length/2 - 3*self.car_size[1]) < pos2 < int(self.road.length/2 + 3*self.car_size[1]):
                self.ov.append(car([self.road.l1c, pos2], self.car_size,  self.road.l1v))
            pos1 += np.random.choice(car_dists)
            pos2 += np.random.choice(car_dists)
        # print(pos1,pos2)

        pos = [self.road.l1c, int(self.road.length/2)]
        self.car = car(pos, self.car_size, self.road.l1v)
        self.camera = [self.car.pos[1]]
        self.render()

    def render(self):

        self.DISPLAY.fill(THECOLORS.get('white'))
        pygame.draw.rect(self.DISPLAY, THECOLORS.get('grey'), (0, 0, ROAD_W*ZOOM, ROAD_H*ZOOM))

        for v in self.ov:
            v.draw(self.DISPLAY, self.car.pos[1])
        self.car.draw(self.DISPLAY, self.car.pos[1])

        pygame.display.update()

        bluepx = pygame.surfarray.pixels_blue(self.DISPLAY)
        bluepx = np.array(bluepx)
        bluepx[bluepx == 190] = 0
        bluepx[bluepx == 255] = 1
        self.road.pole = bluepx[::ZOOM, ::ZOOM]

        quit_ = False
        for event in pygame.event.get():
            if event.type == QUIT:
                quit_ = True

        return quit_


    def move(self, d_v):

        self.dist = d_v.copy()

        if (self.car.pos[0] > self.road.l2c and d_v[0] == -1) or (self.car.pos[0] < self.road.l1c and d_v[0] == 1):
            self.car.pos[0] += d_v[0]

        # self.car.v = self.road.vel[clamp(self.road.l1c - self.car.pos[0], 0, len(self.road.vel)-1)]
        if 0 < self.car.v < 16:
            self.car.v += d_v[1]

        self.update_game()

        if self.collision():
            self.game_over = True
            return -10
        if self.goal_reached():
            self.game_over = True
            return 1
        return 0.1

    def update_game(self):
        for v in self.ov:
            v.pos[1] += v.v
        self.car.pos[1] += self.car.v
        self.render()

    def quit(self):
        pygame.quit()

def manual_control():

    game = Road_game()  # instance of a game

    while True:
        total_rew = game.play_one()
        print(total_rew)

def replay_game(traj):
    game = Road_game()
    game.replay(traj)


if __name__ == '__main__':
    # replay_game(np.load('trajectories/traj_1527959902.npy'))
    manual_control()
