import argparse
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MDP"))
from environment import ROAD_H, ROAD_W


def replay_game(traj):
    fig = plt.figure()

    def f(i):
        return traj[i, :ROAD_W * ROAD_H].reshape((ROAD_W, ROAD_H)).T

    ims = []  # type list
    for i in range(len(traj)):
        im = plt.imshow(f(i), animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=10000, repeat=False)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("--tp", dest='traj_path', type=str, required=True,
                        help='Path to the trajectory for replay.')

    args = parser.parse_args()
    traj_path = args.traj_path
    traj = np.load(traj_path)

    replay_game(traj)
