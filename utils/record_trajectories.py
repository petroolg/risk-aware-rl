import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MDP"))
from environment import Road_game

def manual_control(path):
    game = Road_game(n_steps=100, visible=True)  # instance of a game

    if not os.path.exists(path):
        os.makedirs(path)

    while True:
        game.play_one(save=True, save_path=path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("--tp", dest='traj_path', type=str, default="trajectories",
                        help='Path to save expert\'s trajectories.')

    args = parser.parse_args()
    traj_path = args.traj_path



    manual_control(path=traj_path)