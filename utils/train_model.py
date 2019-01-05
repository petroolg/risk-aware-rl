import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MDP"))
from environment import Road_game
from model import Model


def learn_model(model_name):
    game = Road_game()  # instance of a game
    transition_model = Model()

    learning_steps = 100000
    for i in range(learning_steps):
        game.play_one_learn_model(transition_model)
        print('\r{}/{}'.format(i, learning_steps), end='')
        if i % 1000 == 0:
            with open(model_name, 'wb') as file:
                pickle.dump(transition_model, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("--mn", dest='model_name', type=str, default="trans_model.pckl",
                        help='Path to save transition model.')

    args = parser.parse_args()
    model_name = args.model_name

    learn_model(model_name)
