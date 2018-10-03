import numpy as np
import hashlib
import sys

import matplotlib
from matplotlib import pyplot as plt


def hash18(val):
    return int(hashlib.sha1(val).hexdigest(), 16) % (sys.maxsize * 2) - sys.maxsize


class Model:

    def __init__(self):
        # structure: {s0:
        #   {a0:
        #       {s4:  [38, ['collision']]
        #        s12: [15, ['nil']]
        #       }
        #   {a1:
        #       {s2:  [3, ['overtook_1']]
        #        s14: [123, ['nil']]
        #       }
        #   }
        # }
        #
        self.transition_model = {}

    def get(self, s, default):
        return self.transition_model.get(np.array(s).astype(int).tobytes(), default)

    def get_distr(self, s, a, game):
        if type(s) is np.ndarray or type(s) is list:
            s = np.array(s).astype(int).tobytes()

        s_dict = self.transition_model.get(s, {})
        a_dict = s_dict.get(a, {})
        # plt.imshow(np.frombuffer(s)[:-8].reshape((6,30)))
        # for st in a_dict.keys():
        #     plt.imshow(np.frombuffer(st)[:-8].reshape((6,30)))
        #     plt.show()

        if not a_dict:
            return [], []
        mtx = list(a_dict.values())
        keys = list(a_dict.keys())
        freqs = np.array([v[0] for v in mtx])
        rews = np.array([game.reward(events=v[1]) for v in mtx])

        return np.vstack((freqs / freqs.sum(), rews)).T, keys

    def add_prob(self, s_raw, a, s_prime_raw, reward):

        # s = hash18(np.array(s_raw).astype(int).tobytes())
        # s_prime = hash18(np.array(s_prime_raw).astype(int).tobytes())

        s = np.array(s_raw).astype(int).tobytes()
        s_prime = np.array(s_prime_raw).astype(int).tobytes()

        if not self.transition_model.get(s, {}):
            self.transition_model[s] = {}
        if not self.transition_model[s].get(a, {}):
            self.transition_model[s][a] = {}
        if not self.transition_model[s][a].get(s_prime, []):
            self.transition_model[s][a][s_prime] = [0, 0]
        else:
            if (np.array(self.transition_model[s][a][s_prime][1]) != np.array(reward)).any():
                print("Reward doesnt match!", self.transition_model[s][a][s_prime][1], reward)
        self.transition_model[s][a][s_prime][0] += 1
        self.transition_model[s][a][s_prime][1] = reward
