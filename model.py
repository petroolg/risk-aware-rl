import numpy as np
import hashlib
import sys

def hash18(val):
    return int(hashlib.sha1(val).hexdigest(), 16) % (sys.maxsize * 2) - sys.maxsize

class Model:

    def __init__(self):
        # structure: {s0:
        #   {a0:
        #       {s4:  [38, -10]
        #        s12: [15, 0.1]
        #       }
        #   {a1:
        #       {s2:  [3, 0.1]
        #        s14: [123, 0.1]
        #       }
        #   }
        # }
        #
        self.transition_model = {}

    def get(self, s, default):
        return self.transition_model.get(hash18(np.array(s).astype(int).tobytes()), default)

    def add_prob(self, s_raw, a, s_prime_raw, reward):

        s = hash18(np.array(s_raw).astype(int).tobytes())
        s_prime = hash18(np.array(s_prime_raw).astype(int).tobytes())

        if not self.transition_model.get(s,{}):
            self.transition_model[s] = {}
        if not self.transition_model[s].get(a, {}):
            self.transition_model[s][a] = {}
        if not self.transition_model[s][a].get(s_prime, []):
            self.transition_model[s][a][s_prime] = [0,0]
        else:
            if self.transition_model[s][a][s_prime][1] != reward:
                print("Reward doesnt match!", self.transition_model[s][a][s_prime][1], reward)
        self.transition_model[s][a][s_prime][0] += 1
        self.transition_model[s][a][s_prime][1] = reward