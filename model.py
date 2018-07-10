import numpy as np
from matplotlib import pyplot as plt

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


    def add_prob(self, s_raw, a, s_prime_raw, reward):

        s = hash(np.array(s_raw).tobytes())
        s_prime = hash(np.array(s_prime_raw).tobytes())

        if not self.transition_model.get(s,{}):
            self.transition_model[s] = {}
        if not self.transition_model[s].get(a, {}):
            self.transition_model[s][a] = {}
        if not self.transition_model[s][a].get(s_prime, []):
            self.transition_model[s][a][s_prime] = [0,0]
        else:
            if self.transition_model[s][a][s_prime][1] != reward:
                print("Reward doesnt match!", self.transition_model[s][a][s_prime][1], reward)
                plt.figure()
                plt.imshow(s_raw[:-1].reshape(6,30))
                plt.figure()
                plt.imshow(s_prime_raw[:-1].reshape(6,30))
                plt.show()
                plt.waitforbuttonpress()
        self.transition_model[s][a][s_prime][0] += 1
        self.transition_model[s][a][s_prime][1] = reward