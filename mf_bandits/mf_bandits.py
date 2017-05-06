#mf_bandits.py
#multi-fidelity multi-armed bandit classes
#based on
#Byron Galbraith's multi-armed bandit simulator classes
#source: https://github.com/bgalbraith/bandits/blob/master/bandits/bandit.py

import numpy as np
import pymc3 as pm

class MF_MultiArmedBandit(object):
    """
    A Multi-fidelity Multi-armed Bandit
    with k arms and m fidelities (per each arm)
    """
    def __init__(self, k, m):
        self.k = k
        self.m = m
        self.action_values = np.zeros((k,m))
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros((self.k, self.m))
        self.optimal = 0

    def pull(self, action):
        return 0, True


class MF_GaussianBandit(MF_MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    Each arm k has m fidelities
    Passing in a matrix of mu values allows for each arm and fidelity level to use
    the corresponding mean
    zeta(m) is the bound on the interval about which fidelity m over/undershoots 
    the mean of the highest fidelity
    """
    def __init__(self, k, m, mu=0, sigma=1, zeta=0):
        super(MF_GaussianBandit, self).__init__(k,m)
        self.mu = mu
        self.sigma = sigma
        self.zeta = zeta
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, (self.k, self.m))
        self.optimal = np.argmax(self.action_values,axis=0)

    def pull(self, action):
        return (np.random.normal(self.action_values[action[0], action[1]]),
                action == self.optimal)
