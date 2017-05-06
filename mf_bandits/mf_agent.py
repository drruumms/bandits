#mf_agent.py
#multifidelity extension of
#Byron Galbraith's multi-armed bandit simulator Agent classes
#source: https://github.com/bgalbraith/bandits/blob/master/bandits/policy.py


import numpy as np
import pymc3 as pm


class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step. The
    action is chosen using a strategy based on the history of prior actions
    and outcome observations.
    """
    def __init__(self, bandit, policy, prior=0, gamma=None):
        self.policy = policy
        self.k = bandit.k
        self.m = bandit.m
        self.zeta = bandit.zeta
        self.costs = bandit.costs
        self.prior = prior
        self.gamma = gamma
        self.make_gamma_fn()
        self._value_estimates = prior*np.ones((self.k, self.m))
        self.action_attempts = np.zeros((self.k,self.m))
        self.t = 0
        self.last_action = None

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def make_gamma_fn(self):
        """
        makes the gamma function for choosing a fidelity
        based on relative costs and mean bdry interval
        """
        self.gamma_fn = np.zeros(self.m-1)    
        for m in range(self.m-1):
            self.gamma_fn[m] = np.sqrt(self.costs[m]/self.costs[m+1]*(self.zeta[1,m]**2)) 

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._value_estimates[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1

        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action]
        else:
            g = self.gamma
        q = self._value_estimates[self.last_action]

        self._value_estimates[self.last_action] += g*(reward - q)
        self.t += 1


    @property
    def value_estimates(self):
        return self._value_estimates
