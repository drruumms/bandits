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
        self.k = bandit.k #get number of bandit arms
        self.m = bandit.m #get number of arm fidelities
        self.zeta = bandit.zeta #inherit fidelity mean over/undershooting bounds

        #dont know what these two are for, just using default settings (which do nada)
        self.prior = prior
        self.gamma = gamma

        #gamma function for MFUCB algorithm - choosing fidelities based on costs
        self.make_gamma_fn(bandit.costs)

        #track estimates of values of each arm and fidelity
        self._value_estimates = prior*np.ones((self.k, self.m))
        #keeps track of number of plays at each arm and fidelity
        self.action_attempts = np.zeros((self.k,self.m)) 
        #track number of total plays
        self.t = 0
        #last arm+fidelity play
        self.last_action = None

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def make_gamma_fn(self, costs):
        """
        makes the gamma function for choosing a fidelity
        based on relative costs and mean bdry interval
        """
        self.gamma_fn = np.zeros(self.m-1)    
        for m in range(self.m-1):
            self.gamma_fn[m] = np.sqrt(costs[m]/costs[m+1]*(self.zeta[1,m]**2)) 

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._value_estimates[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        #picks action according to policy
        action = self.policy.choose(self)
        #updates memory of previous action
        self.last_action = action
        #returns the policy-chosen action
        return action

    def observe(self, reward):
        """
        Uses observation of arm reward to update estimates of arm value
        """
        #update count of plays of arm+fidelity
        self.action_attempts[self.last_action[0], self.last_action[1]] += 1

        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action[0], self.last_action[1]]
        else:
            g = self.gamma
        #get historic value estimate for the arm+fidelity
        q = self._value_estimates[self.last_action[0], self.last_action[1]]

        #update estimate of value of current arm+fidelity
        #based on # of plays (1/g), historic estimate (q), and observed reward
        self._value_estimates[self.last_action[0], self.last_action[1]] += g*(reward - q)
        self.t += 1


    @property
    def value_estimates(self):
        return self._value_estimates
