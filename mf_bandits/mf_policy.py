#mf_policy.py
#multi-fidelity extension of
#Byron Galbraith's multi-armed bandit simulator Policy classes
#source: https://github.com/bgalbraith/bandits/blob/master/bandits/policy.py


import numpy as np


class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0


class UCBPolicy(Policy):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def choose(self, agent):
        exploration = np.log(agent.t+1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        q = agent.value_estimates + exploration
        action = np.argmax(q)
        # check = np.where(q == action)[0]
        # if len(check) == 0:
        #     return action
        # else:
        #     return np.random.choice(check)

class MF_UCBPolicy(Policy):
    """
    The Multi-Fidelity Upper Confidence Bound algorithm (MF-UCB). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def choose(self, agent):
        exploration = np.log(agent.t+1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        #recall that rows=arms, cols=fidelities
        q = agent.value_estimates + exploration + agent.zeta
        #get min q bounds across fidelities for each arm
        min_arm_bounds = np.amin(q, axis=1)
        min_fid_indeces = np.argmin(q,axis=1)
        max_arm_index = np.argmax(min_arm_bounds)
        action = [max_arm_index, min_fid_indeces[max_arm_index]]
        return action
        # check = np.where(q == action)[0]
        # if len(check) == 0:
        #     return action
        # else:
        #     return np.random.choice(check)