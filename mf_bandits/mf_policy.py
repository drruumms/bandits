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
        #compute exploration term of UCB
        exploration = np.log(agent.t+1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        #recall that rows=arms, cols=fidelities, compute total UCB for each arm+fidelity
        q = agent.value_estimates + exploration + agent.zeta
        #get min q bounds across fidelities for each arm
        min_arm_bounds = np.amin(q, axis=1)
        #pick arm that maximizes min arm bounds
        max_arm_index = np.argmax(min_arm_bounds)
        #pick lowest, most uncertain fidelity
        for m in range(agent.m-1):
            action = None
            if exploration[max_arm_index, m]>=agent.gamma_fn[m]:
                #print('playing arm ', max_arm_index)
                #print('playing fidelity ', m)
                action=[max_arm_index, m]
                break    
        if action==None:
            #if low fidelities all certain, play at highest fidelity
            action=[max_arm_index,agent.m-1]

        return action
        # check = np.where(q == action)[0]
        # if len(check) == 0:
        #     return action
        # else:
        #     return np.random.choice(check)