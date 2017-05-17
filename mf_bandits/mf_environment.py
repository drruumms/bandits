#mf_environment.py
#multifidelity extension of
#Byron Galbraith's multi-armed bandit simulator Environment class
#source: https://github.com/bgalbraith/bandits/blob/master/bandits/environment.py


#environment object than runs experiments and 
#provides some results plotting methods

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm

from mf_agent import Agent


class Environment(object):
    def __init__(self, bandit, agent, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agent = agent
        self.label = label

    def reset(self):
        self.bandit.reset()
        self.agent.reset()

    def run(self, COST_CONSTRAINT, experiments=1):
        #keep track of plays at each arm+fidelity across experiments
        plays = np.zeros((self.agent.k, self.agent.m))
        ave_regret = 0

        for _ in range(experiments):
            self.reset()
            #initialize regret
            regret = COST_CONSTRAINT*self.bandit.pull(self.bandit.optimal)
            print("opt arm is arm %d" %self.bandit.optimal[0])
            while  self.agent.Lambda <= COST_CONSTRAINT:
                action = self.agent.choose()
                reward = self.bandit.pull(action)
                self.agent.observe(reward)

                arm_index = action[0]
                fidelity_index = action[1]
                #print("arm= %d" %arm_index)
                #print("fidelity = %d" %fidelity_index)
                if self.agent.Lambda <= COST_CONSTRAINT:
                    plays[arm_index, fidelity_index]+=1
                    regret -= self.bandit.costs[fidelity_index]*self.bandit.pull([arm_index, self.bandit.m-1])

            ave_regret+= regret  
                   
        return plays / experiments, ave_regret / experiments

    def plot_plays(self, plays):
        arms = np.linspace(0,499, 500)
        colors = ['r', 'b', 'g', 'y']
        plt.figure()
        for m in range(self.agent.m):
            plt.bar(arms, plays[:,m], label="fidelity {0}".format(m), color = colors[m])
        plt.legend(loc=0)
        axes= plt.gca()
        axes.set_xlim([0, 510])    
        plt.show()

    def plot_cost_vs_regret(self, cost_constraints, regrets):
         #plot regret vs cost
        plt.plot(cost_constraints, regrets)
        axes = plt.gca()
        axes.set_xlim([np.amin(cost_constraints), np.amax(cost_constraints)])
        axes.set_ylim([np.amin(regrets),np.amax(regrets)])
        plt.show()

