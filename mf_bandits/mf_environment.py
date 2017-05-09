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
        #scores = np.zeros((10000000, len(self.agents)))
        #optimal = np.zeros_like(scores)
        #keep track of plays at each arm+fidelity across experiments
        plays = np.zeros((self.agent.k, self.agent.m, experiments))
        regret = COST_CONSTRAINT*self.bandit.pull(self.bandit.optimal)
        regretS = [regret]

        for _ in tqdm(range(experiments)):
            self.reset()
            while  self.agent.Lambda < COST_CONSTRAINT:
                action = self.agent.choose()
                reward = self.bandit.pull(action)
                self.agent.observe(reward)

                arm_index = action[0]
                fidelity_index = action[1]

                plays[arm_index, fidelity_index]+=1
                regret -= self.bandit.costs[fidelity_index]*self.bandit.pull([arm_index, self.bandit.m-1])
                regretS = np.c_[regretS, regret]
                #if is_optimal:
                #        optimal[agent.t, i] += 1
            #print(plays)    
                   

        print(regretS.shape); raise
        return plays / experiments, regretS / experiments

    def plot_plays(self, plays):
        arms = np.linspace(0,499, 500)
        ax = plt.figure()
        for m in range(self.agent.m):
            plt.plot(arms, plays[:,m], label="fidelity %d" %m)
        plt.legend(loc=0)    
        # histogram= plt.figure()
        # bins = np.linspace(1,500,500)
        # #for m in range(self.agent.m):
        # #    plt.hist(plays[:,m,:], bins) 
        # plt.hist(plays[:,0], bins, facecolor='blue', alpha=0.5)
        # plt.hist(plays[:,1], bins, facecolor='green',alpha=0.5)
        # plt.hist(plays[:,2], bins, facecolor='red', alpha=0.5)
        plt.show()   

    def plot_regret(self, regret):
        plt.figure()
        plt.plot(regret)
        plt.show()   
