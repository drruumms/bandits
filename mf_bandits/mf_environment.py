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

    def pull_all_arms(self, plays, regret):
        #pull each arm once at each fidelity to intialize
        for k in range(self.bandit.k):
            #reward for pulling highest fidelity for this arm
            high_fid_reward, dont_use = self.bandit.pull([k, self.bandit.m-1])
            for m in range(self.bandit.m):
                #get reward for arm k fidelity m
                action = [k,m]
                reward, dont_use = self.bandit.pull(action)
                #updates memory of previous action
                self.agent.last_action = action
                #cost of pulling arm            
                pull_cost = self.bandit.costs[m]
                #observe and record plays, rewards, regrets
                self.agent.observe(reward)
                plays[k,m] +=1
                regret -= pull_cost*high_fid_reward
        #return play count, regret        
        return plays, regret        

    def run(self, COST_CONSTRAINT, experiments=1):
        #keep track of plays at each arm+fidelity across experiments
        plays = np.zeros((self.agent.k, self.agent.m))
        ave_regret = 0
        optimal_pulls = 0

        for _ in range(experiments):
            self.reset()
            #initialize regret
            optimal_reward, dont_use = self.bandit.pull(self.bandit.optimal)
            regret = COST_CONSTRAINT*optimal_reward

            #pull each arm once at each fidelity to intialize
            plays, regret = self.pull_all_arms(plays, regret)

            #print("opt arm is arm %d" %self.bandit.optimal[0])
            while  self.agent.Lambda <= COST_CONSTRAINT:
                action = self.agent.choose()
                reward, optimal_pull = self.bandit.pull(action)

                arm_index = action[0]
                fidelity_index = action[1]
                high_fid_reward, dont_use = self.bandit.pull([arm_index, self.bandit.m-1])
                #print("arm= %d" %arm_index)
                #print("fidelity = %d" %fidelity_index)
                pull_cost = self.bandit.costs[fidelity_index]
                optimal_pulls += optimal_pull

                if self.agent.Lambda+pull_cost <= COST_CONSTRAINT:
                    self.agent.observe(reward)
                    plays[arm_index, fidelity_index]+=1
                    regret -= pull_cost*high_fid_reward
                else:
                    break
            ave_regret+= regret             
                   
        return plays / experiments, ave_regret / experiments, optimal_pulls / experiments

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

