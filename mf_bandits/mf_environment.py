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

        for _ in tqdm(range(experiments)):
            self.reset()
            while  self.agent.Lambda < COST_CONSTRAINT:
                action = self.agent.choose()
                reward, is_optimal = self.bandit.pull(action)
                self.agent.observe(reward)

                plays[action[0],action[1]]+=1
                #if is_optimal:
                #        optimal[agent.t, i] += 1
            #print(plays)    
                   

        return plays / experiments

    def plot_plays(self, plays):
        arms = np.linspace(0,499, 500)
        ax = plt.figure()
        for m in range(self.agent.m):
            plt.plot(arms, plays[:,m])
        # histogram= plt.figure()
        # bins = np.linspace(1,500,500)
        # #for m in range(self.agent.m):
        # #    plt.hist(plays[:,m,:], bins) 
        # plt.hist(plays[:,0], bins, facecolor='blue', alpha=0.5)
        # plt.hist(plays[:,1], bins, facecolor='green',alpha=0.5)
        # plt.hist(plays[:,2], bins, facecolor='red', alpha=0.5)
        plt.show()   

    def plot_results(self, scores, optimal):
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(scores)
        plt.ylabel('Average Reward')
        plt.legend(self.agents, loc=4)
        plt.subplot(2, 1, 2)
        plt.plot(optimal * 100)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        sns.despine()
        plt.show()

    def plot_beliefs(self):
        sns.set_context('talk')
        pal = sns.color_palette("cubehelix", n_colors=len(self.agents))
        plt.title(self.label + ' - Agent Beliefs')

        rows = 2
        cols = int(self.bandit.k / 2)

        axes = [plt.subplot(rows, cols, i+1) for i in range(self.bandit.k)]
        for i, val in enumerate(self.bandit.action_values):
            color = 'r' if i == self.bandit.optimal else 'k'
            axes[i].vlines(val, 0, 1, colors=color)

        for i, agent in enumerate(self.agents):
            if type(agent) is not BetaAgent:
                for j, val in enumerate(agent.value_estimates):
                    axes[j].vlines(val, 0, 0.75, colors=pal[i], alpha=0.8)
            else:
                x = np.arange(0, 1, 0.001)
                y = np.array([stats.beta.pdf(x, a, b) for a, b in
                             zip(agent.alpha, agent.beta)])
                y /= np.max(y)
                for j, _y in enumerate(y):
                    axes[j].plot(x, _y, color=pal[i], alpha=0.8)

        min_p = np.argmin(self.bandit.action_values)
        for i, ax in enumerate(axes):
            ax.set_xlim(0, 1)
            if i % cols != 0:
                ax.set_yticklabels([])
            if i < cols:
                ax.set_xticklabels([])
            else:
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0', '', '0.5', '', '1'])
            if i == int(cols/2):
                title = '{}-arm Bandit - Agent Estimators'.format(self.bandit.k)
                ax.set_title(title)
            if i == min_p:
                ax.legend(self.agents)

        sns.despine()
        plt.show()