#mytry.py
#My own try at implementing multi-fidelity multi-armed bandits w/ UCB alg
#based on
#Byron Galbraith's example of bernoulli and binomial bandits
#source: https://github.com/bgalbraith/bandits/blob/master/examples/bayesian.py


import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from policy import UCBPolicy
from environment import Environment
from mf_bandits import MF_GaussianBandit

class GaussianEx(object):
    """
    Multifidelity multiarmed Bandit
    w/ Gaussian rewards
    High fidelities dist. w/ means mu_k uniform in grid (0,1)
    Fidelity "m" means dist. uniformly within a +- zeta(m) band about mu_k
    """
    def __init__(self, no_arms, no_fids, zeta, high_fid_mean_dist='unif'):
        self.label = 'Multi-armed Bandits - Gaussian'
        self.no_arms = no_arms
        self.no_fids = no_fids
        self.bandit = self.make_bandit(zeta, high_fid_mean_dist)
        self.agents = [Agent(self.bandit, UCBPolicy(1))]

    def make_bandit(self,zeta, high_fid_mean_dist):        
        #pick high fidelity means either as a uniform grid in (0,1)
        #or sampled from a N(0,1) dist, sorted low->high
        if high_fid_mean_dist=='unif':
            high_fid_means = np.linspace(0.0,1.0, num=self.no_arms)
        if high_fid_mean_dist=='normal':
            high_fid_means = np.random.normal(0.0, 1.0, size=self.no_arms)
            high_fid_means = np.sort(high_fid_means) 

        #initialize matrix of all fidelity means
        self.fid_means = np.zeros((self.no_arms,self.no_fids-1))
        for m in range(self.no_fids-1):
            #create zeta interval about high fid. mean
            upper_bd = high_fid_means+zeta[m]
            lower_bd = high_fid_means-zeta[m]
            #sample low fidelity means uniformly from the zeta interval
            self.fid_means[:,m] = np.random.uniform(lower_bd, upper_bd, size=self.no_arms)
        #add high fidelity means to fidelity mean matrix    
        self.fid_means = np.c_[self.fid_means, high_fid_means]    
        #create MF-MA bandit w/ Gaussian rewards w/ means according to fid. means matrix
        return MF_GaussianBandit(k=self.no_arms, m=self.no_fids, mu=self.fid_means, sigma=0.2)
    
    def plot_means(self):
        plt.plot(self.fid_means, '.')
        plt.show()
        plt.close()


if __name__ == '__main__':
    experiments = 500
    trials = 1000
    zeta = [0.2, 0.1, 0]
    example = GaussianEx(no_arms=500, no_fids=3, zeta=zeta)
    example.plot_means()

    zeta2 = [1, 0.5, 0.2, 0]
    example2 = GaussianEx(no_arms=500, no_fids=4, zeta=zeta2, high_fid_mean_dist='normal')
    example2.plot_means()

    env = Environment(example.bandit, example.agents, example.label)
    scores, optimal = env.run(trials, experiments)
    env.plot_results(scores, optimal)
    env.plot_beliefs()