#simple_Ex.py
#Simple example to demonstrate whether UCB and MF-UCB alg are working properly


import matplotlib.pyplot as plt
import numpy as np
from mf_agent import Agent
from mf_policy import MF_UCBPolicy, UCBPolicy
from mf_environment import Environment
from mf_bandits import MF_GaussianBandit
from tqdm import tqdm

def make_psi_inv(pow):
    def psi_inv(exploration):
        return np.power(exploration, pow)
    return psi_inv     

if __name__ == '__main__':
    experiments = 1

    #Simple Example - Single Fidelity
    means = np.zeros((5,1))
    means[:,0] = [0.8, 0.6, 0.6, 0.3, 0.1]
    costs=[25]

    #create MA bandit w/ gaussian rewards + these means
    bandit = MF_GaussianBandit(k=5, m=1, mu=means,sigma=0.0001, zeta=0, costs=costs)
    plt.figure(1)
    plt.bar([1,2,3,4,5],bandit.action_values, color='r')
    plt.show()

    #create agent implementing UCB alg w/ psi(x)=x^2, rho=2
    psi_inv = make_psi_inv(1/2)
    agent = Agent(bandit, UCBPolicy(rho=2, psi_inv=psi_inv))

    #Run simulation for Example bandit
    env1 = Environment(bandit, agent, 'Simple ex')

    plays, regrets = env1.run(10000, experiments)
    plt.figure(2)
    plt.bar([1,2,3,4,5], plays[:,0], color='r')
    plt.show()
 

    #Simple Example - Multifidelity
    means = np.zeros((5,3))
    means[:,0] = [0.8, 0.8, 0.8, 0.8, 0.1]
    means[:,1] = [0.8, 0.8, 0.6, 0.3, 0.1]
    means[:,2] = [0.8, 0.6, 0.6, 0.3, 0.1]
    zeta = [0.5, 0.1, 0]
    zeta = np.broadcast_to(zeta, (5,3))
    costs=[1,5,10]

    #create MA bandit w/ gaussian rewards + these means
    bandit = MF_GaussianBandit(k=5, m=3, mu=means,sigma=0.001, zeta=zeta, costs=costs)
    plt.figure(3)
    plt.bar([1,2,3,4,5],bandit.action_values[:,0], color='b')
    plt.bar([1,2,3,4,5],bandit.action_values[:,1], color='g')
    plt.bar([1,2,3,4,5],bandit.action_values[:,2], color='r')
    plt.show()

    #create MF agent implementing UCB alg w/ psi(x)=x^2, rho=2
    psi_inv = make_psi_inv(1/2)
    agent = Agent(bandit, MF_UCBPolicy(rho=2, psi_inv=psi_inv))

    #single fidelity agent on same bandit
    agent2 = Agent(bandit, UCBPolicy(rho=2, psi_inv=psi_inv))

    # #Run simulation for MF bandit
    env1 = Environment(bandit, agent, 'simple MF ex')
    env2 = Environment(bandit, agent2, 'single fid ex')

    plays, regrets = env1.run(50000, experiments)
    plays2, regrets2 = env2.run(50000, experiments)
    print(plays2)
    plt.figure(4)
    plt.bar([1,2,3,4,5], plays[:,0], color='b')
    plt.bar([1,2,3,4,5], plays[:,1], color='g')
    plt.bar([1,2,3,4,5], plays[:,2], color='r')
    plt.show()
    plt.figure(5)
    plt.bar([1,2,3,4,5], plays2[:,2], color='r')
    plt.show()

    # #regret vs cost increases 
    # cost_constraints = np.linspace(50000, 500000, num=10)
    # print(cost_constraints)
    # regrets = np.zeros_like(cost_constraints)
    # regrets2 = np.zeros_like(regrets)

    # for k in tqdm(range(10)):
    #     plays, regrets[k] = env1.run(cost_constraints[k], experiments)
    #     plays2, regrets2[k] = env2.run(cost_constraints[k], experiments)
    #     #plot arm+fidelity plays
    #     plt.figure(5)
    #     plt.bar([1,2,3,4,5], plays[:,0], color='b')
    #     plt.bar([1,2,3,4,5], plays[:,1], color='g')
    #     plt.bar([1,2,3,4,5], plays[:,2], color='r')
    #     plt.show()

    # #plot regret vs cost
    # #env1.plot_cost_vs_regret(cost_constraints, regrets)
    # #plot regret vs cost
    # plt.plot(cost_constraints, regrets, color='b')
    # plt.plot(cost_constraints, regrets2, color='r')
    # axes = plt.gca()
    # axes.set_xlim([np.amin(cost_constraints), np.amax(cost_constraints)])
    # axes.set_ylim([np.amin(regrets),np.amax(regrets2)])
    # plt.show()
