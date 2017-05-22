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

    # #Simple Example - Single Fidelity
    # means = np.zeros((5,1))
    # means[:,0] = [0.8, 0.6, 0.6, 0.3, 0.1]
    # costs=[25]

    # #create MA bandit w/ gaussian rewards + these means
    # bandit = MF_GaussianBandit(k=5, m=1, mu=means,sigma=0.0001, zeta=0, costs=costs)
    # plt.figure(1)
    # plt.bar([1,2,3,4,5],bandit.action_values, color='r')
    # plt.show()

    # #create agent implementing UCB alg w/ psi(x)=x^2, rho=2
    # psi_inv = make_psi_inv(1/2)
    # agent = Agent(bandit, UCBPolicy(rho=2, psi_inv=psi_inv))

    # #Run simulation for Example bandit
    # env1 = Environment(bandit, agent, 'Simple ex')

    # plays, regrets = env1.run(10000, experiments)
    # plt.figure(2)
    # plt.bar([1,2,3,4,5], plays[:,0], color='r')
    # plt.show()
 

    #Simple Example - Multifidelity
    means = np.zeros((5,3))
    means[:,0] = [0.8, 1.1, 1.0, 1.0, 1.1]
    means[:,1] = [0.8, 0.9, 0.4, 0.4, 0.5]
    means[:,2] = [0.8, 0.7, 0.6, 0.6, 0.7]
    zeta = [0.4, 0.2, 0]
    zeta = np.broadcast_to(zeta, (5,3))
    costs=[1,5,25]

    #create MA bandit w/ gaussian rewards + these means
    bandit = MF_GaussianBandit(k=5, m=3, mu=means,sigma=0.0001, zeta=zeta, costs=costs)
    plt.figure(3)
    plt.title('Arm rewards (red = highest fidelity, blue=lowest)')
    plt.bar([1,2,3,4,5],bandit.action_values[:,2], color='r')
    plt.bar([1,2,3,4,5],bandit.action_values[:,1], color='g')
    plt.bar([1,2,3,4,5],bandit.action_values[:,0], color='b')
    plt.show()

    #create MF agent implementing UCB alg w/ psi(x)=x^2, rho=2
    psi_inv = make_psi_inv(1/2)
    agent = Agent(bandit, MF_UCBPolicy(rho=2, psi_inv=psi_inv))

    #single fidelity agent on same bandit (highest fidelity)
    agent2 = Agent(bandit, UCBPolicy(rho=2, psi_inv=psi_inv))

    # #Run simulation for MF bandit
    env_mf = Environment(bandit, agent, 'simple MF ex')
    env_single = Environment(bandit, agent2, 'single fid ex')

    plays, regrets, optimal_pulls = env_mf.run(50000, experiments)
    plays2, regrets2, optimal_pulls2 = env_single.run(50000, experiments)
    print(optimal_pulls)
    print(optimal_pulls2)
    plt.figure(4)
    plt.title('Arm plays (red = highest fidelity, blue=lowest)')
    plt.bar([1,2,3,4,5], plays[:,2], color='r')
    plt.show()
    plt.bar([1,2,3,4,5], plays[:,1], color='g')
    plt.show()
    plt.bar([1,2,3,4,5], plays[:,0], color='b')
    plt.show()
    plt.figure(5)
    plt.title('Arm plays (single fidelity)')
    plt.bar([1,2,3,4,5], plays2[:,2], color='k')
    plt.show()

    #regret vs cost increases 
    cost_constraints = [0, 50, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
    print(cost_constraints)
    regrets = np.zeros_like(cost_constraints)
    regrets2 = np.zeros_like(regrets)

    for k in tqdm(range(len(cost_constraints))):
        plays, regrets[k], optimal_pulls = env_mf.run(cost_constraints[k], experiments)
        plays2, regrets2[k], optimal_pulls2 = env_single.run(cost_constraints[k], experiments)
        print("optimal pulls w/ MF = %d" %optimal_pulls)
        print("opt pulls w/ single fid = %d" %optimal_pulls2)
        #plot arm+fidelity plays
        # plt.figure(6+k)
        # plt.title('Arm plays at total cost %d' %cost_constraints[k])
        # plt.bar([1,2,3,4,5], plays2[:,2], color='k')
        # plt.bar([1,2,3,4,5], plays[:,2], color='r')
        # plt.bar([1,2,3,4,5], plays[:,1], color='g')
        # plt.bar([1,2,3,4,5], plays[:,0], color='b')
        # plt.show()
        # plt.close()

    #plot regret vs cost
    #print(regrets)
    #print(regrets2)
    plt.figure(16)
    plt.plot(cost_constraints, regrets, color='b')
    plt.plot(cost_constraints, regrets2, color='r')
    plt.title('Regret vs Total Cost Constraint (MF=blue, single=red)')
    axes = plt.gca()
    #axes.set_xlim([np.amin(cost_constraints), np.amax(cost_constraints)])
    #axes.set_ylim([np.amin(regrets),np.amax(regrets2)])
    plt.show()
