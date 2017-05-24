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

    #Simple Example - Multifidelity
    means = np.zeros((5,3))

    #Lowest regret for MF-UCB, lower fidelities undervalue subopt arms,
    # and overvalue opt arm
    means[:,0] = [1.2, 0.3, 0.2, 0.2, 0.3]
    means[:,1] = [1.0, 0.5, 0.4, 0.4, 0.5]
    means[:,2] = [0.8, 0.7, 0.6, 0.6, 0.7]   
    zeta = [0.4, 0.2, 0]

    #Lower regret for MF-UCB, lowest fidelities overvalue, but w/ opt arm highest
    #     mid fidelity undervalues subopt arms, overvalues opt
    # means[:,0] = [1.2, 1.1, 1.0, 1.0, 1.1]
    # means[:,1] = [1.0, 0.5, 0.4, 0.4, 0.5]
    # means[:,2] = [0.8, 0.7, 0.6, 0.6, 0.7]   
    # zeta = [0.4, 0.2, 0]

    #Lower regret for MF-UCB, Mid-fidelity upset
    # means[:,0] = [0.8, 1.1, 1.0, 1.0, 1.1]
    # means[:,1] = [0.8, 0.9, 0.4, 0.4, 0.5]
    # means[:,2] = [0.8, 0.7, 0.6, 0.6, 0.7]
    #zeta = [0.4, 0.2, 0]

    #Higher regret for MF-UCB, Mid-fidelity too close together
    #      mid-fid bandit problem consumes too many resources
    # means[:,0] = [1.2, 1.1, 1.0, 1.0, 1.1]
    # means[:,1] = [1.0, 0.8, 0.8, 0.8, 0.8]
    # means[:,2] = [0.8, 0.7, 0.6, 0.6, 0.7]   
    # zeta = [0.4, 0.2, 0]

    #Very bad case - Low+mid fidelities miscalculate best arm
    means[:,0] = [0.8, 1.1, 1.0, 1.0, 1.1]
    means[:,1] = [0.8, 0.9, 0.8, 0.8, 0.9]
    means[:,2] = [0.8, 0.7, 0.6, 0.6, 0.7]   
    zeta = [0.4, 0.2, 0]
    
    zeta = np.broadcast_to(zeta, (5,3))
    costs=[1,5,25]

    #create MA bandit w/ gaussian rewards + these means
    bandit = MF_GaussianBandit(k=5, m=3, mu=means,sigma=0.0001, zeta=zeta, costs=costs)
    fig, ax = plt.subplots()
    index = np.arange(zeta.shape[0])
    bar_width=0.25
    opacity=0.4
    error_config={'ecolor':'0.3'}
    rews0 = plt.bar(index, bandit.action_values[:,0], bar_width, alpha=opacity,
             color='b', yerr=zeta[0,0], error_kw=error_config, label='Fid 0')
    rews1 = plt.bar(index + bar_width, bandit.action_values[:,1], bar_width, alpha=opacity,
                    color='g', yerr=zeta[:,1], error_kw=error_config, label='Fid 1')
    rews2 = plt.bar(index + 2*bar_width, bandit.action_values[:,2], bar_width, alpha=opacity,
                    color='r', yerr=zeta[:,2], error_kw=error_config, label='Fid 2')
    plt.xlabel('Arm')
    plt.ylabel('Rewards')
    plt.title('Arm rewards')
    plt.legend()
    plt.show()

    #create MF agent implementing UCB alg w/ psi(x)=x^2, rho=2
    psi_inv = make_psi_inv(1/2)
    agent = Agent(bandit, MF_UCBPolicy(rho=2, psi_inv=psi_inv))

    #single fidelity agent on same bandit (highest fidelity)
    agent2 = Agent(bandit, UCBPolicy(rho=2, psi_inv=psi_inv))

    # #Run simulation for MF bandit
    env_mf = Environment(bandit, agent, 'simple MF ex')
    env_single = Environment(bandit, agent2, 'single fid ex')

    cost_constraints = 50000

    plays, regrets, optimal_pulls = env_mf.run(cost_constraints, experiments)
    plays2, regrets2, optimal_pulls2 = env_single.run(cost_constraints, experiments)
    print(optimal_pulls)
    print(optimal_pulls2)
    fig, ax = plt.subplots()
    bar_width=0.15
    arm_plays0 = plt.bar(index, plays[:,0], bar_width, alpha=opacity,
             color='b', error_kw=error_config, label='Fid 0 (MF-UCB)')
    arm_plays1 = plt.bar(index + bar_width, plays[:,1], bar_width, alpha=opacity,
                    color='g', error_kw=error_config, label='Fid 1 (MF-UCB)')
    arm_plays2 = plt.bar(index + 2*bar_width, plays[:,2], bar_width, alpha=opacity,
                    color='r',  error_kw=error_config, label='Fid 2 (MF-UCB)')
    arm_plays_sf = plt.bar(index + 3*bar_width, plays2[:,2], bar_width, alpha=opacity,
                    color='k',  error_kw=error_config, label='Fid 2 (UCB)')
    plt.xlabel('Arm')
    plt.ylabel('Plays')
    plt.title('Arm play count w/ resources %d' %cost_constraints)
    plt.legend()
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


    #plot regret vs cost
    #print(regrets)
    #print(regrets2)
    plt.plot(cost_constraints, regrets, color='b', label='MF-UCB')
    plt.plot(cost_constraints, regrets2, color='r', label='UCB')
    plt.title('Regret vs Total Cost Constraint')
    plt.legend()
    plt.show()
