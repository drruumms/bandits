#Multifidelity Extension of the Multi-Armed Bandit

Extends multi-armed bandit classes to include multi-fidelity versions, and implements the MF-UCB algorithm as detailed in [Kandasamy et. al.]()

---

Multi-fidelity multi-armed bandits have several 'versions' of the same arms, with increasing fidelities and costs.  Below is an example of the mean rewards per arm and fidelity of a 5-armed, 3-fidelity Gaussian-reward bandit.  Fidelity 0 are the low fidelity, high bias/variance observations of the bandit arms, and the error bars denote the maximum distance away from the true mean reward possible in the MF-UCB algorithm.
![alt text](https://github.com/drruumms/bandits/blob/master/mf_bandits/mf_rewards_case3.png "MF-MA Bandit Rewards")

In this case, the lower fidelity arms contain enough information about the high fidelity arms for a MF-UCB agent to leverage, so the overall regret is lower compared to a UCB agent playing only the high fidelity arms.
![alt text](https://github.com/drruumms/bandits/blob/master/mf_bandits/mf_plays_case3.png "MF-UCB vs. UCB Plays")
![alt text](https://github.com/drruumms/bandits/blob/master/mf_bandits/mf_regret_Case3.png "MF-UCB vs. UCB Regret")