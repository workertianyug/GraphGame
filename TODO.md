# TODO:

* Finish debugging MADDPG code.
  * Current issues:
    * Sometimes the UAV gets nan values as actions.
    * Sometimes the defender gets nan values as state values.
* Implement PPO -- Steph.
* Extend PPO to multi-agent setting -- Steph.
* Play around with graph embedding to see if we can make changes that yield better results.
* Implement similar domain without using graph embedding to run standard RL algorithms -- Steph. 
* Anything else?

# TODO (in future):
* Make current implementation more easily extensible (wrt general graphs, other algorithms, etc.)