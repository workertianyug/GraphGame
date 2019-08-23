import networkx as nx
import numpy as np
from env import *

class HeurAtt(object):
    def __init__(self):
        self.seed = 17
        self.rand = np.random.RandomState(seed=self.seed)
        self.lastAction = None
        
    def reset(self):
        self.lastAction = None

    def act(self, attState, attNode, eps=0):
        if (attState.nodes[attNode]["isAtt"] != 1):
            raise ValueError("att location doesn't match")

        if ((self.lastAction != None)
            and (self.lastAction[0] == self.lastAction[1])
            and (attState.nodes[attNode]["r"] > 0)):
            return self.lastAction

        # determine the best node to pick  
        validActions = list(attState.out_edges([attNode]))
        validNodes = list(attState.neighbors(attNode))
        current_reward = attState.nodes[attNode]["r"]
        
        rewards = []
        for i in range(len(validNodes)):
            rewards.append(attState.nodes[validNodes[i]]["r"])
        idx = np.argmax(rewards)
            
        # pick an action 
        for potential_action in attState.out_edges([validNodes[idx]]):
            for valid_action in validActions:
                if sorted(potential_action) == sorted(valid_action):
                    # temp stopgap, to get it out of noops 
                    if valid_action[0] == valid_action[1]:
                        break
                    self.lastAction = valid_action
                    return valid_action

    def train(self, *arg):
        return 42
        
