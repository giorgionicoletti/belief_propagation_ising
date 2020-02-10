import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import itertools
import networkx as nx


class beliefs:
    
    def __init__(self, messages, ising_model):
        # in this graph we will also attach beliefs
        self.graph = copy.deepcopy(messages.graph)
        self.beta = messages.beta
        self.pairs_list = ising_model.pairs_list
        
    def ev_site_bel(self):
        for i in self.graph.nodes:
            sbel = np.prod(self.graph.nodes[i]['inc_msg'], axis = 1)
            sbel /= np.sum(sbel)
            self.graph.nodes[i]['node_belief'] = sbel
            self.graph.nodes[i]['magnetization'] = np.diff(sbel)
        
    def ev_pair_bel(self, messages, ising_model):
        for edge in self.pairs_list:
            node_1 = edge[0]
            node_2 = edge[1]
            J = self.graph[node_1][node_2]['weight']
            
            dlist_1 = list(np.where(ising_model.find_nn(node_1) != node_2))[0]
            dlist_2 = list(np.where(ising_model.find_nn(node_2) != node_1))[0]
            
            msg_1 = messages.get_msg_node(node_1)
            msg_1 = np.array(msg_1)[:,dlist_1]
            msg_2 = messages.get_msg_node(node_2)
            msg_2 = np.array(msg_2)[:,dlist_2]
            
            msg_prod = [np.prod(msg_1, axis = 1),
                        np.prod(msg_2, axis = 1)]
            
            pbel = np.ones(4)

            for idx_state, pair_state in enumerate([(-1,-1), (1,-1), (1,-1),(1,1)]):
                cost = J
                for i, s in enumerate(pair_state):
                    cost *= s
                    if s == -1:
                        pbel[idx_state] *= msg_prod[i][0]
                    if s == 1:
                        pbel[idx_state] *= msg_prod[i][1]
                pbel[idx_state] *= np.exp(self.beta*cost)
            
            self.graph[node_1][node_2]['pair_belief'] = pbel/np.sum(pbel)
