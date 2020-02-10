import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import itertools
import networkx as nx


class messages:
    
    def __init__(self, ising_model, T):
        self.graph = copy.deepcopy(ising_model.graph)
        self.beta = 1/T
        
        # initialize to 0.1 the msgs to down spins
        # and to 1 the msgs to up spins
        for i in self.graph.nodes:
            l = len(ising_model.find_nn(i))
            self.graph.nodes[i]['inc_msg'] = [np.ones(l)*0.1, np.ones(l)]
    
    def cost(self, node_1, node_2, s1, s2, real_state = False):
        if real_state:
            s1 = self.graph.nodes[node_1]['state']
            s2 = self.graph.nodes[node_2]['state']
        J = self.graph[node_1][node_2]['weight']
        return np.exp(self.beta*s1*s2*J)
    
    # returns the incoming messages to a node
    def get_msg_node(self, node):
        return copy.deepcopy(self.graph.nodes[node]['inc_msg'])
    
    def msg_vartovar(self, current_node, nn_node, current_state, ising_model):
        ##########################################################################
        ###   since we are saving incoming messages only, in order to update   ###
        ###   the message from j to i we need all the other incoming messages  ###
        ###   to j, that is                                                    ###
        ###                                                                    ###
        ###            ↓                                                       ###
        ###          → j ←                                                     ###
        ###                                                                    ###
        ###   so all except the one in the direction of i                      ###
        ###                                                                    ###
        ###   then the update from H to i is given by the marginalization      ###
        ###   over the states of j (since we are saving the up-incoming        ###
        ###   message for i) times the cost exp(beta*sigma_i*sigma_j)          ###
        ##########################################################################
        
        # compute message from j to i
        # find all the incoming message of node j
        msg_nn = np.array(self.get_msg_node(nn_node))
        # discard the message from i to j
        dlist = list(np.where(ising_model.find_nn(nn_node) != current_node))[0]
        msg_vtv = 0
        
        # marginalize over the states of j
        for idx_state, nn_state in enumerate([-1,1]):
            msg = 1
            # product of the messages
            for m in msg_nn[idx_state, dlist]:
                msg *= m
            # cost
            msg *= self.cost(current_node, nn_node, current_state, nn_state)
            # marginalization
            msg_vtv += msg
        
        return msg_vtv     
    
    
    def update_msg(self, ising_model):
        # create the list of new messages for the parallel update
        new_msg_list = copy.deepcopy(np.array(self.graph.nodes.data('inc_msg'))[:,1])
                
        # iterate over all the graph
        for node in self.graph.nodes():
            # iterate over the nn of the current node
            for idx_nn, nn_node in enumerate(ising_model.find_nn(node)):
                # iterate over the possible states of the node
                for idx_state, node_state in enumerate([-1,1]):
                    msg = self.msg_vartovar(node, nn_node, node_state, ising_model)
                    
                    # update must be done in parallel!
                    new_msg_list[node][idx_state][idx_nn] = msg
        
        
        
        # normalize the messages
        mse = []
        for node, inc_msg in enumerate(new_msg_list):
            inc_msg /= np.sum(inc_msg, axis = 0)
            error = np.array(self.graph.nodes[node]['inc_msg']) - inc_msg
            mse.append(np.mean(error**2))
            self.graph.nodes[node]['inc_msg'] = inc_msg
        
        return np.mean(mse)
