import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import itertools
import networkx as nx


class ising:        
    def __init__(self, N, graph_type, weight_mat): #(self, N, graph_type, **args):
        self.N = N
        #self.graph = nx.Graph(nx.to_undirected(graph_type(n = N, **args)))
        self.graph = graph_type
        
        self.node_init()
        self.pairs_list = copy.deepcopy(list(self.graph.edges))
        self.weight_init(weight_mat)
    
    def node_init(self):
        for i in self.graph.nodes:
            if  np.random.rand() < 0.5:
                self.graph.nodes[i]['state'] = 1
            else:
                self.graph.nodes[i]['state'] = -1
                
            nn_list = list(self.graph.neighbors(i))
            self.graph.nodes[i]['nn'] = tuple(self.graph.neighbors(i))
            
    def weight_init(self, weight_mat):
        for edge in self.pairs_list:
            self.graph[edge[0]][edge[1]]['weight'] = weight_mat[edge[0], edge[1]]

    def find_nn(self, node):
        return copy.deepcopy(np.array(self.graph.nodes[node]['nn']))
    
    def draw(self, pos = None, lattice = False):
        node_color = []
        for i in self.graph.nodes():
            if self.graph.nodes[i]['state'] == 1:
                node_color.append('red')
            if self.graph.nodes[i]['state'] == -1:
                node_color.append('blue')
        if lattice:
            nx.draw(self.graph, pos = pos, node_color = node_color, node_size = 10)
        else:
            nx.draw(self.graph, node_color = node_color, node_size = 10)
        plt.show()
        
    
    def update_state(self, belief):
        for i in self.graph.nodes:
            p_up = belief.graph.nodes[i]['node_belief'][1]
            if np.random.rand() < p_up:
                self.graph.nodes[i]['state'] = 1
            else:
                self.graph.nodes[i]['state'] = -1
                
    def entropy(self, p):
        p = np.array(p)
        return -np.sum(p*np.log(p))
                
    def cumulants(self, belief):
        # 1 point cumulants
        for i in self.graph.nodes:
            p = belief.graph.nodes[i]['node_belief']
            self.graph.nodes[i]['entropy'] = self.entropy(p)
            
        for edge in self.pairs_list:
            node_1 = edge[0]
            node_2 = edge[1]
            
            e1 = self.graph.nodes[node_1]['entropy']
            e2 = self.graph.nodes[node_2]['entropy']
            
            e12 = self.entropy(belief.graph[node_1][node_2]['pair_belief'])
            
            self.graph[node_1][node_2]['two_cumulant'] = e12 - e1 -e2
