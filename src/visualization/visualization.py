#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt


class Visualization(object):
    def __init__(self):
        pass
    
    def network(self, network):
        G = nx.DiGraph()
        
        edges, val_map, weights = self.get_edges(network)
        G.add_edges_from(edges)
        
        values = [val_map.get(node, 0.25) for node in G.nodes()]
        pos = self.get_position(network)
        
        plt.figure(1, figsize=(5.1,3))
        nx.draw(G, pos, node_color = values, node_size = 400, vmin=0, vmax=len(network.hidden)+2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, label_pos=0.7)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True)

        plt.title("Multi-layer Perceptron", fontweight="bold")
        plt.show()
        
    def learning_curve(self, mses):
        plt.title("Learning Curve")
        plt.ylabel("Mean Squared Error")
        plt.xlabel("Cycles")
        plt.plot(mses)
        plt.show()

    def get_edges(self, network):
        val_map = {}
        weights = {}
        edges = []
        
        hidden_nodes = []
        for nodes in network.hidden:
            for node in nodes:
                hidden_nodes.append(node) 
        
        for node in network.inputs+hidden_nodes+network.outputs:
            id_node = node.id+node.layer
            val_map[id_node] = int(node.layer)
            for dest in node.synapses:
                if dest.start == node:
                    id_dest = dest.end.id+dest.end.layer
                    edge = (id_node, id_dest)
                    edges.append(edge)
                    weights[edge] = "W"+dest.end.id+node.id
        
        return edges, val_map, weights
    
    def get_position(self, network):
        pos = {}
        num_in = len(network.inputs)
        num_hid_lyr = len(network.hidden)
        num_out = len(network.outputs)
        
        curr_x = 0
        curr_y = num_in
        for i in range(num_in):
            id_node = network.inputs[i].id+network.inputs[i].layer
            pos[id_node] = [curr_x, curr_y]
            curr_y -= 2
        curr_x += 2
        
        for layer in range(num_hid_lyr):
            curr_y = len(network.hidden[layer])
            for i in range(len(network.hidden[layer])):
                id_node = network.hidden[layer][i].id+network.hidden[layer][i].layer
                pos[id_node] = [curr_x, curr_y]
                curr_y -= 2
            curr_x += 2
        
        curr_y = num_out
        for i in range(num_out):
            id_node = network.outputs[i].id+network.outputs[i].layer
            pos[id_node] = [curr_x, curr_y]
            curr_y -= 2
        
        return pos