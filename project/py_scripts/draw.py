import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class NeuralNetwork:
    def __init__(self, topology):
        self.num_layers = len(topology)
        self.topology = topology
        self.biases = [np.random.randn(y, 1) for y in topology[1:]]  # Initialize biases for hidden and output layers
        self.weights = [np.random.randn(y, x) for x, y in zip(topology[:-1], topology[1:])]  # Initialize weights

    def feedforward(self, input_data):
        activation = input_data
        for bias, weight in zip(self.biases, self.weights):
            activation = self.sigmoid(np.dot(weight, activation) + bias)
        return activation

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def draw_neural_net(ax, num_layers, layer_sizes):
    G = nx.DiGraph()

    # Nodes
    for layer, size in enumerate(layer_sizes):
        for node in range(size):
            G.add_node((layer, node))

    # Edges
    for layer in range(num_layers - 1):
        for from_node in range(layer_sizes[layer]):
            for to_node in range(layer_sizes[layer + 1]):
                G.add_edge((layer, from_node), (layer + 1, to_node))

    pos = {}
    for layer in range(num_layers):
        for node in range(layer_sizes[layer]):
            pos[(layer, node)] = (layer, -node)

    nx.draw(G, pos, ax=ax, node_size=500, node_color='lightblue', with_labels=False, arrows=True)

def main():
    json_data = read_json('../data/network_data.json')
    topology = json_data['topology']

    network = NeuralNetwork(topology)
    print("Neural Network Topology:")
    for layer, neurons in enumerate(topology):
        print(f"Layer {layer + 1}: {neurons} neurons")

    fig, ax = plt.subplots(figsize=(10, 5))
    draw_neural_net(ax, network.num_layers, network.topology)
    ax.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
