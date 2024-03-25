import matplotlib.pyplot as plt

# Define the neural network architecture
input_layer = [0.1, 0.2, 0.7]  # Input layer
hidden_layer = [0.5, 0.4, 0.6]  # Hidden layer
output_layer = [0.3, 0.8]        # Output layer

# Plotting the neural network
plt.figure(figsize=(8, 6))

# Draw nodes for input layer
plt.scatter([1, 1, 1], [3, 2, 1], color='blue', label='Input Layer')
for i, txt in enumerate(input_layer):
    plt.annotate(txt, (1, 3-i), textcoords="offset points", xytext=(0,10), ha='center')

# Draw nodes for hidden layer
plt.scatter([2, 2, 2], [3, 2, 1], color='green', label='Hidden Layer')
for i, txt in enumerate(hidden_layer):
    plt.annotate(txt, (2, 3-i), textcoords="offset points", xytext=(0,10), ha='center')

# Draw nodes for output layer
plt.scatter([3, 3], [2, 1], color='red', label='Output Layer')
for i, txt in enumerate(output_layer):
    plt.annotate(txt, (3, 2-i), textcoords="offset points", xytext=(0,10), ha='center')

# Draw connections between layers
for i in range(len(input_layer)):
    for j in range(len(hidden_layer)):
        plt.plot([1, 2], [3-i, 3-j], color='gray')
for i in range(len(hidden_layer)):
    for j in range(len(output_layer)):
        plt.plot([2, 3], [3-i, 2-j], color='gray')

plt.gca().set_aspect('equal', adjustable='box')
plt.title('Simple Neural Network Architecture')
plt.legend(loc='upper left')
plt.axis('off')
plt.show()
