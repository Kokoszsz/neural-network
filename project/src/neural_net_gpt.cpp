#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Activation function (Sigmoid)
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

class NeuralNetwork {
private:
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    double learning_rate;

    // Weights matrices
    vector<vector<double>> weights_input_hidden;
    vector<vector<double>> weights_hidden_output;

public:
    NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, double learning_rate) {
        this->input_nodes = input_nodes;
        this->hidden_nodes = hidden_nodes;
        this->output_nodes = output_nodes;
        this->learning_rate = learning_rate;

        // Initialize weights randomly between -1 and 1
        weights_input_hidden.resize(input_nodes, vector<double>(hidden_nodes));
        weights_hidden_output.resize(hidden_nodes, vector<double>(output_nodes));

        for (int i = 0; i < input_nodes; ++i) {
            for (int j = 0; j < hidden_nodes; ++j) {
                weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }

        for (int i = 0; i < hidden_nodes; ++i) {
            for (int j = 0; j < output_nodes; ++j) {
                weights_hidden_output[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
    }

    // Train the neural network using backpropagation
    void train(vector<vector<double>>& inputs, vector<vector<double>>& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < inputs.size(); ++i) {
                // Forward pass
                vector<double> hidden_output(hidden_nodes);
                vector<double> final_output(output_nodes);

                // Calculate output of hidden layer
                for (int j = 0; j < hidden_nodes; ++j) {
                    double sum = 0;
                    for (int k = 0; k < input_nodes; ++k) {
                        sum += inputs[i][k] * weights_input_hidden[k][j];
                    }
                    hidden_output[j] = sigmoid(sum);
                }

                // Calculate final output
                for (int j = 0; j < output_nodes; ++j) {
                    double sum = 0;
                    for (int k = 0; k < hidden_nodes; ++k) {
                        sum += hidden_output[k] * weights_hidden_output[k][j];
                    }
                    final_output[j] = sigmoid(sum);
                }

                // Backward pass
                // Calculate output layer errors
                vector<double> output_errors(output_nodes);
                for (int j = 0; j < output_nodes; ++j) {
                    output_errors[j] = targets[i][j] - final_output[j];
                }

                // Calculate hidden layer errors
                vector<double> hidden_errors(hidden_nodes);
                for (int j = 0; j < hidden_nodes; ++j) {
                    double error = 0;
                    for (int k = 0; k < output_nodes; ++k) {
                        error += output_errors[k] * weights_hidden_output[j][k];
                    }
                    hidden_errors[j] = error;
                }

                // Update weights
                // Update hidden to output weights
                for (int j = 0; j < hidden_nodes; ++j) {
                    for (int k = 0; k < output_nodes; ++k) {
                        weights_hidden_output[j][k] += learning_rate * output_errors[k] * hidden_output[j] * sigmoid_derivative(final_output[k]);
                    }
                }

                // Update input to hidden weights
                for (int j = 0; j < input_nodes; ++j) {
                    for (int k = 0; k < hidden_nodes; ++k) {
                        weights_input_hidden[j][k] += learning_rate * hidden_errors[k] * inputs[i][j] * sigmoid_derivative(hidden_output[k]);
                    }
                }
            }
        }
    }

    // Predict the output for given input
    vector<double> predict(vector<double>& input) {
        vector<double> hidden_output(hidden_nodes);
        vector<double> final_output(output_nodes);

        // Calculate output of hidden layer
        for (int j = 0; j < hidden_nodes; ++j) {
            double sum = 0;
            for (int k = 0; k < input_nodes; ++k) {
                sum += input[k] * weights_input_hidden[k][j];
            }
            hidden_output[j] = sigmoid(sum);
        }

        // Calculate final output
        for (int j = 0; j < output_nodes; ++j) {
            double sum = 0;
            for (int k = 0; k < hidden_nodes; ++k) {
                sum += hidden_output[k] * weights_hidden_output[k][j];
            }
            final_output[j] = sigmoid(sum);
        }

        return final_output;
    }
};

int main() {
    // XOR inputs
    vector<vector<double>> inputs = {{1, 1}, {1, 0}, {0, 1}, {0, 0}};
    // XOR targets
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};

    int input_nodes = 2;
    int hidden_nodes = 4;
    int output_nodes = 1;
    double learning_rate = 0.1;
    int epochs = 100000;

    // Create and train the neural network
    NeuralNetwork nn(input_nodes, hidden_nodes, output_nodes, learning_rate);
    nn.train(inputs, targets, epochs);

    // Predict the output for XOR inputs
    cout << "Predictions:" << endl;
    for (int i = 0; i < inputs.size(); ++i) {
        vector<double> prediction = nn.predict(inputs[i]);
        cout << "Input: " << inputs[i][0] << ", " << inputs[i][1] << " --> Output: " << prediction[0] << endl;
    }

    return 0;
}
