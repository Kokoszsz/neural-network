#include "json.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>

#include "include/neuron.h"
#include "include/layer.h"



class Neuron;

class Layer{
    public:
        std::vector<Neuron> m_neurons;
        size_t size() const {return m_neurons.size();}
        void push_back(const Neuron &neuron);
        Neuron &operator[](int i);
        const Neuron &operator[](int i) const;
        Neuron &back(); 
        const Neuron &back() const; 
    private:
};

void Layer::push_back(const Neuron& element) {
    m_neurons.push_back(element);
}
Neuron& Layer::operator[](int i) {
    if (i < 0 || i >= size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_neurons[i];
}

const Neuron& Layer::operator[](int i) const {
    if (i < 0 || i >= size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_neurons[i];
}

Neuron &Layer::back() {
    if (m_neurons.empty()) {
        throw std::out_of_range("Layer is empty");
    }
    return m_neurons.back();
}

const Neuron &Layer::back() const {
    if (m_neurons.empty()) {
        throw std::out_of_range("Layer is empty");
    }
    return m_neurons.back();
}

class Connection{
    public:
        Connection();
        double weight;
        double deltaWeight;
    private:
        double randomWeight();
};

Connection::Connection(){
    weight = randomWeight();
}
double Connection::randomWeight(){
    return rand() / double(RAND_MAX);
}


class Neuron{
    public:
        Neuron(unsigned numOutputs, unsigned myIndex);
        void setOutputVal(double val){outputVal = val;}
        double getOutputVal() const {return outputVal;}
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);
    private:
        static double eta; // [0.0..1.0] overall net training rate
        static double alpha; // [0.0..n] multiplier of last weight change (momentum)
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        double sumDOW(const Layer &nextLayer) const;
        double outputVal;
        std::vector<Connection> outputWeights;
        unsigned m_myIndex;
        double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer &prevLayer){
    for(unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = 
            // Individual input, magnified by the gradient and train rate
            eta
            * neuron.getOutputVal()
            * m_gradient
            // Also add momentum = a fraction of the previous delta weight
            + alpha
            * oldDeltaWeight;
        neuron.outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;
    for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(outputVal);
}

void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(outputVal);
}


// Transfer function (set function to be what your outputs should be now it is tanh [-1.0, 1.0])
double Neuron::transferFunction(double x){
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;
    for(unsigned n = 0; n < prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights[m_myIndex].weight;
    }
    outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for(unsigned c = 0; c < numOutputs; ++c){
        outputWeights.push_back(Connection());
    }
    m_myIndex = myIndex;
}


class Net{
    public:
        Net(const std::vector<unsigned> &topology);
        void feedForward(const std::vector<double> &inputVals);
        void backProp(const std::vector<double> &targetVals);
        void getResults(std::vector<double> &resultVals) const;

    private:
        std::vector<Layer> layers;
        double error;
        double recentAverageError;
        double recentAverageSmoothingFactor;
};

void Net::getResults(std::vector<double> &resultVals) const{
    resultVals.clear();
    for(unsigned n = 0; n < layers.back().size() - 1; ++n){
        resultVals.push_back(layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const std::vector<double> &targetVals){
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = layers.back();
    error = 0.0;
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        error += delta * delta;
    }
    error /= outputLayer.size() - 1;
    error = sqrt(error);

    // Implement a recent average measurement
    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers
    for(unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum){
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];
        for(unsigned n = 0; n < hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer, update connection weights
    for(unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];
        for(unsigned n = 0; n < layer.size() - 1; ++n){
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const std::vector<double> &inputVals){
    assert(inputVals.size() == layers[0].size() - 1);
    for(unsigned i = 0; i < inputVals.size(); ++i){
        layers[0][i].setOutputVal(inputVals[i]);
    }
    // Forward propagate
    for(unsigned layerNum = 1; layerNum < layers.size(); ++layerNum){
        Layer &prevLayer = layers[layerNum - 1];
        for(unsigned n = 0; n < layers[layerNum].size() - 1; ++n){
            layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const std::vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            layers.back().push_back(Neuron(numOutputs, neuronNum));
        }
        layers.back().back().setOutputVal(1.0);
    }
}

int main(){

    std::vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(2);
    topology.push_back(1);
    Net net(topology);

    std::vector<std::vector<double>> inputVals;
    std::vector<std::vector<double>> targetVals;

    // Open the file
    std::ifstream file("XOR_Data.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> inputVals_one_example;
        std::vector<double> targetVals_one_example;
        
        // Split the line based on commas
        while (std::getline(ss, token, ',')) {
            // Convert each token to an integer
            int value = std::stod(token);
            
            // Add first two digits to the first vector
            if (inputVals_one_example.size() < 2) {
                inputVals_one_example.push_back(value);
            }
            // Add remaining digits to the second vector
            else {
                targetVals_one_example.push_back(value);
            }
        }
        inputVals.push_back(inputVals_one_example);
        targetVals.push_back(targetVals_one_example);
        
    }
    // Close the file
    file.close();

    //train neural network
    for (int i = 0 ; i < 10000; i++){
        for(unsigned i = 0; i < inputVals.size(); ++i){
            net.feedForward(inputVals[i]);
            net.backProp(targetVals[i]);
        }
    }

    // see how well the neural network performs
    std::vector<double> resultVals;
    for(unsigned i = 0; i < inputVals.size(); ++i){
        net.feedForward(inputVals[i]);
        net.backProp(targetVals[i]);
        net.getResults(resultVals);
        std::cout << "Input: " << inputVals[i][0] << ", " << inputVals[i][1] << " Target: " << targetVals[i][0] <<" Output: " << resultVals[0] << std::endl;
    }


    return 0;
}  