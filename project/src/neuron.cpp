

#include "net.h"
#include "layer.h"
#include "neuron.h"

Connection::Connection(){
    weight = randomWeight();
}
double Connection::randomWeight(){
    return rand() / double(RAND_MAX);
}

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


// Transfer function (sigmoid function)
double Neuron::transferFunction(double x){
    return 1 / (1 + exp(-x));
}

// Transfer function derivative
double Neuron::transferFunctionDerivative(double x){
    return x * (1 - x);
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