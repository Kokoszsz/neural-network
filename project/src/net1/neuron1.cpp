#include "neuron1.h"

Neuron1::Neuron1(unsigned numOutputs, unsigned myIndex) : Neuron(numOutputs, myIndex) {}

void Neuron1::calcHiddenGradients(const Layer &nextLayer){
    // Implement different calculation for hidden layer gradients
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(outputVal) * 2.0; // Modify gradient calculation
}

void Neuron1::calcOutputGradients(double targetVal){
    // Implement different calculation for output layer gradients
    double delta = targetVal - outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(outputVal) * 0.5; // Modify gradient calculation
}

double Neuron1::transferFunction(double x){
    // Implement a different transfer function
    return sin(x); // Example: Use sine function
}

double Neuron1::transferFunctionDerivative(double x){
    // Implement the derivative of the transfer function
    return cos(x); // Example: Use cosine function as derivative of sine
}
