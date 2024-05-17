#include "net.h"
#include "layer.h"
#include "neuron.h"

double Layer::eta = 0.15;
double Layer::alpha = 0.5;

void Layer::push_back(const Neuron& element) {
    m_neurons.push_back(element);
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

void Layer::feedForwardLayer(const std::shared_ptr<Layer> &prevLayer) {
    
}


double Layer::transferFunction(double x) {
    return 1;
}

double Layer::transferFunctionDerivative(double x) {
    return 1;
}

void Layer::calcOutputGradients(const std::vector<double> &targetVals) {
    
}

void Layer::calcHiddenGradients(const std::shared_ptr<Layer> &nextLayer) {
    
}

void Layer::backPropagation(std::shared_ptr<Layer> &prevLayer) {
    
}
