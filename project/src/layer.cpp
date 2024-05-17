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
