#include "net.h"
#include "layer.h"
#include "neuron.h"

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