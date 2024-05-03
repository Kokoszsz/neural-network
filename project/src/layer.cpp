#include "net.h"
#include "layer.h"
#include "neuron1.h"

void Layer::push_back(const Neuron1& element) {
    m_neurons.push_back(element);
}
Neuron1& Layer::operator[](int i) {
    if (i < 0 || i >= size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_neurons[i];
}

const Neuron1& Layer::operator[](int i) const {
    if (i < 0 || i >= size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_neurons[i];
}

Neuron1 &Layer::back() {
    if (m_neurons.empty()) {
        throw std::out_of_range("Layer is empty");
    }
    return m_neurons.back();
}

const Neuron1 &Layer::back() const {
    if (m_neurons.empty()) {
        throw std::out_of_range("Layer is empty");
    }
    return m_neurons.back();
}