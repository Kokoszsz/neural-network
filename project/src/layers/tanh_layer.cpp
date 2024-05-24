#include "tanh_layer.h"
#include <cmath>

void TanhLayer::feedForwardLayer(const std::shared_ptr<Layer>& prevLayer) {
    for (unsigned n = 0; n < size() - 1; ++n) {
        double sum = 0.0;
        int neuronIndex = m_neurons[n].getMyIndex();
        for (unsigned m = 0; m < prevLayer->size(); ++m) {
            sum += prevLayer->m_neurons[m].getOutputVal() * prevLayer->m_neurons[m].getOutputWeights()[neuronIndex].weight;
        }
        m_neurons[n].outputVal = transferFunction(sum);
    }
}

double TanhLayer::transferFunction(double x) {
    return tanh(x);
}

double TanhLayer::transferFunctionDerivative(double x) {
    return 1 - x * x;
}

void TanhLayer::calcHiddenGradients(const std::shared_ptr<Layer>& nextLayer) {
    for (unsigned n = 0; n < size(); ++n) {
        double dow = m_neurons[n].sumDOW(nextLayer);
        m_neurons[n].m_gradient = dow * transferFunctionDerivative(m_neurons[n].outputVal);
    }
}

void TanhLayer::calcOutputGradients(const std::vector<double>& targetVals) {
    for (unsigned n = 0; n < size() - 1; ++n) {
        double delta = targetVals[n] - m_neurons[n].outputVal;
        m_neurons[n].m_gradient = delta * transferFunctionDerivative(m_neurons[n].outputVal);
    }
}

void TanhLayer::backPropagation(std::shared_ptr<Layer>& prevLayer) {
    for (unsigned n = 0; n < size() - 1; ++n) {
        int m_myIndex = m_neurons[n].getMyIndex();
        for (unsigned m = 0; m < prevLayer->size(); ++m) {
            Neuron &neuron = prevLayer->m_neurons[m];
            double oldDeltaWeight = neuron.getOutputWeights()[m_myIndex].deltaWeight;
            double newDeltaWeight = 
                eta * neuron.getOutputVal() * m_neurons[n].m_gradient + 
                alpha * oldDeltaWeight;
            neuron.updateWeight(m_myIndex, newDeltaWeight);
            neuron.updateDeltaWeight(m_myIndex, newDeltaWeight);
        }
    }
}
