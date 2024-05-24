#include "softmax_layer.h"
#include <cmath>
#include <vector>

void SoftmaxLayer::feedForwardLayer(const std::shared_ptr<Layer>& prevLayer) {
    std::vector<double> exponentials(size() - 1);
    double sum = 0.0;

    for (unsigned n = 0; n < size() - 1; ++n) {
        double exp_sum = 0.0;
        int neuronIndex = m_neurons[n].getMyIndex();
        for (unsigned m = 0; m < prevLayer->size(); ++m) {
            exp_sum += prevLayer->m_neurons[m].getOutputVal() * prevLayer->m_neurons[m].getOutputWeights()[neuronIndex].weight;
        }
        exponentials[n] = transferFunction(exp_sum);
        sum += exponentials[n];
    }

    for (unsigned n = 0; n < size() - 1; ++n) {
        m_neurons[n].outputVal = exponentials[n] / sum;
    }
}

double SoftmaxLayer::transferFunction(double x) {
    return exp(x);  // The actual function is applied in feedForwardLayer
}

void SoftmaxLayer::calcOutputGradients(const std::vector<double>& targetVals) {
    for (unsigned n = 0; n < size() - 1; ++n) {
        double delta = targetVals[n] - m_neurons[n].outputVal;
        m_neurons[n].m_gradient = delta;  // No derivative needed for softmax loss
    }
}

void SoftmaxLayer::calcHiddenGradients(const std::shared_ptr<Layer>& nextLayer) {
    for (unsigned n = 0; n < size(); ++n) {
        double dow = m_neurons[n].sumDOW(nextLayer);
        m_neurons[n].m_gradient = dow;  // No derivative needed for softmax loss
    }
}

void SoftmaxLayer::backPropagation(std::shared_ptr<Layer>& prevLayer) {
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
