#include "linear_layer.h"


void LinearLayer::feedForwardLayer(const std::shared_ptr<Layer> &prevLayer) {

    for (unsigned n = 0; n < size() - 1; ++n) {
        double sum = 0.0;
        int neuronIndex = m_neurons[n].getMyIndex();
        for(unsigned m = 0; m < prevLayer->size(); ++m){

            sum += prevLayer->m_neurons[m].getOutputVal() * prevLayer->m_neurons[m].getoutputWeights()[neuronIndex].weight; 
        }
        m_neurons[n].outputVal = transferFunction(sum);
    }
}

double LinearLayer::transferFunction(double x){
    return 1 / (1 + exp(-x));
}

double LinearLayer::transferFunctionDerivative(double x){
    return x * (1 - x);
}

void LinearLayer::calcHiddenGradients(const std::shared_ptr<Layer> &nextLayer){
        for(unsigned n = 0; n < size(); ++n){
            double dow = m_neurons[n].sumDOW(nextLayer);
            m_neurons[n].m_gradient = dow * LinearLayer::transferFunctionDerivative(m_neurons[n].outputVal);
        }
}

void LinearLayer::calcOutputGradients(const std::vector<double> &targetVals){
    for(unsigned n = 0; n < size() - 1; ++n){
        double delta = targetVals[n] - m_neurons[n].outputVal;
        m_neurons[n].m_gradient = delta * LinearLayer::transferFunctionDerivative(m_neurons[n].outputVal);
    }
}

void LinearLayer::backPropagation(std::shared_ptr<Layer> &prevLayer){
    for(unsigned n = 0; n < size() - 1; ++n){
        int m_myIndex = m_neurons[n].getMyIndex();
        for(unsigned m = 0; m < prevLayer->size(); ++m){
            Neuron &neuron = prevLayer->m_neurons[m];
            double oldDeltaWeight =  neuron.getoutputWeights()[m_myIndex].deltaWeight;
            double newDeltaWeight = 
                // Individual input, magnified by the gradient and train rate
                eta
                * neuron.getOutputVal()
                *  m_neurons[n].m_gradient
                // Also add momentum = a fraction of the previous delta weight
                + alpha
                * oldDeltaWeight;
            neuron.updateWeight(m_myIndex, newDeltaWeight);
            neuron.updateDeltaWeight(m_myIndex, newDeltaWeight);
        }
    }
}

