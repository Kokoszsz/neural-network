#include "linear_layer.h"


void LinearLayer::feedForwardLayer(const std::shared_ptr<Layer> &prevLayer) {

    for (unsigned n = 0; n < size() - 1; ++n) {
        double sum = 0.0;
        for(unsigned m = 0; m < prevLayer->size(); ++m){

            sum += prevLayer->outputVals[m] * prevLayer->outputWeights[m][n].weight; 
        }
        prevLayer->outputVals[n] = sum;
    }
}


void LinearLayer::calcHiddenGradients(const std::shared_ptr<Layer> &nextLayer){
        for(unsigned n = 0; n < size(); ++n){
            double dow = this->sumDOW(nextLayer, n);
            m_gradients[n] = dow;
        }
}

void LinearLayer::calcOutputGradients(const std::vector<double> &targetVals){
    for(unsigned n = 0; n < size() - 1; ++n){
        double delta = targetVals[n] - outputVals[n];
        m_gradients[n] = delta;
    }
}

void LinearLayer::backPropagation(std::shared_ptr<Layer> &prevLayer){
    for(unsigned n = 0; n < size() - 1; ++n){
        for(unsigned m = 0; m < prevLayer->size(); ++m){
            Neuron &neuron = prevLayer->m_neurons[m];
            double oldDeltaWeight =  prevLayer->outputWeights[m][n].deltaWeight;
            double newDeltaWeight = 
                // Individual input, magnified by the gradient and train rate
                eta
                * prevLayer->outputVals[m]
                *  m_gradients[n]
                // Also add momentum = a fraction of the previous delta weight
                + alpha
                * oldDeltaWeight;
            prevLayer->outputWeights[m][n].weight = newDeltaWeight;
            prevLayer->outputWeights[m][n].deltaWeight = newDeltaWeight;
            
        }
    }
}

double LinearLayer::sumDOW(const std::shared_ptr<Layer> &nextLayer, int n) const{
    double sum = 0.0;
    for(unsigned m = 0; m < nextLayer->size() - 1; ++m){
        sum += outputWeights[n][m].weight * nextLayer->m_gradients[m];
    }
    return sum;
}

