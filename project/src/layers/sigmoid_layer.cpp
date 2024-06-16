#include "sigmoid_layer.h"

void SigmoidLayer::feedForwardLayer(const std::shared_ptr<Layer>& prevLayer) {
    for (unsigned n = 0; n < size() - 1; ++n) {
        double sum = 0.0;
        for(unsigned m = 0; m < prevLayer->size(); ++m){
            sum += prevLayer->outputVals[m] * prevLayer->outputWeights[m][n].weight; 
        }
        outputVals[n] = activationFunction(sum);
    }
}

void SigmoidLayer::calcHiddenGradients(const std::shared_ptr<Layer>& nextLayer) {
    for(unsigned n = 0; n < size(); ++n){
        double dow = this->sumDOW(nextLayer, n);
        m_gradients[n] = dow * activationFunctionDerivative(outputVals[n]);
    }
}

void SigmoidLayer::calcOutputGradients(const std::vector<double>& targetVals) {
    for(unsigned n = 0; n < size() - 1; ++n){
        double delta = targetVals[n] - outputVals[n];
        m_gradients[n] = delta * activationFunctionDerivative(outputVals[n]);
    }
}

void SigmoidLayer::backPropagation(std::shared_ptr<Layer>& prevLayer) {
    for(unsigned n = 0; n < size() - 1; ++n){
        for(unsigned m = 0; m < prevLayer->size(); ++m){
            double oldDeltaWeight =  prevLayer->outputWeights[m][n].deltaWeight;
            double newDeltaWeight = 
                eta * prevLayer->outputVals[m] * m_gradients[n] + alpha * oldDeltaWeight;
            prevLayer->outputWeights[m][n].weight += newDeltaWeight;
            prevLayer->outputWeights[m][n].deltaWeight = newDeltaWeight;
        }
    }
}

double SigmoidLayer::activationFunction(double x)  {
    return 1.0 / (1.0 + exp(-x)); 
}

double SigmoidLayer::activationFunctionDerivative(double x)  {
    return x * (1.0 - x); 
}

