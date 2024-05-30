#include "tanh_layer.h"

void TanhLayer::feedForwardLayer(const std::shared_ptr<Layer>& prevLayer) {
    for (unsigned n = 0; n < size() - 1; ++n) {
        double sum = 0.0;
        for(unsigned m = 0; m < prevLayer->size(); ++m){
            sum += prevLayer->outputVals[m] * prevLayer->outputWeights[m][n].weight; 
        }
        outputVals[n] = activationFunction(sum);
    }
}

void TanhLayer::calcHiddenGradients(const std::shared_ptr<Layer>& nextLayer) {
    for(unsigned n = 0; n < size(); ++n){
        double dow = this->sumDOW(nextLayer, n);
        m_gradients[n] = dow * activationFunctionDerivative(outputVals[n]);
    }
}

void TanhLayer::calcOutputGradients(const std::vector<double>& targetVals) {
    for(unsigned n = 0; n < size() - 1; ++n){
        double delta = targetVals[n] - outputVals[n];
        m_gradients[n] = delta * activationFunctionDerivative(outputVals[n]);
    }
}

void TanhLayer::backPropagation(std::shared_ptr<Layer>& prevLayer) {
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

double TanhLayer::activationFunction(double x) {
    return tanh(x);
}

double TanhLayer::activationFunctionDerivative(double x) {
    return 1.0 - x * x; // tanh derivative
}

