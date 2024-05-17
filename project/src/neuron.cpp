#include <cmath>// must be because ubuntu lol

#include "net.h"
#include "linear_layer.h"
#include "neuron.h"

Connection::Connection(){
    weight = randomWeight();
}
double Connection::randomWeight(){
    return rand() / double(RAND_MAX);
}

double Neuron::sumDOW(const std::shared_ptr<Layer> &nextLayer) const{
    double sum = 0.0;
    for(unsigned n = 0; n < nextLayer->size() - 1; ++n){
        sum += outputWeights[n].weight * nextLayer->m_neurons[n].m_gradient;
    }
    return sum;
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for(unsigned c = 0; c < numOutputs; ++c){
        outputWeights.push_back(Connection());
    }
    m_myIndex = myIndex;
}

void Neuron::updateDeltaWeight(int m_myIndex, double newDeltaWeight){
    outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
}

void Neuron::updateWeight(int m_myIndex, double newDeltaWeight){
    outputWeights[m_myIndex].weight += newDeltaWeight;
}

