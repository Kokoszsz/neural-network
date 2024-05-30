#include "net.h"
#include "layer.h"
#include "neuron.h"

double Layer::eta = 0.15;
double Layer::alpha = 0.5;


double Layer::sumDOW(const std::shared_ptr<Layer> &nextLayer, int n) const{
    double sum = 0.0;
    for(unsigned m = 0; m < nextLayer->size() - 1; ++m){
        sum += outputWeights[n][m].weight * nextLayer->m_gradients[m];
    }
    return sum;
}

