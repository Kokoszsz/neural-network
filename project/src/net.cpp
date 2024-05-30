#include <cmath>// must be because ubuntu lol


#include "net.h"
#include "linear_layer.h"
#include "sigmoid_layer.h"
#include "neuron.h"

void Net::getResults(std::vector<double> &resultVals) const{
    resultVals.clear();
    for(unsigned n = 0; n < layers.back()->size() - 1; ++n){
        resultVals.push_back(layers.back()->outputVals[n]);
    }
}

void Net::backProp(const std::vector<double> &targetVals){
    // Calculate overall net error (RMS of output neuron errors)
    std::shared_ptr<Layer> &outputLayer = layers.back();
    error = 0.0;
    for(unsigned n = 0; n < outputLayer->size() - 1; ++n){
        double delta = targetVals[n] - outputLayer->outputVals[n];
        error += delta * delta;
    }
    error /= outputLayer->size() - 1;
    error = sqrt(error);

    // Implement a recent average measurement
    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    outputLayer->calcOutputGradients(targetVals);
    
    // Calculate gradients on hidden layers
    for(unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum){
        std::shared_ptr<Layer> &nextLayer = layers[layerNum + 1];
        layers[layerNum]->calcHiddenGradients(nextLayer);
    }

    // For all layers from outputs to first hidden layer, update connection weights
    for(unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum){
        std::shared_ptr<Layer> &prevLayer = layers[layerNum - 1];
        layers[layerNum]->backPropagation(prevLayer);
    }
}

void Net::feedForward(const std::vector<double> &inputVals){
    assert(inputVals.size() == layers[0]->size() - 1);
    for(unsigned i = 0; i < inputVals.size(); ++i){
        layers[0]->outputVals[i] = inputVals[i];
    }
    // Forward propagate
    for(unsigned layerNum = 1; layerNum < layers.size(); ++layerNum){
        std::shared_ptr<Layer> &prevLayer = layers[layerNum - 1];
        layers[layerNum]->feedForwardLayer(prevLayer); 

    }
}

Net::Net(const std::vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        std::shared_ptr<Layer> layer = std::make_shared<SigmoidLayer>();
        layers.push_back(layer);
        std::shared_ptr<Layer> &prevLayer = layers[layerNum];
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            layers.back()->outputWeights.push_back(std::vector<Neuron>());
            layers.back()->outputVals.push_back(0.0);
            layers.back()->m_gradients.push_back(0.0);
            for(unsigned c = 0; c < numOutputs; ++c){
                layers.back()->outputWeights[neuronNum].push_back(Neuron());
            }
        }
        layers.back()->outputVals.back() = 1.0;
    }

}