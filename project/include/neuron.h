#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>
#include "layer.h"

class Layer;

class Connection{


    public:
        Connection();
        double weight;
        double deltaWeight;
    private:
        double randomWeight();
};

class Neuron{
    public:
        Neuron(unsigned numOutputs, unsigned myIndex);
        void setOutputVal(double val){outputVal = val;}
        double getOutputVal() const {return outputVal;}
        std::vector<Connection> getOutputWeights() const {return outputWeights;}
        double getGradient() const {return m_gradient;} // Added for testing
        int getMyIndex() const {return m_myIndex;}
        int getOutputWeightsSize() const {return outputWeights.size();} // Added for testing
        void updateWeight(int m_myIndex, double newDeltaWeight);
        void updateDeltaWeight(int m_myIndex, double newDeltaWeight);
 
        double sumDOW(const std::shared_ptr<Layer> &nextLayer) const;
        double outputVal;
        double m_gradient;
    private:
        std::vector<Connection> outputWeights;
        unsigned m_myIndex;

};


#endif