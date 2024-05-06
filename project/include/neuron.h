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
        double getGradient() const {return m_gradient;} // Added for testing
        int getMyIndex() const {return m_myIndex;} // Added for testing
        int getOutputWeightsSize() const {return outputWeights.size();} // Added for testing
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);
    private:
        static double eta; // [0.0..1.0] overall net training rate
        static double alpha; // [0.0..n] multiplier of last weight change (momentum)
        double transferFunction(double x);
        double transferFunctionDerivative(double x);
        double sumDOW(const Layer &nextLayer) const;
        double outputVal;
        std::vector<Connection> outputWeights;
        unsigned m_myIndex;
        double m_gradient;
};


#endif