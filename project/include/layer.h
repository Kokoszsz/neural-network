#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <memory>
#include "neuron.h"


class Neuron;


class Layer{
    public:
        std::vector<Neuron> m_neurons;
        size_t size() const {return m_neurons.size();}
        void push_back(const Neuron &neuron);
        Neuron &back(); 
        const Neuron &back() const; 


        virtual void feedForwardLayer(const std::shared_ptr<Layer>& prevLayer);
        virtual double transferFunction(double x);
        virtual double transferFunctionDerivative(double x);
        virtual void calcOutputGradients(const std::vector<double> &targetVals);
        virtual void calcHiddenGradients(const std::shared_ptr<Layer> &nextLayer);
        virtual void backPropagation(std::shared_ptr<Layer> &prevLayer);



    protected:
        static double eta; // [0.0..1.0] overall net training rate
        static double alpha; // [0.0..n] multiplier of last weight change (momentum)
};



#endif