#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <memory>
#include "connection.h"


class Connection;


class Layer{
    public:
        size_t size() const {return outputWeights.size();}


        virtual void feedForwardLayer(const std::shared_ptr<Layer>& prevLayer) = 0;
        virtual void calcOutputGradients(const std::vector<double> &targetVals) = 0;
        virtual void calcHiddenGradients(const std::shared_ptr<Layer> &nextLayer) = 0;
        virtual void backPropagation(std::shared_ptr<Layer> &prevLayer) = 0;

        virtual double sumDOW(const std::shared_ptr<Layer> &nextLayer, int n) const;

        std::vector<double> m_gradients;
        std::vector<double> outputVals;
        std::vector<std::vector<Connection>> outputWeights;

    protected:
        static double eta; // [0.0..1.0] overall net training rate
        static double alpha; // [0.0..n] multiplier of last weight change (momentum)

};



#endif