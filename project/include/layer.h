#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include "neuron.h"


class Neuron;


class Layer{
    public:
        std::vector<Neuron> m_neurons;
        size_t size() const {return m_neurons.size();}
        void push_back(const Neuron &neuron);
        Neuron &operator[](int i);
        const Neuron &operator[](int i) const;
        Neuron &back(); 
        const Neuron &back() const; 
    private:
};



#endif