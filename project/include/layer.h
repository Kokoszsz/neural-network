#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>


class Neuron;


class Layer{
    public:
        std::vector<Neuron> m_neurons;
        virtual size_t size() const {return m_neurons.size();}
        virtual void push_back(const Neuron &neuron);
        virtual Neuron &operator[](int i);
        virtual const Neuron &operator[](int i) const;
        virtual Neuron &back(); 
        virtual const Neuron &back() const; 
    private:
};



#endif