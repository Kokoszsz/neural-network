#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>



class Neuron{


    public:
        Neuron();
        double weight;
        double deltaWeight;
    private:
        double randomWeight();
};


#endif