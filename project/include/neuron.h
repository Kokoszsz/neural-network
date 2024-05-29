#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>



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

};


#endif