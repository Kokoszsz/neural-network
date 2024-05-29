#include <cmath>// must be because ubuntu lol

#include "net.h"
#include "linear_layer.h"
#include "neuron.h"

Neuron::Neuron(){
    weight = randomWeight();
}
double Neuron::randomWeight(){
    return rand() / double(RAND_MAX);
}


