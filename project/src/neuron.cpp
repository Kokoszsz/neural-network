#include <cmath>// must be because ubuntu lol

#include "net.h"
#include "linear_layer.h"
#include "neuron.h"

Connection::Connection(){
    weight = randomWeight();
}
double Connection::randomWeight(){
    return rand() / double(RAND_MAX);
}


