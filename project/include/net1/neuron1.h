#ifndef NEURON1_H
#define NEURON1_H

#include "neuron.h"

class Neuron1 : public Neuron {
public:
    Neuron1(unsigned numOutputs, unsigned myIndex); // Constructor
    // Override methods with different implementations
    void calcHiddenGradients(const Layer &nextLayer) override;
    void calcOutputGradients(double targetVal) override;
    double transferFunction(double x) override;
    double transferFunctionDerivative(double x) override;
};

#endif // NEURON1_H
