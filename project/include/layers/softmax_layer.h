#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layer.h"

class SoftmaxLayer : public Layer {
public:
    void feedForwardLayer(const std::shared_ptr<Layer>& prevLayer) override;
    double transferFunction(double x) ;
    double transferFunctionDerivative(double x) ;
    void calcOutputGradients(const std::vector<double>& targetVals) override;
    void calcHiddenGradients(const std::shared_ptr<Layer>& nextLayer) override;
    void backPropagation(std::shared_ptr<Layer>& prevLayer) override;
};

#endif // SOFTMAX_LAYER_H
