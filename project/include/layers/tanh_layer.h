#ifndef TANH_LAYER_H
#define TANH_LAYER_H

#include "layer.h"

class TanhLayer : public Layer {
public:
    void feedForwardLayer(const std::shared_ptr<Layer>& prevLayer) override;
    double transferFunction(double x);
    double transferFunctionDerivative(double x);
    void calcHiddenGradients(const std::shared_ptr<Layer>& nextLayer) override;
    void calcOutputGradients(const std::vector<double>& targetVals) override;
    void backPropagation(std::shared_ptr<Layer>& prevLayer) override;
};

#endif // TANH_LAYER_H
