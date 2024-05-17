#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H
#include "layer.h"


class LinearLayer: public Layer{
        public:
                void feedForwardLayer(const std::shared_ptr<Layer> &prevLayer) override;
                double transferFunction(double x) override;
                double transferFunctionDerivative(double x) override;
                void calcOutputGradients(const std::vector<double> &targetVals) override;
                void calcHiddenGradients(const std::shared_ptr<Layer> &nextLayer) override;
                void backPropagation(std::shared_ptr<Layer> &prevLayer) override;
};


#endif