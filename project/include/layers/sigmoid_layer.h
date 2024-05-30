#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H
#include "layer.h"


class SigmoidLayer: public Layer{
        public:
                void feedForwardLayer(const std::shared_ptr<Layer> &prevLayer) override;
                void calcOutputGradients(const std::vector<double> &targetVals) override;
                void calcHiddenGradients(const std::shared_ptr<Layer> &nextLayer) override;
                void backPropagation(std::shared_ptr<Layer> &prevLayer) override;


                double activationFunction(double x);

                double activationFunctionDerivative(double x);

};


#endif