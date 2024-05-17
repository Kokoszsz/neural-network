#ifndef NET_H
#define NET_H

#include <iostream>
#include <vector>
#include <cassert>
#include "linear_layer.h"


class Net{
    public:
        Net(const std::vector<unsigned> &topology);
        void feedForward(const std::vector<double> &inputVals);
        void backProp(const std::vector<double> &targetVals);
        void getResults(std::vector<double> &resultVals) const;

    private:
        std::vector<std::shared_ptr<Layer>> layers;
        double error;
        double recentAverageError;
        double recentAverageSmoothingFactor;
};






#endif