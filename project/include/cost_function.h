#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>



class CostFunction {
public:
    static double calculate_mse(const std::vector<std::vector<double>> &targetVals, const std::vector<std::vector<double>> &resultVals); 
};





#endif
