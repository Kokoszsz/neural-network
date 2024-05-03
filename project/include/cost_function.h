#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>



class CostFunction {
public:
    static double calculate_mse(const std::vector<double> &predicted, const std::vector<double> &target); 
};





#endif
