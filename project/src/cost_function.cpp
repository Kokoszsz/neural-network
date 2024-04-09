
#include "cost_function.h"
//sdklgndsi;ugfbdsiugfbds


// Mean Squared Error (MSE) cost function
double CostFunction::calculate_mse(const std::vector<double> &predicted, const std::vector<double> &target) {
    if (predicted.size() != target.size()) {
        std::cerr << "Error: predicted and target vectors must be of the same size." << std::endl;
        return 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < predicted.size(); ++i) {
        sum += pow(predicted[i] - target[i], 2);
    }
    return sum / predicted.size();
}
