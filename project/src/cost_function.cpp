
#include "cost_function.h"


// Mean Squared Error (MSE) cost function
double CostFunction::calculate_mse(const std::vector<std::vector<double>> &targetVals, const std::vector<std::vector<double>> &resultVals) {
    if (targetVals.size() != resultVals.size()) {
        std::cerr << "Error: predicted and target vectors must be of the same size." << std::endl;
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < targetVals.size(); ++i) {
            if (targetVals[i].size() != resultVals[i].size()) {
                std::cerr << "Error: each predicted and target pair must be of the same size." << std::endl;
                return 0.0;
            }

            double exampleSum = 0.0;
            for (size_t j = 0; j < targetVals[i].size(); ++j) {
                exampleSum += pow(targetVals[i][j] - resultVals[i][j], 2);
            }
            sum += exampleSum;
        }
    return sum / targetVals.size();
}
