// test_cost_function.cpp

#include "gtest/gtest.h"
#include "cost_function.h"
#include <vector>
#include <cmath>

// Test case for the calculate_mse function
TEST(CostFunctionTest, CalculateMSE) {
    CostFunction costFunction;

    // Test 1: Basic case with small vectors
    std::vector<std::vector<double>> targetVals1 = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> resultVals1 = {{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_DOUBLE_EQ(costFunction.calculate_mse(targetVals1, resultVals1), 0.0);

    // Test 2: Case with non-zero MSE
    std::vector<std::vector<double>> targetVals2 = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> resultVals2 = {{2.0, 1.0}, {4.0, 3.0}};
    EXPECT_DOUBLE_EQ(costFunction.calculate_mse(targetVals2, resultVals2), 2.0);

    // Test 3: Larger vectors
    std::vector<std::vector<double>> targetVals3 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<std::vector<double>> resultVals3 = {{1.0, 2.0, 2.0}, {4.0, 4.0, 6.0}};
    double expectedMSE3 = (0.0 + 0.0 + 1.0 + 0.0 + 1.0 + 0.0) / 2;
    EXPECT_DOUBLE_EQ(costFunction.calculate_mse(targetVals3, resultVals3), expectedMSE3);

    // Test 4: Mismatched sizes
    std::vector<std::vector<double>> targetVals4 = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> resultVals4 = {{1.0, 2.0}};
    EXPECT_DOUBLE_EQ(costFunction.calculate_mse(targetVals4, resultVals4), 0.0);
}
