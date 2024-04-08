#include <gtest/gtest.h>
#include "cost_function.h"

class CostFunctionTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        // Set up any variables or configurations needed for the tests
    }

    virtual void TearDown() {
        // Clean up after the tests
    }

    // Declare any additional helper functions if needed
};

// Test case for calculate_mse function
TEST_F(CostFunctionTest, CalculateMSE) {
    CostFunction cost_function;

    // Test case 1: Equal vectors
    std::vector<double> predicted1 = {1.0, 2.0, 3.0};
    std::vector<double> target1 = {1.0, 2.0, 3.0};
    EXPECT_EQ(cost_function.calculate_mse(predicted1, target1), 0.0);

    // Test case 2: Different vectors
    std::vector<double> predicted2 = {1.0, 2.0, 3.0};
    std::vector<double> target2 = {4.0, 5.0, 6.0};
    EXPECT_EQ(cost_function.calculate_mse(predicted2, target2), 9.0);

    // Test case 3: Unequal size vectors
    std::vector<double> predicted3 = {1.0, 2.0, 3.0};
    std::vector<double> target3 = {1.0, 2.0};
    // Expect an error message to be printed to cerr
    testing::internal::CaptureStderr();
    EXPECT_EQ(cost_function.calculate_mse(predicted3, target3), 0.0);
    std::string error_message = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(error_message.empty()); // Check if an error message was printed
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
