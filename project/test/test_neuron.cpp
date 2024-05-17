#include <gtest/gtest.h>
#include "neuron.h"

// Mock implementation of Layer for testing Neuron methods
class MockLayer : public Layer {
public:
    // Mock implementation of methods required for testing
    // You can add more if needed
    MockLayer() {}

    // Implement size method
    size_t size() const {
        return 2; // Return a fixed size for testing
    }

    // Implement getOutputVal method
    double getOutputVal(size_t index) const {
        return index == 0 ? 0.2 : 0.7; // Return fixed output values for testing
    }
};

// Test fixture for Neuron class
class NeuronTest : public ::testing::Test {
protected:
    Neuron neuron = Neuron(2, 0); // Create an instance of Neuron for testing
};

// Test case to verify the behavior of the setOutputVal and getOutputVal methods
TEST_F(NeuronTest, SetAndGetOutputVal) {
    neuron.setOutputVal(0.5);
    EXPECT_EQ(neuron.getOutputVal(), 0.5);
}
