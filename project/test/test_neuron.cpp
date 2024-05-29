#include <gtest/gtest.h>
#include "neuron.h"


// Test fixture for Neuron class
class NeuronTest : public ::testing::Test {
protected:
    Neuron neuron = Neuron(2, 0); // Create an instance of Neuron for testing
};

