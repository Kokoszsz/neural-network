#include <gtest/gtest.h>
#include "linear_layer.h"
#include "neuron.h"

// Test fixture for Layer class
class LayerTest : public ::testing::Test {
protected:
    LinearLayer l_layer;
};

// Mock implementation of Layer for testing methods
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

// Test case to verify the behavior of push_back method
TEST_F(LayerTest, PushBack) {
    // Create a neuron
    Neuron neuron = Neuron(2, 0);

    // Add the neuron to the layer
    l_layer.push_back(neuron);

    // Verify that the size of the layer is increased by 1
    EXPECT_EQ(l_layer.size(), 1);
}

// Test case to verify the behavior of operator[] method (non-const version)
TEST_F(LayerTest, OperatorIndexNonConst) {
    // Create neurons and add them to the layer
    Neuron neuron1 = Neuron(2, 0);
    Neuron neuron2 = Neuron(2, 1);
    l_layer.push_back(neuron1);
    l_layer.push_back(neuron2);

    // Access the neurons using operator[]

    Neuron& retrievedNeuron1 = l_layer.m_neurons[0];
    Neuron& retrievedNeuron2 = l_layer.m_neurons[1];

    // Verify that the retrieved neurons are correct
    EXPECT_EQ(retrievedNeuron1.getMyIndex(), neuron1.getMyIndex());
    EXPECT_EQ(retrievedNeuron1.getOutputWeightsSize(), neuron1.getOutputWeightsSize());
    EXPECT_EQ(retrievedNeuron2.getMyIndex(), neuron2.getMyIndex());
    EXPECT_EQ(retrievedNeuron2.getOutputWeightsSize(), neuron2.getOutputWeightsSize());

}

// Test case to verify the behavior of operator[] method (const version)
TEST_F(LayerTest, OperatorIndexConst) {
    // Create neurons and add them to the layer
    Neuron neuron1 = Neuron(2, 0);
    Neuron neuron2 = Neuron(2, 1);
    l_layer.push_back(neuron1);
    l_layer.push_back(neuron2);

    // Access the neurons using operator[] (const version)
    const Layer& constLayer = l_layer;
    const Neuron& retrievedNeuron1 = constLayer.m_neurons[0];
    const Neuron& retrievedNeuron2 = constLayer.m_neurons[1];

    // Verify that the retrieved neurons are correct
    EXPECT_EQ(retrievedNeuron1.getMyIndex(), neuron1.getMyIndex());
    EXPECT_EQ(retrievedNeuron1.getOutputWeightsSize(), neuron1.getOutputWeightsSize());
    EXPECT_EQ(retrievedNeuron2.getMyIndex(), neuron2.getMyIndex());
    EXPECT_EQ(retrievedNeuron2.getOutputWeightsSize(), neuron2.getOutputWeightsSize());
}

// Test case to verify the behavior of back method (non-const version)
TEST_F(LayerTest, BackNonConst) {
    // Create neurons and add them to the layer
    Neuron neuron1 = Neuron(2, 0);
    Neuron neuron2 = Neuron(2, 1);
    l_layer.push_back(neuron1);
    l_layer.push_back(neuron2);

    // Get the last neuron using back method
    Neuron& backNeuron = l_layer.back();

    // Verify that the last neuron is correct
    EXPECT_EQ(backNeuron.getMyIndex(), neuron2.getMyIndex());
    EXPECT_EQ(backNeuron.getOutputWeightsSize(), neuron2.getOutputWeightsSize());
}

// Test case to verify the behavior of back method (const version)
TEST_F(LayerTest, BackConst) {
    // Create neurons and add them to the layer
    Neuron neuron1 = Neuron(2, 0);
    Neuron neuron2 = Neuron(2, 1);
    l_layer.push_back(neuron1);
    l_layer.push_back(neuron2);

    // Get the last neuron using back method (const version)
    const Layer& constLayer = l_layer;
    const Neuron& backNeuron = constLayer.back();

    // Verify that the last neuron is correct
    EXPECT_EQ(backNeuron.getMyIndex(), neuron2.getMyIndex());
    EXPECT_EQ(backNeuron.getOutputWeightsSize(), neuron2.getOutputWeightsSize());
}

