#include <gtest/gtest.h>
#include "layer.h"
#include "neuron.h"

// Test fixture for Layer class
class LayerTest : public ::testing::Test {
protected:
    Layer layer;
};

// Test case to verify the behavior of push_back method
TEST_F(LayerTest, PushBack) {
    // Create a neuron
    Neuron neuron = Neuron(2, 0);

    // Add the neuron to the layer
    layer.push_back(neuron);

    // Verify that the size of the layer is increased by 1
    EXPECT_EQ(layer.size(), 1);
}

// Test case to verify the behavior of operator[] method (non-const version)
TEST_F(LayerTest, OperatorIndexNonConst) {
    // Create neurons and add them to the layer
    Neuron neuron1 = Neuron(2, 0);
    Neuron neuron2 = Neuron(2, 1);
    layer.push_back(neuron1);
    layer.push_back(neuron2);

    // Access the neurons using operator[]

    Neuron& retrievedNeuron1 = layer[0];
    Neuron& retrievedNeuron2 = layer[1];

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
    layer.push_back(neuron1);
    layer.push_back(neuron2);

    // Access the neurons using operator[] (const version)
    const Layer& constLayer = layer;
    const Neuron& retrievedNeuron1 = constLayer[0];
    const Neuron& retrievedNeuron2 = constLayer[1];

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
    layer.push_back(neuron1);
    layer.push_back(neuron2);

    // Get the last neuron using back method
    Neuron& backNeuron = layer.back();

    // Verify that the last neuron is correct
    EXPECT_EQ(backNeuron.getMyIndex(), neuron2.getMyIndex());
    EXPECT_EQ(backNeuron.getOutputWeightsSize(), neuron2.getOutputWeightsSize());
}

// Test case to verify the behavior of back method (const version)
TEST_F(LayerTest, BackConst) {
    // Create neurons and add them to the layer
    Neuron neuron1 = Neuron(2, 0);
    Neuron neuron2 = Neuron(2, 1);
    layer.push_back(neuron1);
    layer.push_back(neuron2);

    // Get the last neuron using back method (const version)
    const Layer& constLayer = layer;
    const Neuron& backNeuron = constLayer.back();

    // Verify that the last neuron is correct
    EXPECT_EQ(backNeuron.getMyIndex(), neuron2.getMyIndex());
    EXPECT_EQ(backNeuron.getOutputWeightsSize(), neuron2.getOutputWeightsSize());
}


