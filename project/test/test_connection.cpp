#include <gtest/gtest.h>
#include <cmath> // Include cmath if needed for mathematical functions
#include "connection.h" // Assuming this is where Connection class is defined

// Assuming Connection class has a public constructor that initializes weight
TEST(ConnectionTest, ConstructorTest) {
    // Initialize a Connection object
    Connection conn;

    // Get the weight (assuming a getter method)
    double weight = conn.weight; // Assuming getWeight() retrieves the weight

    // Check that the weight is within the valid range [0.0, 1.0]
    EXPECT_GE(weight, 0.0); // Expect weight to be greater than or equal to 0.0
    EXPECT_LE(weight, 1.0); // Expect weight to be less than or equal to 1.0
}

