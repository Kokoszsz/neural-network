#include "json.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib> // For system() function
#include <chrono> // For timing

#include "connection.h"
#include "linear_layer.h"
#include "sigmoid_layer.h"
#include "tanh_layer.h"
#include "net.h"
#include "cost_function.h"

using json = nlohmann::json;

int main(){

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<unsigned> topology = {2, 3, 3, 1};

    Net net(topology);

    std::vector<std::vector<double>> inputVals;
    std::vector<std::vector<double>> targetVals;

    // Open the file
    std::ifstream file("training_data/XOR_Data.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> inputVals_one_example;
        std::vector<double> targetVals_one_example;
        
        // Split the line based on commas
        while (std::getline(ss, token, ',')) {
            // Convert each token to an integer
            double value = std::stod(token);
            
            // Add first two digits to the first vector
            if (inputVals_one_example.size() < 2) {
                inputVals_one_example.push_back(value);
            }
            // Add remaining digits to the second vector
            else {
                targetVals_one_example.push_back(value);
            }
        }
        inputVals.push_back(inputVals_one_example);
        targetVals.push_back(targetVals_one_example);
        
    }
    // Close the file
    file.close();

    //train neural network
    for (int i = 0 ; i < 10000; i++){
        for(unsigned j = 0; j < inputVals.size(); ++j){
            net.feedForward(inputVals[j]);
            net.backProp(targetVals[j]);
        }
    }

    // see how well the neural network performs
    std::vector<std::vector<double>> resultVals(inputVals.size());
    for(unsigned i = 0; i < inputVals.size(); ++i){
        net.feedForward(inputVals[i]);
        net.backProp(targetVals[i]);
        net.getResults(resultVals[i]);
        std::cout << "Input: " << inputVals[i][0] << ", " << inputVals[i][1] << " Target: " << targetVals[i][0] <<" Output: " << resultVals[i][0] << std::endl;
    }


    // Calculate the Mean Squared Error (MSE)
    CostFunction costFunction;
    double mse = costFunction.calculate_mse(targetVals, resultVals);
    std::cout << "Mean Squared Error: " << mse << std::endl;

    // Create a JSON object to hold the network topology and training data
    json data;
    data["topology"] = topology;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Output the duration in milliseconds
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    // Write the JSON data to a file
    std::ofstream jsonFile("data/network_data.json");
    if (jsonFile.is_open()) {
        jsonFile << std::setw(4) << data << std::endl; // Pretty-print with indentation of 4 spaces
        jsonFile.close();
        std::cout << "JSON data has been written to network_data.json" << std::endl;
    } else {
        std::cerr << "Error opening JSON file for writing!" << std::endl;
        return 1;
    }

    // Run the Python script to visualize the neural network
    int result = std::system("python py_scripts/draw.py");
    if (result != 0) {
        std::cerr << "Error running Python script!" << std::endl;
        return 1;
    }

    return 0;
}  