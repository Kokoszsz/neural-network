#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include <fstream>
#include <sstream>

//#include <nlohmann/json.hpp>



class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned m_myindex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal() const { return m_outputVal; }

    void calcOutputGradients(double targetVal){
        double delta = targetVal - m_outputVal;
        m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
    }

    void calcHiddenGradients(const Layer &nextLayer){
        double dow = sumDOW(nextLayer);
        m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
    }

    void feedForwardNeuron(const Layer &prevLayer){
        double sum = 0.0;

        for(unsigned n = 0; n < prevLayer.size(); ++n){
            sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myindex].weight;
        }

        m_outputVal = transferFunction(sum);
    }

    void updateInputWeights(Layer &prevLayer){
        for(unsigned n = 0; n < prevLayer.size(); ++n){
            Neuron &neuron = prevLayer[n];
            double oldDeltaWeight = neuron.m_outputWeights[m_myindex].deltaWeight;
            double newDeltaWeight = 
                // Individual input, magnified by the gradient and train rate
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight
                + alpha
                * oldDeltaWeight;
            neuron.m_outputWeights[m_myindex].deltaWeight = newDeltaWeight;
            neuron.m_outputWeights[m_myindex].weight += newDeltaWeight;
        }
    }

private:
    static double eta; // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double transferFunction(double x) { return tanh(x); }
    static double transferFunctionDerivative(double x) { return 1.0 - x * x; }
    static double randomWeight() { return rand() / double(RAND_MAX); }
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    unsigned m_myindex;
    double m_gradient;
    double sumDOW(const Layer &nextLayer) const{
        double sum = 0.0;
        for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
            sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
        }
        return sum;
    }
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;


Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for(unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myindex = myIndex;
}


class NeuralNet{
public:
    NeuralNet(const std::vector<unsigned> &topology);
    void getResults(std::vector<double> &resultVals) const
    {
        resultVals.clear();

        for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
            resultVals.push_back(m_layers.back()[n].getOutputVal());
        }
    }
    void backProp(const std::vector<double> &targetVals){
    
        Layer &outputLayer = m_layers.back();
        m_error = 0.0;
        for (unsigned n = 0; n < outputLayer.size() - 1; ++n){
            double delta = targetVals[n] - outputLayer[n].getOutputVal();
            m_error += delta * delta;
        }
        m_error /= outputLayer.size() - 1;
        m_error = sqrt(m_error);


        for (unsigned n = 0; n < outputLayer.size() - 1; ++n){
            outputLayer[n].calcOutputGradients(targetVals[n]);
        }

        for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum){
            Layer &hiddenLayer = m_layers[layerNum];
            Layer &nextLayer = m_layers[layerNum + 1];

            for (unsigned n = 0; n < hiddenLayer.size(); ++n){
                hiddenLayer[n].calcHiddenGradients(nextLayer);
            }
        }

        for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
            Layer &layer = m_layers[layerNum];
            Layer &prevLayer = m_layers[layerNum - 1];

            for (unsigned n = 0; n < layer.size() - 1; ++n){
                layer[n].updateInputWeights(prevLayer);
            }
        }


    }

    void feedForward(const std::vector<double> &inputVals){
        //assert(inputVals.size() == m_layers[0].size() - 1);
        for(unsigned i = 0; i < inputVals.size(); ++i){
            m_layers[0][i].setOutputVal(inputVals[i]);
        }

        for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
            Layer &prevLayer = m_layers[layerNum - 1];
            for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
                m_layers[layerNum][n].feedForwardNeuron(prevLayer);
            }
        }
    }

private:
    std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
};


NeuralNet::NeuralNet(const std::vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        std::cout << "Made a Layer!" << std::endl;
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Made a Neuron!" << std::endl;
        }
    }
}


void showVectorVals(std::string label, std::vector<double> &v)
{
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
}

int main(){
    std::vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    NeuralNet myNet(topology);

    std::vector<std::vector<double>> inputVals;
    std::vector<std::vector<double>> targetVals;
    std::vector<std::vector<double>> resultVals;

    std::ifstream infile("data.txt");
    std::string line;

    // get data from file
    while (std::getline(infile, line)) {
        std::vector<double> temp;
        // Find the position of the "," symbol
        size_t comma_position = line.find(',');
        
            // Extract substrings before and after the ","
        std::string input = line.substr(0, comma_position);
        size_t dot_position = input.find('.');
        std::string part1 = input.substr(0, dot_position);
        std::string part2 = input.substr(dot_position + 1);

        temp.push_back(std::stod(part1));
        temp.push_back(std::stod(part2));
        inputVals.push_back(temp);

        temp.clear();
    
        std::string target = line.substr(comma_position + 1);
        dot_position = target.find('.');
        part1 = target.substr(0, dot_position);
        part2 = target.substr(dot_position + 1);

        temp.push_back(std::stod(part1));
        temp.push_back(std::stod(part2));
        targetVals.push_back(temp);



    }
    
    // Train network
    for (int i = 0; i < 1000; ++i) { 
        for (int j = 0; j < inputVals.size(); ++j) {
            myNet.feedForward(inputVals[j]);
            myNet.backProp(targetVals[j]);
        }
    }

    // Test the trained network
    for (int i = 0; i < inputVals.size(); ++i) {
        myNet.feedForward(inputVals[i]);
        std::vector<double> resultVals;
        myNet.getResults(resultVals);
        showVectorVals("Input: ", inputVals[i]);
        showVectorVals("Output: ", resultVals);
        showVectorVals("Target: ", targetVals[i]);
        std::cout << std::endl;
    }


    std::cout << "Done!" << std::endl;


    return 0;
}
