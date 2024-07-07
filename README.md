## Neural Network Project

Welcome to the Neural Network Project! This project implements a simple neural network framework in C++. It includes various classes representing different layers of the network, each with its own activation function and methods for forward propagation and backpropagation.

For detailed information and documentation, please visit the [Documentation](./docu).

### Features
- **Modular Layer Design:** Easily add different types of layers such as Linear, Sigmoid, and Tanh layers.
- **Flexible Architecture:** Design and train neural networks with customizable architectures.
- **Forward and Backpropagation:** Efficient algorithms for forward propagation and backpropagation to train your neural network.

### Usage (Windows)
To build and run the project, follow these steps:

1. **Download Repo**
    ```sh
    git clone <repo-url>
    cd Project
    ```

2. **Create Build Directory:**
   ```sh
   mkdir build
   cd build
   ```
3. **Configure Project:**
   ```sh
   cmake ..
   ```

4. **Build Project:**
   ```sh
   cmake --build .
   ```

5. **Run Executable:**
   ```sh
   .\Debug\neural_net_main.exe
   ```



### Dependencies
This project requires:

    A C++ compiler that supports C++17 or later.
    CMake for build configuration.
    Google Test for running unit tests (included as a submodule).


### Project Structure
```bash   
project/
│
├── include/            # Header files for various classes and layers
├── src/                # Source files for the implementation of the neural network
├── test/               # Unit tests for the project
├── data/               # JSON representation of the neural network, used in Python visualization scripts
├── libraries/          # Libraries (in the form of header files) used in the project
├── py_script/          # Python scripts used for simple visualization of neural network architecture
├── training_data/      # Data used to train a neural network
└── cmake/              # CMake file for updating modules

sources_used/           # Additional materials/tutorials used by the author to help create this project

docu/                   # Documentation files

```
    
    

### License

This project is licensed under the MIT License. Feel free to use, modify, and distribute this code as you see fit.
