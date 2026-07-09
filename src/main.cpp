#include <iostream>
#include "ml_model.hpp"
#include "../include/tensor.hpp"
#include "../include/mat_mul.hpp"
#include "../include/TaskQueue.hpp"

int main() {
    TaskQueue pool;
    std::cout << "Total number of threads available: " << std::thread::hardware_concurrency() << std::endl;
    Tensor input(4, 3, { 
        1.0f,  2.0f, -1.0f, 
        0.5f,  1.5f,  2.0f, 
       -2.0f,  0.0f,  1.0f, 
        3.0f, -1.0f,  0.0f 
    });
    Tensor weights(3, 2, { 
         0.5f, -0.2f, 
         0.8f,  0.1f, 
        -0.4f,  0.9f 
    });
    Tensor biases(1, 2, {0.1f, -0.1f});
    std::cout << "Result of Matrix Multiplication:\n";
    Tensor output = Mat_Mul::mat_mul(input,weights,biases,pool);
    output.print();
    return 0; 
}