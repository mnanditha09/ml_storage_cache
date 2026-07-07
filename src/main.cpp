#include <iostream>
#include "ml_model.hpp"
#include "../include/tensor.hpp"
#include "../include/mat_mul.hpp"

int main() {
    Tensor input(1, 3, { 1.0f, 2.0f, 3.0f });
    Tensor weights(3, 2, { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f });
    Tensor bias(1, 2, { 0.5f, 1.0f });
    Tensor output = Mat_Mul::mat_mul(input, weights, bias);
    std::cout << "Output Vector:\n";
    output.print();
    return 0;
}