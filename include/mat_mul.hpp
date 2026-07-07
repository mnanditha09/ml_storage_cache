#pragma once
#include <iostream>
#include <cmath>
#include "tensor.hpp"

class Mat_Mul {
    public:
        static Tensor mat_mul(const Tensor& input, const Tensor& weights, const Tensor& bias) {
            if(input.cols != weights.rows){
                throw std::runtime_error("Dimension mismatch for matrix multiplication.");
            }
            Tensor result(input.rows, weights.cols);
            for(size_t i=0; i<input.rows; ++i){
                for(size_t k=0; k<input.cols; ++k){
                    float val = input(i,k);
                    for(size_t j=0; j<weights.cols; ++j){
                        result(i,j) += val * weights(k,j);
                    }
                }
                for(size_t j=0; j<weights.cols; ++j){
                    result(i,j) += bias(0,j);
                }
            }
            return result;
        }
};