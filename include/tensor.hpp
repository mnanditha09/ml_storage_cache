#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>

class Tensor {
    public:
        size_t rows;
        size_t cols;
        std::vector<float> data;

        Tensor(size_t r, size_t c) : rows(r), cols(c), data(r*c,0.0f) {}
        Tensor(size_t r, size_t c, std::initializer_list<float> list) : rows(r), cols(c), data(list) {
            if (list.size() != r * c) throw std::runtime_error("Matrix Dimension Mismatch.");
        }
        inline float& operator()(size_t r, size_t c) {
            return data[r*cols + c];
        }
        inline const float& operator()(size_t r, size_t c) const {
            return data[r*cols + c]; 
        }
        void print() const {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) 
                    std::cout << data[i * cols + j] << " ";
                std::cout << "\n";
            }
        }
};