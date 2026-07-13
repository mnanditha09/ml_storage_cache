#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <atomic>
#include <memory>
#include <mutex>
#include "tensor.hpp"
#include "TaskQueue.hpp"

class Mat_Mul {
public:
    static Tensor mat_mul(const Tensor& input, const Tensor& weights, const Tensor& bias, TaskQueue& pool) {
        if(input.cols != weights.rows) throw std::runtime_error("Dimension mismatch for matrix multiplication.");
        Tensor result(input.rows, weights.cols);
        auto current_row = std::make_shared<std::atomic<size_t>>(0);
        auto rows_completed = std::make_shared<std::atomic<size_t>>(0);
        size_t total_rows = input.rows;
        size_t num_workers = std::thread::hardware_concurrency();
        for(size_t w=0; w < num_workers; ++w){
            pool.addTask([current_row, rows_completed, total_rows, &input, &weights, &bias, &result](){
                while(true) {
                    size_t i = current_row->fetch_add(1);
                    if(i >= total_rows) break;
                    for(size_t j=0; j<weights.cols; ++j) 
                        result(i,j) = bias(0,j);
                    for(size_t k=0; k<input.cols; ++k){
                        float tmp = input(i,k);
                        for(size_t j=0; j<weights.cols; ++j) 
                            result(i,j) += tmp * weights(k,j);
                    }
                    rows_completed->fetch_add(1);
                }
            });
        }
        while(rows_completed->load() < total_rows) {
            std::this_thread::yield();
        }
        return result;
    }
};