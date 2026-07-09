#pragma once
#include <iostream>
#include <cmath>
#include <atomic>
#include "tensor.hpp"
#include "TaskQueue.hpp"

class Mat_Mul {
    private:
        struct RowTask {
            int row;
            const Tensor& input;
            const Tensor& weights;
            const Tensor& bias;
            Tensor& result;
            std::atomic<int>& counter;
            RowTask(int r, const Tensor& in, const Tensor& w, const Tensor& b, Tensor& res, std::atomic<int>& c)
            : row(r), input(in), weights(w), bias(b), result(res), counter(c) {}
            void operator()(){
                for(size_t j=0; j<weights.cols; ++j){
                    result(row,j) = bias(0,j);
                }
                for(size_t k=0; k<input.cols; ++k){
                    float val = input(row,k);
                    for(size_t j=0; j<weights.cols; ++j){
                        result(row,j) += val*weights(k,j);
                    }
                }
                counter++;
            }
        };
    public:
        static Tensor mat_mul(const Tensor& input, const Tensor& weights, const Tensor& bias, TaskQueue& pool) {
            if(input.cols != weights.rows){
                throw std::runtime_error("Dimension mismatch for matrix multiplication.");
            }
            Tensor result(input.rows, weights.cols);
            std::atomic<int> completed_rows{0};
            for(size_t i=0; i<input.rows; ++i){
                pool.addTask(RowTask(i,input,weights,bias,result,completed_rows));
            }
            while(completed_rows < static_cast<int>(input.rows)){
                std::this_thread::yield();
            }
            return result;
    }        
};