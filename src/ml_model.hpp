#pragma once
#include <array>
#include <cstddef>
#include <cmath> 

struct BlockFeatures {
    float recency;
    float frequency;
    float spatial_stride;
};

class Perceptron {
public:
    Perceptron() : weights{0.8f, 0.5f, 0.1f}, bias(0.0f), learning_rate(0.01f) {}
    inline float predict(const BlockFeatures& features) const {
        float dot_product = (features.recency * weights[0]) + 
                            (features.frequency * weights[1]) + 
                            (features.spatial_stride * weights[2]);
        return 1.0f / (1.0f + std::exp(-(dot_product + bias)));
    }
    inline void train_step(const BlockFeatures& features, float label) {
        float prediction = predict(features);
        float error = label - prediction;
        float gradient = error * prediction * (1.0f - prediction); 
        weights[0] += learning_rate * gradient * features.recency;
        weights[1] += learning_rate * gradient * features.frequency;
        weights[2] += learning_rate * gradient * features.spatial_stride;
        bias += learning_rate * gradient;
    }
    
private:
    std::array<float, 3> weights;
    float bias;
    float learning_rate;
};