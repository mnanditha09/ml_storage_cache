#pragma once
#include <array>
#include <cstddef>
using namespace std;

struct BlockFeatures {
    float recency;
    float frequency;
    float spatial_stride;
};

class Perceptron {
    public:
        Perceptron() : weights{0.1f,0.5f,-0.2f}, bias(0.0f), learning_rate(0.01f) {}
        float predict(const BlockFeatures& features) const;
        void train_step(const BlockFeatures& features, float label);

    private:
        array<float, 3> weights;
        float bias;
        float learning_rate; 
};