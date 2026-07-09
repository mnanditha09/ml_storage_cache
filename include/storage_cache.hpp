#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include "ml_model.hpp"
#include "TaskQueue.hpp"

struct CacheBlock {
    size_t block_id;
    std::vector<char> payload;
    BlockFeatures features;
    size_t time_last_accessed;
};

class StorageCache {
    private:
        std::mutex cache_mutex;
        size_t capacity;
        size_t time_counter;
        std::unordered_map<size_t, std::shared_ptr<CacheBlock>> cache_map;
        Perceptron model;
        TaskQueue& task_pool;

        void evict_lowest_scoring_block();
        void update_block_metrics(std::shared_ptr<CacheBlock> block);
        void prefetch_async(size_t next_block_id);

    public:
        StorageCache(size_t cache_capacity, TaskQueue& pool): capacity(cache_capacity), time_counter(0), task_pool(pool) {}
        std::vector<char> read_block(size_t block_id);
        void write_block(size_t block_id, const std::vector<char>& data);
        size_t current_size() const {
            return cache_map.size();
        }
        bool is_cached(size_t block_id) const {
            return cache_map.find(block_id) != cache_map.end();
        }
};