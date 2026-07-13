#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include "ml_model.hpp"

class TaskQueue;

struct CacheBlock {
    size_t block_id;
    std::vector<char> payload;
    size_t time_last_accessed;
    BlockFeatures features; 
};

class StorageCache {
public:
    StorageCache(size_t cap, TaskQueue& pool);
    ~StorageCache() = default;
    std::vector<char> read_block(size_t block_id);
    void write_block(size_t block_id, const std::vector<char>& data);
    size_t actual_hits = 0;
    size_t actual_misses = 0;

private:
    void update_block_metrics(std::shared_ptr<CacheBlock> block);
    void evict_lowest_scoring_block();
    void prefetch_async(size_t next_block_id);

    size_t capacity;
    TaskQueue& task_pool;
    size_t time_counter;
    Perceptron model; 
    std::mutex cache_mutex; 
    std::unordered_map<size_t, std::shared_ptr<CacheBlock>> cache_map;
};