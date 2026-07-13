#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include "storage_cache.hpp"
#include "ml_model.hpp"
#include "TaskQueue.hpp"

namespace {
    size_t global_capacity = 3; 
    size_t global_time_counter = 0;
    Perceptron global_model;
    std::recursive_mutex global_cache_mutex; 
    std::unordered_map<size_t, std::shared_ptr<CacheBlock>> global_cache_map;
    
    TaskQueue* global_task_pool_ptr = nullptr;
}

void StorageCache::update_block_metrics(std::shared_ptr<CacheBlock> block) {
    block->features.recency = 1.0f;
    block->features.frequency += 1.0f;
    block->features.spatial_stride = 1.0f;
    block->time_last_accessed = global_time_counter;
}

void StorageCache::evict_lowest_scoring_block() {
    size_t evic_id = 0;
    float lowest_score = 10.0f;
    bool found = false;
    for(const auto& [id, block] : global_cache_map) {
        size_t idle_time = global_time_counter - block->time_last_accessed;
        block->features.recency = 1.0f / (1.0 + static_cast<float>(idle_time));
        float score = global_model.predict(block->features);
        if(score < lowest_score) {
            lowest_score = score;
            evic_id = id;
            found = true;
        }
    }
    if(found) {
        global_model.train_step(global_cache_map[evic_id]->features, 0.0f);
        //std::cout << "Removing Block ID " << evic_id << " with low cache retention score: " << lowest_score << "\n";
        global_cache_map.erase(evic_id);
    }
}

void StorageCache::prefetch_async(size_t next_block_id) {
    if (global_task_pool_ptr == nullptr) return;
    if(global_cache_map.find(next_block_id) != global_cache_map.end()) return;

    global_task_pool_ptr->addTask([next_block_id](){
        std::unique_lock<std::recursive_mutex> lock(global_cache_mutex);
        if(global_cache_map.find(next_block_id) != global_cache_map.end()) return;
        if(global_cache_map.size() >= global_capacity) return;
        
        //std::cout << "Fetching next Block ID: " << next_block_id << "\n";
        auto prefetched_block = std::make_shared<CacheBlock>();
        prefetched_block->block_id = next_block_id;
        prefetched_block->payload = {'p','r','e','f','e','t','c','h'}; 
        prefetched_block->time_last_accessed = global_time_counter;
        prefetched_block->features.recency = 1.0f;
        prefetched_block->features.frequency = 1.0f;
        prefetched_block->features.spatial_stride = 1.0f;
        global_cache_map[next_block_id] = prefetched_block;
    });
}

StorageCache::StorageCache(size_t cap, TaskQueue& pool) 
    : capacity(cap), task_pool(pool), time_counter(0) 
{
    global_capacity = cap;
    global_task_pool_ptr = &pool;
}

std::vector<char> StorageCache::read_block(size_t block_id) {
    std::lock_guard<std::recursive_mutex> lock(global_cache_mutex);
    global_time_counter++;
    auto it = global_cache_map.find(block_id);
    if(it != global_cache_map.end()) {
        actual_hits++; 
        //std::cout << "Cache Hit -> Block ID: " << block_id << "\n";
        update_block_metrics(it->second);
        global_model.train_step(it->second->features, 1.0f);
        
        if (global_cache_map.find(block_id + 1) == global_cache_map.end()) {
            prefetch_async(block_id + 1);
        }
        return it->second->payload;
    }
    
    actual_misses++; 
    //std::cout << "Cache Miss -> Block ID: " << block_id << "\n";
    if(global_cache_map.size() >= global_capacity) {
        /*for(auto& [id, block] : global_cache_map) {
            block->features.frequency *= 0.95f; 
        }*/
        evict_lowest_scoring_block();
    }
    
    std::vector<char> mock_disk_data = {'d','a','t','a'};
    auto missing_block = std::make_shared<CacheBlock>();
    missing_block->block_id = block_id;
    missing_block->payload = mock_disk_data;
    missing_block->time_last_accessed = global_time_counter;
    missing_block->features.recency = 1.0f;
    missing_block->features.frequency = 1.0f;
    missing_block->features.spatial_stride = 1.0f;
    global_cache_map[block_id] = missing_block;
    global_model.train_step(missing_block->features, 1.0f);
    return mock_disk_data;
}

void StorageCache::write_block(size_t block_id, const std::vector<char>& data) {
    std::lock_guard<std::recursive_mutex> lock(global_cache_mutex);
    global_time_counter++;
    auto it = global_cache_map.find(block_id);
    if(it != global_cache_map.end()) {
        it->second->payload = data;
        update_block_metrics(it->second);
    } else {
        if(global_cache_map.size() >= global_capacity) {
            /*for(auto& [id, block] : global_cache_map) {
                block->features.frequency *= 0.95f; 
            }*/
            evict_lowest_scoring_block();
        }
        auto new_block = std::make_shared<CacheBlock>();
        new_block->block_id = block_id;
        new_block->payload = data;
        new_block->time_last_accessed = global_time_counter;
        new_block->features.frequency = 1.0f;
        new_block->features.recency = 1.0f;
        new_block->features.spatial_stride = 1.0f;
        global_cache_map[block_id] = new_block;
    }
}