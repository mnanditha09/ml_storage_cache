#include <iostream>
#include <algorithm>
#include "storage_cache.hpp"

void StorageCache::update_block_metrics(std:: shared_ptr<CacheBlock> block) {
    size_t idle_time = time_counter - block->time_last_accessed;
    block->features.recency = 1.0f / (1.0f + static_cast<float>(idle_time));
    block->time_last_accessed = time_counter;
}

void StorageCache::evict_lowest_scoring_block(){
    size_t evic_id = 0;
    float lowest_score = 2.0f;
    bool found = false;

    for(const auto& [id, block] : cache_map) {
        float score = model.predict(block->features);
        if(score < lowest_score) {
            lowest_score = score;
            evic_id = id;
            found = true;
        }
    }
    if(found) {
        std::cout << "Removing Block ID " << evic_id << "with low cache retention score: " << lowest_score << std::endl;
        cache_map.erase(evic_id); 
    }
}
 
void StorageCache::prefetch_async(size_t next_block_id) {
    if(cache_map.find(next_block_id) != cache_map.end())
        return;
    task_pool.addTask([this,next_block_id](){
        std::unique_lock<std::mutex> lock(this->cache_mutex);
        if(this->cache_map.find(next_block_id) != this->cache_map.end())
            return;
        lock.unlock();
    });
}

std::vector<char> StorageCache::read_block(size_t block_id){
    std::lock_guard<std::mutex> lock(cache_mutex);
    time_counter++;
    auto it = cache_map.find(block_id);
    if(it != cache_map.end()){
        std::cout << "Cache Hit -> Block ID: " << block_id << "\n";
        it->second->features.frequency += 1.0f;
        update_block_metrics(it->second);
        prefetch_async(block_id + 1);
        return it->second->payload;
    }
    std::cout << "Cache Miss -> Block ID: " << block_id << std::endl;

    BlockFeatures missed_feature{1.0f, 1.0f, 1.0f};
    model.train_step(missed_feature, 1.0f);
    if(cache_map.size() >= capacity) {
        evict_lowest_scoring_block();
    }
}