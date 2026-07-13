#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>
#include <chrono>
#include <iomanip>
#include <thread>
#include <random>
#include <cmath>
#include "../include/storage_cache.hpp"
#include "../include/TaskQueue.hpp"

class LRUCacheBenchmark {
private:
    size_t capacity;
    std::list<size_t> lru_list; 
    std::unordered_map<size_t, std::list<size_t>::iterator> cache_map;

public:
    size_t hits = 0;
    size_t misses = 0;
    LRUCacheBenchmark(size_t cap) : capacity(cap) {}
    void access(size_t block_id) {
        auto it = cache_map.find(block_id);
        if (it != cache_map.end()) {
            hits++;
            lru_list.erase(it->second);
            lru_list.push_front(block_id);
            cache_map[block_id] = lru_list.begin();
            return;
        }
        misses++;
        if (cache_map.size() >= capacity) {
            size_t oldest = lru_list.back();
            lru_list.pop_back();
            cache_map.erase(oldest);
        }
        lru_list.push_front(block_id);
        cache_map[block_id] = lru_list.begin();
    }
};

std::vector<size_t> generate_zipf_workload(size_t num_operations, size_t pool_size, double alpha) {
    std::vector<size_t> workload;
    workload.reserve(num_operations);
    std::default_random_engine generator(1337); 
    std::vector<double> probabilities(pool_size);
    double sum = 0.0;
    for (size_t i = 1; i <= pool_size; ++i) {
        probabilities[i - 1] = 1.0 / std::pow(static_cast<double>(i), alpha);
        sum += probabilities[i - 1];
    }
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    for (size_t i = 0; i < num_operations; ++i)
        workload.push_back(1000 + distribution(generator)); 
    return workload;
}

int main() {
    const size_t CACHE_CAPACITY = 256;
    const size_t TOTAL_OPERATIONS = 100000; 
    const size_t BLOCK_POOL_SIZE = 5000;
    const double ZIPF_SKEW_FACTOR = 0.85;  

    std::cout << "====================================================\n";
    std::cout << "     ENTERPRISE DATASET CACHE BENCHMARK (100K)      \n";
    std::cout << "====================================================\n";
    std::cout << "Generating Zipfian storage workload stream...\n";
    std::vector<size_t> workload = generate_zipf_workload(TOTAL_OPERATIONS, BLOCK_POOL_SIZE, ZIPF_SKEW_FACTOR);
    std::cout << "Generation complete. Dataset loaded into memory.\n\n";
    LRUCacheBenchmark lru_cache(CACHE_CAPACITY);
    TaskQueue pool(std::thread::hardware_concurrency());
    StorageCache custom_cache(CACHE_CAPACITY, pool);
    auto start_lru = std::chrono::high_resolution_clock::now();
    for (size_t block : workload) {
        lru_cache.access(block);
    }
    auto end_lru = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_lru = end_lru - start_lru;
    std::cout << "Executing the proposed algorithm across dataset...\n";
    auto start_custom = std::chrono::high_resolution_clock::now();
    for (size_t block : workload) {
        custom_cache.read_block(block);
    }
    auto end_custom = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_custom = end_custom - start_custom;
    size_t my_hits = custom_cache.actual_hits;
    size_t my_misses = custom_cache.actual_misses;
    double lru_hit_rate = ((double)lru_cache.hits / TOTAL_OPERATIONS) * 100.0;
    double custom_hit_rate = ((double)my_hits / TOTAL_OPERATIONS) * 100.0;
    std::cout << "\n========================================================================\n";
    std::cout << "                       LARGE DATASET FINAL RESULTS                      \n";
    std::cout << "========================================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " " << std::left << std::setw(18) << "ALGORITHM" 
            << " | " << std::right << std::setw(8) << "HITS" 
            << " | " << std::setw(8) << "MISSES" 
            << " | " << std::setw(10) << "HIT RATE" 
            << " | " << std::setw(10) << "EXEC TIME" << "\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << " " << std::left << std::setw(18) << "LRU" 
            << " | " << std::right << std::setw(8) << lru_cache.hits 
            << " | " << std::setw(8) << lru_cache.misses 
            << " | " << std::setw(9) << lru_hit_rate << "%"
            << " | " << std::setw(7) << time_lru.count() << " ms\n";
    std::cout << " " << std::left << std::setw(18) << "Proposed Algorithm" 
            << " | " << std::right << std::setw(8) << my_hits 
            << " | " << std::setw(8) << my_misses 
            << " | " << std::setw(9) << custom_hit_rate << "%"
            << " | " << std::setw(7) << time_custom.count() << " ms\n";
    std::cout << "========================================================================\n";
    return 0;
}