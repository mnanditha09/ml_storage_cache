#pragma once
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <condition_variable>
#include <functional>

class TaskQueue {
    private:
        std::mutex queue_mutex;
        std::condition_variable cv;
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        bool stop = false;
        void worker_loop(){
            while(true){
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    cv.wait(lock, [this] {return stop || !tasks.empty();});
                    if(stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                std::cout << "Thread " << std::this_thread::get_id() << " is processing a matrix row task.\n";
                task();
            }
        }
    public:
        TaskQueue(size_t total_threads = std::thread::hardware_concurrency()){
            for(size_t i=0; i<total_threads; ++i){
                workers.emplace_back(&TaskQueue::worker_loop,this);
            }
        }
        void addTask(std::function<void()> job){
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if(stop) return;
                tasks.push(job);
            }
            cv.notify_one();
        }
        ~TaskQueue(){
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
            }
            cv.notify_all();
            for(std::thread &worker:workers){
                if(worker.joinable()) worker.join();
            }
        }
};