# Predictive Storage Cache Simulator

An enterprise-grade, multi-threaded block storage cache simulator featuring an online, inline **Sigmoid Perceptron (Logistic Regression)** inference and training loop for predictive block eviction and hardware-throttled asynchronous prefetching.

---

## 🛠️ Architectural Paradigm Shift: Why This Engine Beats LRU

Traditional storage caching strategies rely on static heuristics like **LRU (Least Recently Used)** that use O(1) pointer manipulations to infer data value. While computationally cheap, LRU is inherently short-sighted. It suffers from severe cache pollution and performance thrashing because it assumes that whatever was accessed most recently is universally the most valuable block.

This repository implements a **Learned Infrastructure** model designed for high-density I/O platforms (SAN/NAS/NVMe-oF fabric controllers). Instead of applying flat rules, it uses a lightweight online machine learning model to track multi-dimensional metadata features (`recency`, `frequency`, and `spatial_stride`). 

By mapping these access metrics through an online Sigmoid activation layer, the system forms a highly refined linear separation boundary. It retains a long-term memory of high-frequency Logical Block Addresses (LBAs), **capturing thousands of critical cache hits that traditional LRU completely drops.**

---

## 🚀 Large-Scale Enterprise Dataset Evaluation

The simulator evaluates the custom predictive runtime directly against an industry-standard LRU engine. 

### 📊 Evaluation Workload Metrics

To validate performance under realistic production conditions, the cache structures were subjected to a high-volume streaming workload:
* **Dataset Scale:** 100,000 discrete Logical Block Address (LBA) storage operations.
* **Data Skew Model:** Mathematically generated **Zipfian Distribution (α = 0.85)** over a 5,000-block pool, precisely modeling the heavy temporal locality and 80/20 data skew seen in production enterprise architectures.
* **Cache Bounding:** Static Cache Capacity constrained to **256 Blocks** to simulate a highly competitive cache-pressure environment.

### 📈 Comparative Scorecard

```text
========================================================================
 ALGORITHM          |   HITS   |  MISSES  |  HIT RATE  |  EXEC TIME 
------------------------------------------------------------------------
 LRU                |   37298  |   62702  |   37.30%   |   34.46 ms
 Proposed Algorithm |   47644  |   52356  |   47.64%   |  296.63 ms
========================================================================
```

## 🧠 Deep-Dive Engineering Insights

* **Stable Error Tracking via Sigmoid:** A raw perceptron outputs unbounded linear numbers, causing training weights to overshoot and "forget" historic block value. Bounding output scores onto a strict $0.0$ to $1.0$ probability scale using a Sigmoid function enables precise, fractional reinforcement updates ($1.0f$ for hits, $0.0f$ for structural evictions).
* **Multi-Threaded Lock-Contention Mitigation:** Asynchronous sequential prefetching cascades often choke low-level worker pools. This architecture solves the bottleneck by placing a **non-blocking atomic fast-check barrier** directly before task injection in the `TaskQueue` thread pool. If the next sequential LBA descriptor is already resident, the task drops instantly, eliminating thread contention and preventing context-switching overhead under continuous I/O saturation.

---

## 🎛️ Low-Level Core Primitives & Systems Pipeline

1. **Host I/O Intercept:** System acquires a lock-guarded `std::recursive_mutex` thread boundary context to check block residency.
2. **Cache Hit Subsystem:** Increments global atomic `actual_hits`. Executes model online reinforcement loop via `train_step(..., 1.0f)` to strengthen weight coefficients for active features. Dispatches zero-copy non-blocking `prefetch_async` pointer commands to the worker queue.
3. **Cache Miss & Eviction Subsystem:** Increments global atomic `actual_misses`. If cache footprint reaches capacity constraint, an inline Sigmoid probability pass isolates the lowest-scoring metadata block. The losing descriptor is penalized via `train_step(..., 0.0f)` to adapt the linear separation boundary against historical noise and is evicted from memory to accept incoming disk payload.

---

## 💻 Tech Stack & Engineering Primitives

* **System Language:** Modern C++ (C++20 optimized structures)
* **Concurrency Primitives:** Lock-free atomic synchronization (`std::atomic`), worker thread distribution via a custom condition variable engine (`std::condition_variable`, `std::mutex`), RAII stack lifecycle management.
* **Inline AI Architecture:** Lightweight, dependency-free online Logistic Regression engine compiled with internal header space optimization to leverage compiler-driven inline acceleration.

---

## 📦 Compilation & Deployment Environment

The repository utilizes **CMake** for build automation and requires a compiler that supports **C++20** primitives. For maximum throughput performance, the project must be compiled with high-efficiency Release optimizations enabled:

```bash
# Configure the build directory with target C++20 release optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the system binary artifacts
cmake --build build

# Execute the high-volume streaming data simulation pipeline
./build/ml_cache_sim
```

---
