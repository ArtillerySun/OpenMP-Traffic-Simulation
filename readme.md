# High-Performance Parallel Traffic Simulation (C++ & OpenMP)

This project implements a high-performance, double-lane traffic flow simulation using **C++** and **OpenMP**. It is specifically designed to handle large-scale, computationally intensive scenarios by maximizing parallel efficiency and demonstrating expertise in low-level system optimization.

### Key Technical Achievements

* **Extreme Parallel Efficiency:** The simulator achieved a peak **1400% CPU Utilization** on the Intel Xeon Server (xs-4114) for a $10^6$ data scale. This performance successfully demonstrates near-linear parallel scaling across approximately 14 cores.
* **Hardware-Optimized Search (Bitmask):** The critical neighbor car search was optimized using a **Bitmask** data structure combined with **GCC Built-in Functions** (`__builtin_ctzll`, `__builtin_clzll`). This allows for checking **64 road positions in a single CPU instruction**, drastically reducing the core loop's execution time.
* **Lock-Free Synchronization Model:** A robust, two-stage parallel model employs **Double Buffering** (using `std::vector<int>` for current/next states) to prevent data races. This approach eliminates the overhead of mutex locks, ensuring high-speed, synchronous updates critical for traffic simulation correctness.
* **Data Locality:** Work is manually distributed into fixed chunks per thread. This strategy enhances **data locality** by ensuring each thread continuously works on the same subset of the car array, minimizing cache misses.

### Implementation and Parallel Strategy

The simulation is structured to guarantee correctness while achieving maximal speed:

#### 1. Synchronization and Update

* **Double Buffering:** Threads read from one buffer and write to the next buffer simultaneously. Buffers are swapped only once per global time step, guaranteeing that all cars make decisions based on the *exact* previous state.
* **Two-Stage Parallel Regions:** Each time step is split into two distinct `#pragma omp parallel` regions, utilizing the implicit barrier between them to enforce synchronization:
    1.  Lane Change Decision Phase.
    2.  Velocity and Position Update Phase.

#### 2. Performance Analysis

The achieved high CPU utilization confirms the program's efficiency in leveraging multi-core hardware.

| Platform | Data Scale | Peak CPU Utilization | Note |
| :--- | :--- | :--- | :--- |
| **Xeon Server (xs-4114)** | $10^6$ | $\sim 1400%$ | High server core utilization |
| **i7-13700 (Desktop)** | $10^6$ | $\sim 585%$ | Efficient desktop core utilization |

### Build Instructions

The project uses a standard `Makefile` for compilation.

```bash
# Clone the repository
git clone [https://github.com/ArtillerySun/OpenMP-Traffic-Simulation.git](https://github.com/ArtillerySun/OpenMP-Traffic-Simulation.git)
cd OpenMP-Traffic-Simulation

# Compile the final Bitmask version
make

# Run the simulation (example: running 100 steps on the 1e6 input file)
./sim.perf < [input_file]
