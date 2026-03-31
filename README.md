# High-Performance Parallel Computation in C (MPI)

## Overview
This project focuses on optimizing a parallel computation algorithm using **C** and the **Message Passing Interface (MPI)**. The goal was to refactor a naive distributed memory approach into a highly optimized, scalable solution by minimizing network latency and synchronization overhead.

*This project was developed as part of the High-Performance Computing course at Universidad Autónoma de Madrid (UAM).*

## The Problem: Point-to-Point Bottleneck (`baseline_compute.c`)
The initial implementation relied on blocking, point-to-point communications (`MPI_Send` and `MPI_Recv`) inside a loop. 
* **The issue:** The root process acted as a bottleneck, sending and receiving chunks of data sequentially to each worker node. This caused massive synchronization overhead and poor scalability as the number of processes increased.

* ## The Solution: Collective Operations (`optimized_mpi_compute.c`)
To achieve true high performance, the architecture was redesigned using **MPI Collective Operations**:
* Replaced manual loops with `MPI_Scatterv` to distribute dynamically sized data chunks across all nodes simultaneously.
* Implemented `MPI_Reduce` to aggregate the local computations into the global result in a highly optimized tree-structure.
* Added accurate profiling using `MPI_Barrier` and `MPI_Wtime` to isolate pure computation time from network communication latency.
