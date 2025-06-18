# CUDA-C Square Matrix Multiplication
As the title says, this program simply computes for the product matrix of two input square matrices written in CUDA-C; an extension of ANSI C for general-purpose programming on CUDA-enabled NVidia GPUs.

Though the program uses the **CPU** _(Host)_ for the majority of the tasks, such as scanning input matrices matrix memory allocation, the **GPU** _(Device)_ does most of the heavy-lifting. That is, it performs the matrix multiplication via a **kernel** invocation which launches a large number of threads.


### PREREQUISITES AND HOW TO RUN
In order to compile and run this program, you need to have a CUDA-enabled NVidia GPU as well as the NVidia CUDA toolkit installed.

To compile this program, navigate to the working directory and use _nvcc_ :
```
 nvcc <file1.cu> <file2.cu> <..> -o <output.exe>
```
To run the program, do so like how any C/C++ program is usually ran :
```
 ./output.exe
```

## Why on the GPU?
The idea behind the choice is simple. A product matrix is composed of all of the dot products for each row and column pair of two input matrices. A CPU is actually capable of performing these series of calculations. But for matrices with larger dimensions, it becomes inefficient
and slow. This is where we leverage the GPU's computing prowess.

A GPU uses threads to perform a computation. Unlike CPUs, which normally do work sequentially, GPUs excel at parallel computations and high throughput allowing them to perform computational workloads at a significantly faster rate. This is due to the fact that most of the
chip area on a GPU is composed of massively large amounts of slower cores. In contrast, CPU chip area is dedicated to fewer, more powerful cores and larger cache memory for lower latency.

Each individual element of the product matrix can be represented and calculated using a thread. So, if we are to multiply two M x M matrices, we would need M x M threads to represent each element of the product. Using the GPU for this computational task also makes sense because
it is highly parallelizable. That is, computing the dot product between a row and column of the two matrices, regardless of the order of which we select which row-col pair comes first, will not affect the final result as the dot products do not depend on each other.

## CUDA Thread Organization
In CUDA, threads are the smallest unit of computation which are tasked with executing a kernel function on the GPU concurrently. A group of threads can be represented as a **block** and a group of blocks is known as a **grid**, which can both have up to three dimensions. In 
newer GPU architectures, each block has a maximum number of 1024 threads across all dimensions(512 threads for older architectures). The maximum number of blocks depend on the specific GPU architecture. For the Ada Lovelace architecture, the maximum number of blocks
is 32 per streaming multiprocessor (SM).

## Problem Specification
The problem used to test the program is to find the product of two 1000 x 1000 matrices filled with random integer values in the range [0,10].

Given arrays of dimension 1000 x 1000, the product matrix will also have a dimension 1000 x 1000. To calculate the product matrix, we will need 1000 x 1000 = 1 000 000 threads.

Before deciding on the block and grid dimensions to be used, we should first determine the hardware specifications of the device used to run the kernel.

**GPU** : NVidia GeForce RTX 5070 

| Feature                   |     Value      |
|:--------------------------|:--------------:|
| VRAM                      |  12 GB GDDR7   | 
| Threads per Block         | 1024 (Typical) | 
| Streaming Multiprocessors |       48       | 


For a 1000 x 1000 matrix, we can "tile" or break the matrix into smaller parts using a 16x16 thread block. This gives 256 threads to work on 256 corresponding elements on the matrix. A 32x32 = 1024 thread configuration would work but using the maximum number of threads
on a single block may affect warp scheduling, induce pressure on shared memory, etc.

Now, for a 16x16 thread block, we would need a 63x63 grid. This gives 63 x 63 = 3969 thread blocks, and 16 x 16 = 256 threads per block. So 3969 x 256 = 1016064 total threads to calculate each product matrix element.

The problem, however, only needs exactly 1 000 000 threads. Though there is a way to launch an exact number of threads, it is not feasible to do so as CUDA prefers thread blocks with a size that is a multiple of 32 (standard warp size). Also, you would want maximize occupancy
of the GPU and thread grids align with memory access patterns. So, with the current setup, there will be 16064 threads wasted but this is okay (in this case, at least) since GPU's focus on throughput rather than per-thread efficiency. 




More information can be found within the code. As of now, the kernel uses the naive approach in accessing global memory. That is, techniques such as memory coalescing are not currently implemented. 

