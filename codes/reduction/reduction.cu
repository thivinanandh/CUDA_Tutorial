#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#include "reduction_algorithms.h"

// Structure to time cuda events
struct CUDATimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDATimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CUDATimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer()
    {
        cudaEventRecord(start);
    }

    float stopTimer()
    {
        float milliseconds = 0;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

int main()
{
    // Problem size
    int N = 1 << 16;

    // Store time for all results
    float time_taken[6];

    // Host memory
    float *h_A = (float *)malloc(N * sizeof(float));
    float sum = 0.0f;

    // Initialize host memory and also compute the sum
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        sum += h_A[i];
    }

    // Device memory
    float *d_A;
    cudaMalloc(&d_A, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    // Setup thread configuration
    int threads = 256;

    // Grid size
    int blocks = (N) / threads; //

    // Allocate another array which is size of blocks
    float *d_B;
    cudaMalloc(&d_B, blocks * sizeof(float));

    // --------------------------------------------------------------------------------- //
    // ------- Kernel 1 : Non optimized reduction kernel with divergent branches ------- //
    // --------------------------------------------------------------------------------- //

    // Create cuda events to record time
    CUDATimer timer;
    timer.startTimer();
    // Call the kernel with all the blocks
    reduction<<<blocks, threads, threads * sizeof(float)>>>(d_A, d_B);
    // Synchronize
    cudaDeviceSynchronize();
    // now we have blocks number of elements in d_B, reduce that using a kernel call with 1 block
    reduction<<<1, threads, threads * sizeof(float)>>>(d_B, d_B);
    // Synchronize
    cudaDeviceSynchronize();
    time_taken[0] = timer.stopTimer();

    // Copy the result back to host, only the first element of d_B will have the result
    float result;
    cudaMemcpy(&result, d_B, sizeof(float), cudaMemcpyDeviceToHost);
    // Check the result
    assert(fabs(result - sum) < 1e-5);

    // Print the time
    printf("Time taken for reduction 1 : %f ms\n", time_taken[0]);

    // --------------------------------------------------------------------------------- //

    // --------------------------------------------------------------------------------- //
    // ------- Kernel 2 : Optimized reduction kernel with no divergent branches ------- //
    // --------------------------------------------------------------------------------- //

    // make a fresh copy of the data in device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuda events to record time
    timer.startTimer();

    // Call the kernel with all the blocks
    reduction_2<<<blocks, threads, threads * sizeof(float)>>>(d_A, d_B);
    // Synchronize
    cudaDeviceSynchronize();
    // now we have blocks number of elements in d_B, reduce that using a kernel call with 1 block
    reduction_2<<<1, threads, threads * sizeof(float)>>>(d_B, d_B);
    // Synchronize
    cudaDeviceSynchronize();

    time_taken[1] = timer.stopTimer();

    // Copy the result back to host, only the first element of d_B will have the result
    cudaMemcpy(&result, d_B, sizeof(float), cudaMemcpyDeviceToHost);
    // Check the result
    assert(fabs(result - sum) < 1e-5);

    // Print the time
    printf("Time taken for reduction 2 : %f ms\n", time_taken[1]);

    // --------------------------------------------------------------------------------- //

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
}