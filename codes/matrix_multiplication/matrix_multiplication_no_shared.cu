// Author : Thivin Anandh
// Purpose : Baselevel CUDA kernel matrix multiplication code without shared memory

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>


__global__ void matrix_multiplication(double *A, double* B, double* C, int N)
{
    // Get the the thread parameters
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    // for each row perform the computations
    if (row < N)
    {
        // loop over j
        for (int j = 0 ; j < N ; j++)
        {
            C[row*N + j] = 0.0;
            for (int k = 0; k < N ; k++)
            {
                C[row*N + j] += A[row*N + k] * B[N*k + j];
            }
        }
    }
}

void matrix_multiplication_host(double *A, double* B, double* C, int N)
{
    // loop over i
    for (int i = 0 ; i < N ; i++)
    {
        // loop over j
        for (int j = 0 ; j < N ; j++)
        {
            C[i*N + j] = 0.0;
            for (int k = 0; k < N ; k++)
            {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}



// main function
int main(int argc, char** argv)
{
    // Allocate double array for storing matrix
    double *host_A, *host_B, *host_C;
    double *device_A, *device_B, *device_C;

    // Matrix size
    int N = 1000;

    //Allocate memory on host
    host_A = (double*)malloc(N*N*sizeof(double));
    host_B = (double*)malloc(N*N*sizeof(double));
    host_C = (double*)malloc(N*N*sizeof(double));

    // Initialize matrix
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            host_A[i*N+j] = i-j + 3;
            host_B[i*N+j] = i+j - 2;
        }
    }

    // Allocate memory on device
    cudaMalloc(&device_A, N*N*sizeof(double));
    cudaMalloc(&device_B, N*N*sizeof(double));
    cudaMalloc(&device_C, N*N*sizeof(double));

    // Copy data from host to device
    cudaMemcpy(device_A, host_A, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, host_C, N*N*sizeof(double), cudaMemcpyHostToDevice);

    // lets setup the grid and block size
    int num_threads = 256;
    int thread_block_size = ceil(N*N/num_threads);

    // Start the timer
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch kernel
    matrix_multiplication<<<thread_block_size, num_threads>>>(device_A, device_B, device_C, N);


    // Stop the timer
    cudaEventRecord(stop);

    // Synchronize the events
    cudaEventSynchronize(stop);

    // Calculate the time
    float milliseconds_gpu = 0;
    cudaEventElapsedTime(&milliseconds_gpu, start, stop);

    printf("Time taken for matrix multiplication is %f ms\n", milliseconds_gpu);

    // wait for the kernel to finish
    cudaDeviceSynchronize();

    // transfer the array to host
    cudaMemcpy(host_C, device_C, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    // create a new array for validation of answr
    double* C_check = (double*)malloc(N*N*sizeof(double));

    // time the host code
    cudaEventRecord(start);

    // callhost matrix multiplication routines
    matrix_multiplication_host(host_A, host_B, host_C, N);

    // Stop the timer
    cudaEventRecord(stop);

    // Synchronize the events
    cudaEventSynchronize(stop);

    // Calculate the time
    float milliseconds_host = 0;
    cudaEventElapsedTime(&milliseconds_host, start, stop);

    printf("Time taken for matrix multiplication on host is %f ms\n", milliseconds_host);

    // print the speedup
    printf("Speedup is %f\n", milliseconds_host/milliseconds_gpu);


    // check the results
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            if (host_C[i*N+j] != host_C[i*N+j])
            {
                printf("Error in the results\n");
                return 1;
            }
        }
    }

    return 0;
}