// Author : Thivin Anandh
// Purpose : Baselevel CUDA kernel matrix multiplication code with Shared Memory - Tile Matrix Multiplication

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

constexpr int TILE_WIDTH = 16;

__global__ void matrix_multiplication_shared(double* A, double* B, double* C, int N)
{
    // create shared memory for the tiles , This is for each thread block
    __shared__ double shared_a[TILE_WIDTH * TILE_WIDTH];
    __shared__ double shared_b[TILE_WIDTH * TILE_WIDTH];

    // Compute the row and column number
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    int num_tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    // Setup Temp sum value to 0
    double ans = 0.0;

    // Loop over the major tile dimension ( along n )
    for (int tile_id = 0 ; tile_id < num_tiles ; tile_id++ )
    {
        // --------------------------------------------------------------- 
        // Fill the Current tile of A and B which contributes to C
        // ---------------------------------------------------------------
        // index of the shared array to fill,  this will just be the Idx within the block 
        int shared_index = threadIdx.y * TILE_WIDTH + threadIdx.x;

        // index of global array which is gonna be filled in Shared Array
        int global_index_A = (row * N) + (tile_id * TILE_WIDTH) + threadIdx.x;
        int global_index_B = (tile_id * TILE_WIDTH + threadIdx.y)*N  +  col;

        // Fill the Shared Array
        if (row < N && ( (tile_id * TILE_WIDTH) + threadIdx.x ) < N  )
            shared_a[shared_index] = A[global_index_A];
        else
            shared_a[shared_index] = 0.0;
        if (col < N && (tile_id * TILE_WIDTH + threadIdx.y) < N )
            shared_b[shared_index] = B[global_index_B];
        else
            shared_b[shared_index] = 0.0;

        // Synchronize the threads
        __syncthreads();

        // ---------------------------------------------------------------------
        //  Loop over the internal Dimension of the tile to sum for each thread
        // ---------------------------------------------------------------------

        for(int k = 0 ; k < TILE_WIDTH ; k++)
        {
            // ans += shared(i*k) + shared(k*j)
            ans += shared_a[threadIdx.y * TILE_WIDTH + k] * shared_b[k * TILE_WIDTH + threadIdx.x];
        }

        // Synchronise the threads
        __syncthreads();
    }

    // Fill the Solution on global array
    if ( row < N && col < N)
        C[row * N + col] = ans;
}

// function for tiled matrix multiplication in CPU
void matrix_multiplication_host(double *A, double* B, double* C, int N)
{
    // Initialize C to zero first
    for(int i = 0; i < N*N; i++) {
        C[i] = 0.0;
    }

    // Loop over the tile dimension
    for(int i = 0; i < N; i += TILE_WIDTH)
    {
        for(int j = 0; j < N; j += TILE_WIDTH)
        {
            for(int k = 0; k < N; k += TILE_WIDTH)
            {
                // Loop over the internal dimension of the tile
                // Use min to handle boundary cases
                for(int ii = i; ii < min(i + TILE_WIDTH, N); ii++)
                {
                    for(int jj = j; jj < min(j + TILE_WIDTH, N); jj++)
                    {
                        for(int kk = k; kk < min(k + TILE_WIDTH, N); kk++)
                        {
                            C[ii*N + jj] += A[ii*N + kk] * B[kk*N + jj];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    // Allocate the arrays
    int N = 1024;

    // Allocate memory on host
    double *host_A, *host_B, *host_C;
    host_A = (double*)malloc(N*N*sizeof(double));
    host_B = (double*)malloc(N*N*sizeof(double));
    host_C = (double*)malloc(N*N*sizeof(double));

    //Initialize the arrays
    for (int i = 0 ; i < N ; i++)
    {
        for (int j = 0 ; j < N ; j++)
        {
            host_A[i*N + j] = i - j + 5;
            host_B[i*N + j] = i + j;
        }
    }
    
    // Allocate memory on device
    double *device_A, *device_B, *device_C;

    cudaMalloc(&device_A, N*N*sizeof(double));
    cudaMalloc(&device_B, N*N*sizeof(double));
    cudaMalloc(&device_C, N*N*sizeof(double));

    // Copy the data to the device
    cudaMemcpy(device_A, host_A, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, N*N*sizeof(double), cudaMemcpyHostToDevice);
    
    // Parameters of tiled matrix multiplication
    int num_threads_1d  =   TILE_WIDTH;
    int num_blocks_1d   =   ceil(N/num_threads_1d);

    // setup 2d grid for cuda
    dim3 dimGrid(num_blocks_1d, num_blocks_1d);
    dim3 dimBlock(num_threads_1d, num_threads_1d);

    // create cuda event for time calculation
    cudaEvent_t start, stop;
    // create the events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    cudaEventRecord(start);
    // Launch the kernel
    matrix_multiplication_shared<<<dimGrid, dimBlock>>>(device_A, device_B, device_C, N);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Stop the timer
    cudaEventRecord(stop);
    // Synchronise for stop event
    cudaEventSynchronize(stop);

    // Calculate the time
    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);

    

    // Copy the solutions to the host
    cudaMemcpy(host_C, device_C, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    // Create one more array for CPU validation
    double *host_C_CPU;
    host_C_CPU = (double*)malloc(N*N*sizeof(double));

    // setup the CPU timer
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    // Start the timer
    cudaEventRecord(start_cpu);

    // Call the CPU function
    matrix_multiplication_host(host_A, host_B, host_C_CPU, N);

    // Stop the timer
    cudaEventRecord(stop_cpu);

    // Synchronize the events
    cudaEventSynchronize(stop_cpu);

    // Calculate the time
    float time_cpu = 0.0;
    cudaEventElapsedTime(&time_cpu, start_cpu, stop_cpu);


    // Compare the results
    for (int i = 0 ; i < N ; i++)
    {
        for (int j = 0 ; j < N ; j++)
        {
            if (host_C[i*N + j] != host_C_CPU[i*N + j])
            {
                printf("Mismatch at %d %d : Expected Val: %f : Actual : %f\n", i, j,host_C[i*N + j],host_C_CPU[i*N + j] );
                return 0;
            }
        }
    }

    printf("Time taken for CPU matrix multiplication is %f ms\n", time_cpu);
    printf("Time taken for GPU matrix multiplication is %f ms\n", time);

    // print the speedup
    printf("Speedup is %f\n", time_cpu/time);

    

    return 0;
}