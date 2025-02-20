// This program is to reduce an array using cuda. 
// 

__global__ void reduction(int* g_idata, int *g_odata)
{
    // Copy the corresponding arrays to the shared memory
    extern __shared__ int sdata[];

    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // copy corresponding block to the shared memory
    sdata[tid] = g_idata[i];

    __syncthreads();

    // loop
    for(int s = 1 ; s < blockDim.x ; s *= 2)
    {   
        if( tid % (2*s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads(); 
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// Reduction 2 

__global__ void reduction_2(int* g_idata, int *g_odata)
{
    // Copy the corresponding arrays to the shared memory
    extern __shared__ int sdata[];

    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // copy corresponding block to the shared memory
    sdata[tid] = g_idata[i];

    __syncthreads();

    // Inner loop to be thread divergent
    // In 1st loop, thread 1d 0: ( operate on 0, 1 )
    //              thread 1d 1: ( operate on 2, 3 )                        
    for(int s = 1 ; s < blockDim.x ; s *= 2)
    {   
        int index = 2 * s * tid;
        if(index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Reduction 3 --> Avoiding bank conflicts
// Accessing elements next to each other creates bank conflicts.. 
// So we will access elements which are far apart. 

__global__ void reduction_3(int* g_idata, int *g_odata)
{
    // Copy the corresponding arrays to the shared memory
    extern __shared__ int sdata[];

    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // copy corresponding block to the shared memory
    sdata[tid] = g_idata[i];

    __syncthreads();

    // Inner loop
    // Start from a larger stride and go untill its smaller                   
    for(int s = blockDim.x/2 ; s > 0 ; s >>=1)
    {   
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Reduction 3 -->Occupying all the threads .. previous operation leaves half the threads ( certain blocks )
// to be completely in active. 
// So we will try to actually half the number of blocks required by adding two elements, while loading the array into shared memory
// and proceed only with half the arrays, 

__global__ void reduction_4(int* g_idata, int *g_odata)
{
    // Copy the corresponding arrays to the shared memory
    extern __shared__ int sdata[];

    // Get the index, cos
    int i =  (blockDim.x*2) * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // copy corresponding block to the shared memory
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];

    __syncthreads();

    // Inner loop
    // Start from a larger stride and go untill its smaller     
    // Since the first reduction is already performed, We will start from 2nd reduction stage               
    for(int s = blockDim.x/2 ; s > 0 ; s >>=1)
    {   
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// Reduction 5: UNROLLING Last Wrap
// Now, when the Stride reaches 32 (when only 32 threads are suppsoed to be active)
// 1, We do not need to sync them as the warp size is 32 and its synced
// 2, At this level we still need to perform reduction at strides, 16 followed by 8, followed by 4 and 2 
//    Since this is happening at warp level, this will now introduce thread divergence, So what we will do is 
//    unroll these 5 loops into set of sequential instructions which can run without thread divergence in a single operation

__device__ void warpReduce(volatile int* sdata, int tid)
{
    // At stride len of 32, we need to add elements offset by 32
    sdata[tid] += sdata[tid + 32];
    // Now at stride len of 16, we need to add elements offset by 16
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduction_5(int* g_idata, int *g_odata)
{
    // Copy the corresponding arrays to the shared memory
    extern __shared__ int sdata[];

    // Get the index, cos
    int i =  (blockDim.x*2) * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // copy corresponding block to the shared memory
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];

    __syncthreads();

    // Inner loop
    // Start from a larger stride and go untill its smaller     
    // Since the first reduction is already performed, We will start from 2nd reduction stage               
    for(int s = blockDim.x/2 ; s >32  ; s >>=1)
    {   
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid < 32) warpReduce(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}