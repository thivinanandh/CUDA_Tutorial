# CUBLAS
---

CUBLAS is a GPU-accelerated library for linear algebra operations. It is built on top of the CUDA runtime and provides a high-performance implementation of the BLAS (Basic Linear Algebra Subprograms) library. CUBLAS provides routines for common linear algebra operations such as matrix multiplication, matrix factorization, and solving systems of linear equations.

### Transfer of Arrays for CUBLAS

Instead of `cudaMemcpy`, we can use `cublasSetVector` and `cublasGetVector` to transfer arrays to and from the GPU. These functions are optimized for transferring data to and from the GPU and can be more efficient than `cudaMemcpy` for large arrays.

```cpp
// Using cudaMemcpy for contiguous data
float* h_array = new float[N];  // Host array
float* d_array;                 // Device array
cudaMalloc(&d_array, N * sizeof(float));

// Simple contiguous copy
cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

// Using cublasSeteVector for strided data
cublasHandle_t handle;
cublasCreate(&handle);

// Copy every second element
int incx = 2;  // Stride in host array
int incy = 1;  // Stride in device array
cublasSeteVector(N/2, sizeof(float), h_array, incx, d_array, incy);
```

#### Purpose:

- Part of CUBLAS library (optimized for linear algebra)
- Handles strided data through `incx` and `incy` parameters
- Specifically designed for vector operations
- Internally optimized for BLAS operations
- Can handle non-contiguous memory through stride parameters


# Understanding CUBLAS Handles

## What is a CUBLAS Handle?

A CUBLAS handle (`cublasHandle_t`) is an opaque structure that holds the CUBLAS library context. It contains:
- Internal state information
- CUDA stream associations
- Workspace memory allocations
- Algorithm preferences
- Configuration settings

## Basic Usage

```cpp
// Creating and destroying a handle
cublasHandle_t handle;
cublasStatus_t status;

// Initialize handle
status = cublasCreate(&handle);
if (status != CUBLAS_STATUS_SUCCESS) {
    // Handle error
}

// Use handle for CUBLAS operations
// ...

// Destroy handle when done
cublasDestroy(handle);
```

## Why Are Handles Important?

1. **Resource Management**
   ```cpp
   // Handle manages internal resources
   float* d_workspace;  // Internal workspace
   size_t workspace_size;
   // CUBLAS automatically manages this through the handle
   ```

2. **Stream Association**
   ```cpp
   cudaStream_t stream;
   cudaStreamCreate(&stream);
   
   // Associate handle with a stream
   cublasSetStream(handle, stream);
   
   // Now all CUBLAS operations using this handle will use this stream
   cublasSgemm(handle, ...); // Will execute in the specified stream
   ```

3. **Math Mode Configuration**
   ```cpp
   // Set math mode for all operations using this handle
   cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
   ```

## Benefits of Using Handles

### 1. Performance Optimization
```cpp
// Single handle for multiple operations
cublasHandle_t handle;
cublasCreate(&handle);

// Multiple operations using same context
cublasSgemm(handle, ...);  // Matrix multiply 1
cublasSgemm(handle, ...);  // Matrix multiply 2
// Resources are reused efficiently
```

### 2. Memory Management
```cpp
// Handle manages workspace memory
// No need for manual workspace allocation
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             m, n, k, alpha,
             A, CUDA_R_16F, lda,
             B, CUDA_R_16F, ldb,
             beta,
             C, CUDA_R_32F, ldc,
             CUDA_R_32F,
             CUBLAS_GEMM_DEFAULT);
```

### 3. Multi-GPU Support
```cpp
// Different handles for different GPUs
cublasHandle_t handle1, handle2;

cudaSetDevice(0);
cublasCreate(&handle1);

cudaSetDevice(1);
cublasCreate(&handle2);

// Now can perform operations on different GPUs concurrently
```

## Advanced Features

### 1. Algorithm Selection
```cpp
// Set preferred algorithm through handle
cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
cublasGemmEx(handle, ..., algo);
```

### 2. Atomics Mode
```cpp
// Enable atomics mode for specific operations
cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);
```

### 3. Error Handling
```cpp
cublasStatus_t status;
// All operations with handle return status
status = cublasSgemm(handle, ...);
if (status != CUBLAS_STATUS_SUCCESS) {
    const char* error_string;
    switch(status) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            error_string = "CUBLAS not initialized";
            break;
        // ... other error cases
    }
    printf("CUBLAS error: %s\n", error_string);
}
```

## Best Practices

1. **Handle Lifetime Management**
   ```cpp
   class CublasWrapper {
   private:
       cublasHandle_t handle;
   public:
       CublasWrapper() {
           cublasCreate(&handle);
       }
       ~CublasWrapper() {
           cublasDestroy(handle);
       }
   };
   ```

2. **Stream Synchronization**
   ```cpp
   cudaStream_t stream;
   cudaStreamCreate(&stream);
   cublasSetStream(handle, stream);
   
   // Do CUBLAS operations
   cublasSgemm(handle, ...);
   
   // Synchronize when needed
   cudaStreamSynchronize(stream);
   ```

3. **Resource Reuse**
   ```cpp
   // Reuse handle for multiple operations instead of creating new ones
   for(int i = 0; i < num_operations; i++) {
       cublasSgemm(handle, ...);  // Same handle, different operations
   }
   ```

## Common Pitfalls to Avoid

1. **Creating Too Many Handles**
   - Creating a new handle for each operation is inefficient
   - Reuse handles when possible

2. **Forgetting to Destroy Handles**
   - Memory leaks can occur if handles aren't destroyed
   - Use RAII or similar patterns to manage handle lifetime

3. **Incorrect Stream Management**
   - Not synchronizing streams when needed
   - Using wrong stream associations

4. **Thread Safety**
   - Handles are not thread-safe by default
   - Use separate handles for different threads


## Math modes in CUBLAS
---

# CUBLAS Math Modes

## Overview of Math Modes

CUBLAS provides different math modes that control how mathematical operations are performed on the GPU. These modes are set using `cublasSetMathMode()`:

```cpp
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
```

## Available Math Modes

### 1. CUBLAS_DEFAULT_MATH
```cpp
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
```
- Default FP32 arithmetic
- Standard IEEE-754 compliant operations
- No tensor core acceleration
- Highest precision but lower performance
- Best for algorithms requiring strict IEEE compliance

### 2. CUBLAS_TENSOR_OP_MATH
```cpp
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
```
- Enables Tensor Core operations
- Uses mixed-precision computing
- Significantly faster than default math
- Slight reduction in precision
- Ideal for deep learning workloads

### 3. CUBLAS_PEDANTIC_MATH
```cpp
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
```
- Strict IEEE-754 compliance
- No optimizations that might affect precision
- Slowest performance
- Highest numerical accuracy
- Used for applications requiring bit-exact results

### 4. CUBLAS_TF32_TENSOR_OP_MATH
```cpp
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
```
- Uses TF32 format on Ampere and newer GPUs
- 19-bit mantissa truncated to 10 bits
- Same range as FP32
- Excellent performance/accuracy trade-off
- Primarily for AI/ML workloads

## Usage Examples

### 1. Basic Math Mode Setting
```cpp
cublasHandle_t handle;
cublasCreate(&handle);

// Set to tensor operation mode
cublasStatus_t status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
if (status != CUBLAS_STATUS_SUCCESS) {
    // Handle error
}

// Perform matrix multiplication
cublasSgemm(handle, ...);
```
