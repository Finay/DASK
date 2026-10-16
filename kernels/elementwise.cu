#include <cuda_runtime.h>


extern "C" {
    __global__ void add(const float* a, const float* b, float* c, int N) {
        int threadI = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = threadI; i < N; i += stride) {
            c[i] = a[i] + b[i];
        }
    }

    __global__ void subtract(const float* a, const float* b, float* c, int N) {
        int threadI = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = threadI; i < N; i += stride) {
            c[i] = a[i] - b[i];
        }
    }

    __global__ void divide(const float* a, const float* b, float* c, int N) {
        int threadI = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = threadI; i < N; i += stride) {
            c[i] = a[i] / b[i];
        }
    }

    __global__ void add_(float* a, const float* b, int N) {
        int threadI = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = threadI; i < N; i += stride) {
            a[i] += b[i];
        }
    }

    __global__ void subtract_(float* a, const float* b, int N) {
        int threadI = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = threadI; i < N; i += stride) {
            a[i] -= b[i];
        }
    }
}
