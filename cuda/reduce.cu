#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define DEVICE_ID 0
#define BLOCK_SIZE 4

#define MALLOC_CHECK(ptr)                                                      \
  {                                                                            \
    if (!ptr) {                                                                \
      fprintf(stderr, "Error allocating memory! %s %d\n", __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Cuda runtime error: %s %s %d\n",                        \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

int cpu_reduce(int *data, const int size) {
  if (size == 1)
    return data[0];

  const int stride = size / 2;
  for (int i = 0; i < stride; i++) {
    data[i] += data[i + stride];
  }
  if (size % 2 == 1) {
    data[0] += data[size - 1];
  }
  return cpu_reduce(data, stride);
} /* cpu_reduce */

__global__ void gpu_reduce(int *g_idata, int *g_odata, size_t n) {
  int tid = threadIdx.x;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  int *idata = g_idata + blockDim.x * blockIdx.x;

  if (idx >= n) {
    return;
  }

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
} /* gpu_reduce */

int main(int argc, char **argv) {
  int size = (1 << 3) + 2;
  dim3 block(BLOCK_SIZE);
  dim3 grid((size + block.x - 1) / block.x);
  printf("Grid: %d\tBlock: %d\n", block.x, grid.x);

  size_t bytes = size * sizeof(int);
  int *h_idata = (int *)malloc(bytes);
  MALLOC_CHECK(h_idata);
  int *h_odata = (int *)malloc(grid.x * sizeof(int));
  MALLOC_CHECK(h_odata);
  int *temp = (int *)malloc(bytes);
  MALLOC_CHECK(temp);

  for (int i = 0; i < size; i++) {
    h_idata[i] = (int)(rand() & 0xFF);
  }
  memcpy(temp, h_idata, bytes);

  int *d_idata = NULL;
  int *d_odata = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_idata, bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_odata, grid.x * sizeof(int)));

  // cpu reduction
  printf("CPU reduction: %d\n", cpu_reduce(temp, size));

  // gpu reduction
  CUDA_CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
  gpu_reduce<<<grid, block>>>(d_idata, d_odata, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                        cudaMemcpyDeviceToHost));

  int gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += h_odata[i];
  }
  printf("GPU reduction: %d\n", gpu_sum);
} /* main */
