#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256

#define MALLOC_CHECK(ptr)                                                      \
  {                                                                            \
    if (!ptr) {                                                                \
      fprintf(stderr, "Error allocating memory at %s: %d\n", __FILE__,         \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Cuda runtime error at %s: %d %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

void init_array(float *m, size_t len);
void cpu_reduction(float *in, double *out, size_t len);
__global__ void gpu_reduction(float *in, double *out, size_t len);
__global__ void gpu_reduction_cascaded(float *in, double *out, size_t len);
float benchmark_gpu_reduction(dim3 grid, dim3 block, float *d_in, double *d_out,
                              size_t len);
float benchmark_gpu_reduction_cascaded(dim3 grid, dim3 block, float *d_in,
                                       double *d_out, size_t len);
void log_results(const char *filename, float cpu_time_ms,
                 float gpu_time_global_ms, float gpu_time_cascaded_ms);

int main() {
  int array_size = 40 * 1024 * 1024;
  int array_length = array_size / sizeof(float);

  // input array setup and initialization
  float *arr = (float *)malloc(array_size);
  MALLOC_CHECK(arr);
  init_array(arr, array_length);

  double cpu_sum;
  float exec_time_cpu;
  clock_t start_cpu = clock();
  cpu_reduction(arr, &cpu_sum, array_length);
  clock_t end_cpu = clock();
  exec_time_cpu = (float)(end_cpu - start_cpu) * 1000.0f / CLOCKS_PER_SEC;

  // device launch initialization
  dim3 grid, block;
  // device memory setup and initialization
  float *d_arr;
  double *d_res;

  CUDA_CHECK(cudaMalloc((void **)&d_arr, array_size));
  CUDA_CHECK(cudaMalloc((void **)&d_res, sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_arr, arr, array_size, cudaMemcpyHostToDevice));

  /*  Global Atomic Reduction */
  // kernel launch
  block.x = BLOCK_SIZE;
  grid.x = (array_length + block.x - 1) / block.x;

  CUDA_CHECK(cudaMemset(d_res, 0, sizeof(double)));
  gpu_reduction<<<grid, block>>>(d_arr, d_res, array_length);
  CUDA_CHECK(cudaGetLastError());

  // copying result back to host
  double gpu_sum;
  CUDA_CHECK(
      cudaMemcpy(&gpu_sum, d_res, sizeof(double), cudaMemcpyDeviceToHost));

  // correction check
  if (fabs(gpu_sum - cpu_sum) > 1e-6) {
    printf("Results dont match! %lf %lf\n", cpu_sum, gpu_sum);
  }

  // benchmark
  CUDA_CHECK(cudaMemset(d_res, 0, sizeof(double)));
  float exec_time_gpu_reduction =
      benchmark_gpu_reduction(grid, block, d_arr, d_res, array_length);

  /* Cascaded Reduction */
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned int grid_size =
      prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor;

  // kernel launch
  block.x = BLOCK_SIZE;
  grid.x = grid_size;

  CUDA_CHECK(cudaMemset(d_res, 0, sizeof(double)));
  gpu_reduction_cascaded<<<grid, block>>>(d_arr, d_res, array_length);
  CUDA_CHECK(cudaGetLastError());

  // copying result back to host
  double gpu_sum_cascaded;
  CUDA_CHECK(cudaMemcpy(&gpu_sum_cascaded, d_res, sizeof(double),
                        cudaMemcpyDeviceToHost));

  // correction check
  if (fabs(gpu_sum_cascaded - cpu_sum) > 1e-6) {
    printf("Results dont match! %lf %lf\n", cpu_sum, gpu_sum);
  }

  // benchmark
  CUDA_CHECK(cudaMemset(d_res, 0, sizeof(double)));
  float exec_time_gpu_reduction_cascaded =
      benchmark_gpu_reduction_cascaded(grid, block, d_arr, d_res, array_length);

  // cleanup
  free(arr);
  CUDA_CHECK(cudaFree(d_arr));
  CUDA_CHECK(cudaFree(d_res));

  log_results("metrics.csv", exec_time_cpu, exec_time_gpu_reduction,
              exec_time_gpu_reduction_cascaded);
  return 0;
} /* main */

void init_array(float *m, size_t len) {
  for (size_t i = 0; i < len; i++) {
    m[i] = (float)rand() / RAND_MAX;
  }
} /* init_matrix */

void cpu_reduction(float *in, double *out, size_t len) {
  double sum = 0.0;
  for (size_t i = 0; i < len; i++) {
    sum += (double)in[i];
  }
  *out = sum;
} /* cpu_reduction */

__global__ void gpu_reduction(float *in, double *out, size_t len) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len) {
    atomicAdd(out, (double)in[tid]);
  }
} /* gpu_reduction_global */

__global__ void gpu_reduction_cascaded(float *in, double *out, size_t len) {

  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  __shared__ double block_sum;
  if (threadIdx.x == 0) {
    block_sum = 0.0;
  }
  __syncthreads();

  // thread level reduction
  double local_sum = 0.0f;
  for (size_t i = tid; i < len; i += stride) {
    local_sum += (double)in[i];
  }

  // block level reduction
  atomicAdd(&block_sum, local_sum);
  __syncthreads();

  // grid level aggregation
  if (threadIdx.x == 0) {
    atomicAdd(out, block_sum);
  }
} /* gpu_reduction_cascaded */

float benchmark_gpu_reduction(dim3 grid, dim3 block, float *d_in, double *d_out,
                              size_t len) {
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // warmup run
  gpu_reduction<<<grid, block>>>(d_in, d_out, len);
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));

  // timed run
  CUDA_CHECK(cudaEventRecord(start, 0));
  gpu_reduction<<<grid, block>>>(d_in, d_out, len);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float exec_time;
  CUDA_CHECK(cudaEventElapsedTime(&exec_time, start, end));

  return exec_time;
} /* benchmark_gpu_reduction*/

float benchmark_gpu_reduction_cascaded(dim3 grid, dim3 block, float *d_in,
                                       double *d_out, size_t len) {
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // warmup run
  gpu_reduction_cascaded<<<grid, block>>>(d_in, d_out, len);
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));

  // timed run
  CUDA_CHECK(cudaEventRecord(start, 0));
  gpu_reduction_cascaded<<<grid, block>>>(d_in, d_out, len);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float execTime;
  CUDA_CHECK(cudaEventElapsedTime(&execTime, start, end));

  return execTime;
} /* benchmark_gpu_reduction_cascaded */

void log_results(const char *filename, float cpu_time_ms,
                 float gpu_time_global_ms, float gpu_time_cascaded_ms) {
  FILE *fp = fopen(filename, "a");
  if (!fp) {
    fprintf(stderr, "Error opening file %s for writing.\n", filename);
    exit(EXIT_FAILURE);
  }

  fprintf(fp, "%f,%f,%f\n", cpu_time_ms, gpu_time_global_ms,
          gpu_time_cascaded_ms);

  fclose(fp);
} /* log_results */
