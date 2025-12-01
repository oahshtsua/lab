#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 32
#define TILE_PADDING 1

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

template <typename T> void init_matrix(T *m, size_t rows, size_t cols);
template <typename T>
void matrix_transpose_cpu(T *in, T *out, size_t rows_in, size_t cols_in);
template <typename T>
__global__ void matrix_copy_gpu(T *in, T *out, size_t rows_in, size_t cols_in);
template <typename T>
__global__ void matrix_transpose_gpu_naive(T *in, T *out, size_t rows_in,
                                           size_t cols_in);
template <typename T>
__global__ void matrix_transpose_gpu_tiled(T *in, T *out, size_t rows_in,
                                           size_t cols_in);
template <typename T>
float benchmark_matrix_transpose_cpu(T *h_in, T *h_out, size_t rows,
                                     size_t cols);
template <typename T>
float benchmark_matrix_copy(dim3 block, dim3 grid, T *d_in, T *d_out,
                            size_t rows, size_t cols);
template <typename T>
float benchmark_matrix_transpose_gpu_naive(dim3 block, dim3 grid, T *d_in,
                                           T *d_out, size_t rows, size_t cols);
template <typename T>
float benchmark_matrix_transpose_gpu_tiled(dim3 block, dim3 grid, T *d_in,
                                           T *d_out, size_t rows, size_t cols);
template <typename T> void verify_correctness(T *h_r, T *h_dr, size_t n);
void log_results(const char *filename, float cpu_time, float gpu_copy_time,
                 float gpu_transpose_naive, float matrix_transpose_gpu_tiled);

int main() {
  size_t rows = 10000, cols = 5000;

  using T = float;
  T *h_m = (T *)malloc(rows * cols * sizeof(T));
  T *h_n = (T *)malloc(rows * cols * sizeof(T));
  T *h_dn;

  init_matrix(h_m, rows, cols);

  struct timespec cpu_start, cpu_end;
  clock_gettime(CLOCK_MONOTONIC, &cpu_start);
  matrix_transpose_cpu(h_m, h_n, rows, cols);
  clock_gettime(CLOCK_MONOTONIC, &cpu_end);

  float bandwidth_cpu = benchmark_matrix_transpose_cpu(h_m, h_n, rows, cols);

  // Initialize and allocate device matrices
  T *d_m, *d_n;
  CUDA_CHECK(cudaMalloc(&d_m, rows * cols * sizeof(T)));

  // Copy values from host to device
  CUDA_CHECK(
      cudaMemcpy(d_m, h_m, rows * cols * sizeof(T), cudaMemcpyHostToDevice));

  // Device configuration
  dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, 1);

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  /* Copy Kernel */
  CUDA_CHECK(cudaMalloc(&d_n, rows * cols * sizeof(T)));
  // Warmup run
  matrix_copy_gpu<T><<<grid, block>>>(d_m, d_n, rows, cols);

  float bandwidth_gpu_copy =
      benchmark_matrix_copy(block, grid, d_m, d_n, rows, cols);

  CUDA_CHECK(cudaFree(d_n));
  /* Copy Kernel */

  /* Naive Transpose */
  CUDA_CHECK(cudaMalloc(&d_n, rows * cols * sizeof(T)));
  // Warmup run
  matrix_transpose_gpu_naive<T><<<grid, block>>>(d_m, d_n, rows, cols);
  // Copy result from device to host for verification
  h_dn = (float *)malloc(rows * cols * sizeof(T));
  MALLOC_CHECK(h_dn);
  CUDA_CHECK(
      cudaMemcpy(h_dn, d_n, rows * cols * sizeof(T), cudaMemcpyDeviceToHost));
  verify_correctness(h_n, h_dn, rows * cols);
  free(h_dn);

  float bandwidth_gpu_naive =
      benchmark_matrix_transpose_gpu_naive(block, grid, d_m, d_n, rows, cols);

  CUDA_CHECK(cudaFree(d_n));
  /* Naive Transpose */

  /* Tiled transpose */
  CUDA_CHECK(cudaMalloc(&d_n, rows * cols * sizeof(T)));
  // Warmup run
  matrix_transpose_gpu_tiled<T><<<grid, block>>>(d_m, d_n, rows, cols);
  // Copy result from device to host for verification
  h_dn = (float *)malloc(rows * cols * sizeof(T));
  MALLOC_CHECK(h_dn);
  CUDA_CHECK(
      cudaMemcpy(h_dn, d_n, rows * cols * sizeof(T), cudaMemcpyDeviceToHost));
  verify_correctness(h_n, h_dn, rows * cols);
  free(h_dn);

  float bandwidth_gpu_tiled =
      benchmark_matrix_transpose_gpu_tiled(block, grid, d_m, d_n, rows, cols);

  CUDA_CHECK(cudaFree(d_n));
  /* Tiled Transpose */

  CUDA_CHECK(cudaFree(d_m));
  free(h_m);
  free(h_n);

  log_results("metrics.csv", bandwidth_cpu, bandwidth_gpu_copy,
              bandwidth_gpu_naive, bandwidth_gpu_tiled);
  return 0;
}

template <typename T> void init_matrix(T *m, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows * cols; i++) {
    m[i] = (T)rand() / RAND_MAX;
  }
} /* init_matrix */

template <typename T>
void matrix_transpose_cpu(T *in, T *out, size_t rows_in, size_t cols_in) {
  for (size_t i = 0; i < rows_in; i++) {
    for (size_t j = 0; j < cols_in; j++) {
      out[j * rows_in + i] = in[i * cols_in + j];
    }
  }
} /* matrix_transpose_cpu */

template <typename T>
__global__ void matrix_copy_gpu(T *in, T *out, size_t rows_in, size_t cols_in) {
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < rows_in && j < cols_in) {
    size_t idx = i * cols_in + j;
    out[idx] = in[idx];
  }
} /* matrix_copy_gpu */

template <typename T>
__global__ void matrix_transpose_gpu_naive(T *in, T *out, size_t rows_in,
                                           size_t cols_in) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < rows_in && j < cols_in) {
    out[j * rows_in + i] = in[i * cols_in + j];
  }
} /* matrix_transpose_gpu_naive */

template <typename T>
__global__ void matrix_transpose_gpu_tiled(T *in, T *out, size_t rows_in,
                                           size_t cols_in) {
  __shared__ T tile[TILE_WIDTH][TILE_WIDTH + TILE_PADDING];
  int i_row = blockIdx.y * blockDim.y + threadIdx.y;
  int i_col = blockIdx.x * blockDim.x + threadIdx.x;

  if (i_row < rows_in && i_col < cols_in) {
    tile[threadIdx.y][threadIdx.x] = in[i_row * cols_in + i_col];
  }

  __syncthreads();

  int o_row = blockIdx.x * TILE_WIDTH + threadIdx.y;
  int o_col = blockIdx.y * TILE_WIDTH + threadIdx.x;

  if (o_row < cols_in && o_col < rows_in) {
    out[o_row * rows_in + o_col] = tile[threadIdx.x][threadIdx.y];
  }

} /* matrix_transpose_gpu_tiled */

template <typename T> void verify_correctness(T *h_r, T *h_dr, size_t n) {
  const double error_margin = 1e-6;
  for (size_t i = 0; i < n; i++) {
    if (fabs(h_r[i] - h_dr[i]) > error_margin) {
      fprintf(stderr, "Results dont match!\n");
      exit(EXIT_FAILURE);
    }
  }
} /* verify_correctness */

template <typename T>
float benchmark_matrix_transpose_cpu(T *h_in, T *h_out, size_t rows,
                                     size_t cols) {
  struct timespec cpu_start, cpu_end;
  clock_gettime(CLOCK_MONOTONIC, &cpu_start);
  matrix_transpose_cpu(h_in, h_out, rows, cols);
  clock_gettime(CLOCK_MONOTONIC, &cpu_end);
  float exec_time = (float)(cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                    (float)(cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000000.0;
  float bandwidth = 2 * (sizeof(T) * rows * cols) / (exec_time / 1000.0) / 1e9;
  return bandwidth;
} /* benchmark_matrix_transpose_cpu */

template <typename T>
float benchmark_matrix_copy(dim3 block, dim3 grid, T *d_in, T *d_out,
                            size_t rows, size_t cols) {
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start, 0));
  matrix_copy_gpu<T><<<grid, block>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float exec_time;
  CUDA_CHECK(cudaEventElapsedTime(&exec_time, start, end));
  float bandwidth = 2 * (sizeof(T) * rows * cols) / (exec_time / 1000.0) / 1e9;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  return bandwidth;
} /* benchmark_matrix_copy */;

template <typename T>
float benchmark_matrix_transpose_gpu_naive(dim3 block, dim3 grid, T *d_in,
                                           T *d_out, size_t rows, size_t cols) {
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start, 0));
  matrix_transpose_gpu_naive<T><<<grid, block>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float exec_time;
  CUDA_CHECK(cudaEventElapsedTime(&exec_time, start, end));
  float bandwidth = 2 * (sizeof(T) * rows * cols) / (exec_time / 1000.0) / 1e9;
  return bandwidth;
} /* benchmark_matrix_transpose_gpu_naive */

template <typename T>
float benchmark_matrix_transpose_gpu_tiled(dim3 block, dim3 grid, T *d_in,
                                           T *d_out, size_t rows, size_t cols) {
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start, 0));
  matrix_transpose_gpu_tiled<T><<<grid, block>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float exec_time;
  CUDA_CHECK(cudaEventElapsedTime(&exec_time, start, end));
  float bandwidth = 2 * (sizeof(T) * rows * cols) / (exec_time / 1000.0) / 1e9;
  return bandwidth;
} /* benchmark_matrix_transpose_gpu_tiled */

void log_results(const char *filename, float cpu_bandwidth,
                 float gpu_copy_bandwidth, float gpu_transpose_naive_bandwidth,
                 float matrix_transpose_gpu_tiled_bandwidth) {
  FILE *fp = fopen(filename, "a");
  if (!fp) {
    fprintf(stderr, "Error opening file.\n");
    exit(EXIT_FAILURE);
  }
  time_t t = time(NULL);
  fprintf(fp, "%.6f,%.6f,%.6f,%.6f\n", cpu_bandwidth, gpu_copy_bandwidth,
          gpu_transpose_naive_bandwidth, matrix_transpose_gpu_tiled_bandwidth);
  fclose(fp);
} /* log_results */
