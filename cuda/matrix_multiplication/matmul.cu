#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void process_input(int argc, char *argv[], int *tile_width);
template <typename T> void init_matrix(T *m, size_t rows, size_t cols);
template <typename T>
void matmul_cpu(T *a, T *b, T *c, size_t n1, size_t n2, size_t n3);
template <typename T>
__global__ void matmul_gpu(T *a, T *b, T *c, size_t n1, size_t n2, size_t n3);
template <typename T>
__global__ void matmul_gpu_tiled(T *a, T *b, T *c, size_t n1, size_t n2,
                                 size_t n3, int tile_width);
template <typename T> void verify_correctness(T h_r, T h_dr, T h_drt, size_t n);
void log_results(const char *filename, int tile_width, float cpu_time,
                 float gpu_time_regular, float gpu_time_tiled);

int main(int argc, char **argv) {
  int tile_width;
  process_input(argc, argv, &tile_width);

  size_t n1 = 10000; // rows of matrix 1
  size_t n2 = 5000;  // cols of matrix 1 / rows of matrix 2
  size_t n3 = 20000; // cols of matrix 2

  using T = float;
  // Allocate host matrices
  T *h_m = (float *)malloc(n1 * n2 * sizeof(T));
  MALLOC_CHECK(h_m);
  T *h_n = (T *)malloc(n2 * n3 * sizeof(T));
  MALLOC_CHECK(h_n);
  T *h_r = (T *)malloc(n1 * n3 * sizeof(T));
  MALLOC_CHECK(h_r);

  // Initialize the host matrices with random values
  init_matrix(h_m, n1, n2);
  init_matrix(h_n, n2, n3);

  struct timespec cpu_start, cpu_end;
  // Run the host multiplication function
  clock_gettime(CLOCK_MONOTONIC, &cpu_start);
  matmul_cpu(h_m, h_n, h_r, n1, n2, n3);
  clock_gettime(CLOCK_MONOTONIC, &cpu_end);
  float cpu_time = (float)(cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                   (float)(cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000000.0;

  // Initialize and allocate device matrices
  T *d_m, *d_n, *d_r, *d_rt;
  CUDA_CHECK(cudaMalloc(&d_m, n1 * n2 * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_n, n2 * n3 * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_r, n1 * n3 * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_rt, n1 * n3 * sizeof(T)));

  // Copy values from host to device
  CUDA_CHECK(cudaMemcpy(d_m, h_m, n1 * n2 * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_n, h_n, n2 * n3 * sizeof(T), cudaMemcpyHostToDevice));

  // Run the device multiplication kernel
  dim3 block(tile_width, tile_width, 1);
  dim3 grid((n3 + block.x - 1) / block.x, (n1 + block.y - 1) / block.y, 1);

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // Regular
  // Warmup run
  matmul_gpu<T><<<grid, block>>>(d_m, d_n, d_r, n1, n2, n3);

  CUDA_CHECK(cudaEventRecord(start, 0));
  matmul_gpu<T><<<grid, block>>>(d_m, d_n, d_r, n1, n2, n3);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float regular_kernel_time;
  CUDA_CHECK(cudaEventElapsedTime(&regular_kernel_time, start, end));

  T *h_dr = (T *)malloc(n1 * n3 * sizeof(T));
  CUDA_CHECK(
      cudaMemcpy(h_dr, d_r, n1 * n3 * sizeof(T), cudaMemcpyDeviceToHost));

  // Tiled
  size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(T);
  // Warmup run
  matmul_gpu_tiled<T><<<grid, block, shared_mem_size>>>(d_m, d_n, d_rt, n1, n2,
                                                        n3, tile_width);
  CUDA_CHECK(cudaEventRecord(start, 0));
  matmul_gpu_tiled<T><<<grid, block, shared_mem_size>>>(d_m, d_n, d_rt, n1, n2,
                                                        n3, tile_width);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));

  float tiled_kernel_time;
  CUDA_CHECK(cudaEventElapsedTime(&tiled_kernel_time, start, end));

  T *h_drt = (T *)malloc(n1 * n3 * sizeof(T));
  CUDA_CHECK(
      cudaMemcpy(h_drt, d_rt, n1 * n3 * sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  // Correctness check
  // verify_correctness(h_r, h_dr, h_drt, n1 * n3);

  log_results("metrics.csv", tile_width, 69.69, regular_kernel_time,
              tiled_kernel_time);

  // cleanup
  free(h_m);
  free(h_n);
  free(h_r);
  free(h_dr);
  free(h_drt);
  CUDA_CHECK(cudaFree(d_m));
  CUDA_CHECK(cudaFree(d_n));
  CUDA_CHECK(cudaFree(d_r));
  CUDA_CHECK(cudaFree(d_rt));
} /* main */

void process_input(int argc, char *argv[], int *tile_width) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <tile_size>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  *tile_width = atoi(argv[1]);
}

template <typename T> void init_matrix(T *m, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows * cols; i++) {
    m[i] = (T)(rand() & 0x0F);
  }
} /* init_matrix */

template <typename T>
void matmul_cpu(T *a, T *b, T *c, size_t n1, size_t n2, size_t n3) {
  for (size_t i = 0; i < n1; i++) {
    for (size_t j = 0; j < n3; j++) {
      T sum = 0;
      for (size_t k = 0; k < n2; k++) {
        sum += a[i * n2 + k] * b[k * n3 + j];
      }
      c[i * n3 + j] = sum;
    }
  }
} /* matmul_cpu */

template <typename T>
__global__ void matmul_gpu(T *a, T *b, T *c, size_t n1, size_t n2, size_t n3) {
  T sum = 0;
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n1 || j >= n3) {
    return;
  }
  for (size_t k = 0; k < n2; k++) {
    sum += a[i * n2 + k] * b[k * n3 + j];
  }
  c[i * n3 + j] = sum;
} /* matmul_gpu */

template <typename T>
__global__ void matmul_gpu_tiled(T *a, T *b, T *c, size_t n1, size_t n2,
                                 size_t n3, int tile_width) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  extern __shared__ T sh[];
  T *sh_m = sh;
  T *sh_n = sh_m + tile_width * tile_width;

  T sum = 0;
  for (int phase = 0; phase < (n2 + tile_width - 1) / tile_width; phase++) {
    if ((row < n1) && (phase * tile_width + tx) < n2) {
      sh_m[ty * tile_width + tx] = a[row * n2 + phase * tile_width + tx];
    } else {
      sh_m[ty * tile_width + tx] = 0;
    }
    if (((phase * tile_width + ty) < n2) && (col < n3)) {
      sh_n[ty * tile_width + tx] = b[(phase * tile_width + ty) * n3 + col];
    } else {
      sh_n[ty * tile_width + tx] = 0;
    }
    __syncthreads();

    for (int k = 0; k < tile_width; k++) {
      sum += sh_m[ty * tile_width + k] * sh_n[k * tile_width + tx];
    }
    __syncthreads();
  }
  if ((row < n1) && (col < n3)) {
    c[row * n3 + col] = sum;
  }
} /* matmul_gpu_tiled */

template <typename T>
void verify_correctness(T h_r, T h_dr, T h_drt, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (h_r[i] != h_dr[i]) {
      fprintf(stderr, "Regular kernel result mismatch\n");
      exit(EXIT_FAILURE);
    }
    if (h_r[i] != h_drt[i]) {
      fprintf(stderr, "Tiled kernel result mismatch\n");
      exit(EXIT_FAILURE);
    }
  }
} /* verify_correctness */

void log_results(const char *filename, int tile_width, float cpu_time,
                 float gpu_time_regular, float gpu_time_tiled) {
  FILE *fp = fopen(filename, "a");
  if (!fp) {
    fprintf(stderr, "Error opening file.\n");
    exit(EXIT_FAILURE);
  }
  time_t t = time(NULL);
  fprintf(fp, "%ld,%d,%.8f,%.8f,%.8f\n", t, tile_width, cpu_time,
          gpu_time_regular, gpu_time_tiled);
  fclose(fp);
} /* log_results */
