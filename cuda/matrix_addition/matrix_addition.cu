#include <cuda_runtime.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MALLOC_CHECK(val) malloc_check((val), __FILE__, __LINE__)
void malloc_check(void *ptr, const char *file, const int line) {
  if (!ptr) {
    fprintf(stderr, "Error allocating memory: %s:%d\n", file, line);
    exit(EXIT_FAILURE);
  }
} /* malloc_check */

#define CUDA_CHECK(val) cuda_check((val), __FILE__, __LINE__)
void cuda_check(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Cuda runtime error: %s:%d\n", file, line);
    exit(EXIT_FAILURE);
  }
} /* cuda_check */

double matrix_add_cpu(float *h_m, float *h_n, float *h_s, size_t rows,
                      size_t cols);
double matrix_add_gpu(float *h_m, float *h_n, float *h_s, size_t rows,
                      size_t cols, size_t block_xdim, size_t block_ydim);
void process_input(int argc, char *argv[], size_t *rows, size_t *cols,
                   size_t *block_xdim, size_t *block_ydim);
float sum_abs_error(float *h_m, float *h_n, size_t rows, size_t cols);
void matrix_print(float *h_m, size_t rows, size_t cols);
void log_results(const char *filename, size_t rows, size_t cols,
                 size_t block_xdim, size_t block_ydim, float cpu_time,
                 float gpu_time, float error);

__global__ void Matrix_Add(float *d_m, float *d_n, float *d_s, size_t rows,
                           size_t cols) {
  size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t ty = blockDim.y * blockIdx.y + threadIdx.y;

  size_t idx = ty * cols + tx;
  if ((tx < cols) && (ty < rows)) {
    d_s[idx] = d_m[idx] + d_n[idx];
  }
} /* Matrix_Add */

int main(int argc, char *argv[]) {
  srand48(time(NULL));
  // Process user input for matrix dimensions
  size_t rows, cols, block_xdim, block_ydim;
  process_input(argc, argv, &rows, &cols, &block_xdim, &block_ydim);

  // Allocate memory for the matrices
  float *h_m = (float *)malloc(rows * cols * sizeof(float));
  MALLOC_CHECK(h_m);
  float *h_n = (float *)malloc(rows * cols * sizeof(float));
  MALLOC_CHECK(h_n);
  float *h_s = (float *)malloc(rows * cols * sizeof(float));
  MALLOC_CHECK(h_s);

  // Initialize input matrices with random values
  for (size_t i = 0; i < rows * cols; i++) {
    h_m[i] = (float)drand48();
    h_n[i] = (float)drand48();
  }

  // Matrix addition CPU
  double cpu_time = matrix_add_cpu(h_m, h_n, h_s, rows, cols);

  // Matrix addition GPU
  float *h_ds = (float *)malloc(rows * cols * sizeof(float));
  MALLOC_CHECK(h_ds);
  double gpu_time =
      matrix_add_gpu(h_m, h_n, h_ds, rows, cols, block_xdim, block_ydim);

  // Correctness check
  float error = sum_abs_error(h_s, h_ds, rows, cols);
  printf("%e\n", error);

  // Cleanup
  free(h_m);
  free(h_n);
  free(h_s);
  free(h_ds);

  // Log performance
  log_results("metrics.csv", rows, cols, block_xdim, block_ydim, cpu_time,
              gpu_time, error);
  return 0;
} /* main */

double matrix_add_cpu(float *h_m, float *h_n, float *h_s, size_t rows,
                      size_t cols) {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (size_t i = 0; i < rows * cols; i++) {
    h_s[i] = h_m[i] + h_n[i];
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  float cpu_time = (float)(end.tv_sec - start.tv_sec) * 1000.0 +
                   (float)(end.tv_nsec - start.tv_nsec) / 1000000.0;
  return cpu_time;
} /* matrix_add_cpu */

double matrix_add_gpu(float *h_m, float *h_n, float *h_s, size_t rows,
                      size_t cols, size_t block_xdim, size_t block_ydim) {

  dim3 block(block_xdim, block_ydim);
  dim3 grid((cols + block_xdim - 1) / block_xdim,
            (rows + block_ydim - 1) / block_ydim);

  float *d_m, *d_n, *d_s;

  // Allocate memory on device for the matrices
  CUDA_CHECK(cudaMalloc((void **)&d_m, rows * cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_n, rows * cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_s, rows * cols * sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_m, h_m, rows * cols * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_n, h_n, rows * cols * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Setup profiling
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start, 0));
  // Invoke the kernel
  Matrix_Add<<<grid, block>>>(d_m, d_n, d_s, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(
      end)); // Waits for the kernel to complete before recording the event
  float gpu_time;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, end));

  // Teardown profiling
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  // Copy result back to the host
  CUDA_CHECK(cudaMemcpy(h_s, d_s, rows * cols * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_m));
  CUDA_CHECK(cudaFree(d_n));
  CUDA_CHECK(cudaFree(d_s));

  return gpu_time;
} /* matrix_add_gpu */

void process_input(int argc, char *argv[], size_t *rows, size_t *cols,
                   size_t *block_xdim, size_t *block_ydim) {
  int option;
  char *r_val = NULL, *c_val = NULL, *x_val = NULL, *y_val = NULL;

  while ((option = getopt(argc, argv, "r:c:x:y:")) != -1) {
    switch (option) {
    case 'r':
      r_val = optarg;
      break;
    case 'c':
      c_val = optarg;
      break;
    case 'x':
      x_val = optarg;
      break;
    case 'y':
      y_val = optarg;
      break;
    case '?':
      fprintf(
          stderr,
          "Usage: %s -r <matRows> -c <matCols> -x <blockDimX> -y <blockDimY>\n",
          argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if (!r_val || !c_val || !x_val || !y_val) {
    fprintf(stderr, "All options -r -c -x -y are required.\n");
    fprintf(
        stderr,
        "Usage: %s -r <matRows> -c <matCols> -x <blockDimX> -y <blockDimY>\n",
        argv[0]);
    exit(EXIT_FAILURE);
  }

  *rows = atol(r_val);
  *cols = atol(c_val);
  *block_xdim = atol(x_val);
  *block_ydim = atol(y_val);
} /* process_input */

float sum_abs_error(float *h_m, float *h_n, size_t rows, size_t cols) {
  float sum = 0.0f;
  for (size_t i = 0; i < rows * cols; i++) {
    sum += fabsf(h_m[i] - h_n[i]);
  }
  return sum;
} /* sum_abs_error */

void matrix_print(float *h_m, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      printf("%f ", h_m[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
} /* matrix_print */

void log_results(const char *filename, size_t rows, size_t cols,
                 size_t block_xdim, size_t block_ydim, float cpu_time,
                 float gpu_time, float error) {
  FILE *fp = fopen(filename, "a");
  if (!fp) {
    fprintf(stderr, "Error opening file.\n");
    exit(EXIT_FAILURE);
  }
  time_t t = time(NULL);
  fprintf(fp, "%ld,%ldx%ld,%ldx%ld,%.6f,%.6f,%.6f\n", t, rows, cols, block_xdim,
          block_ydim, cpu_time, gpu_time, error);
} /* log_results */
