/* A vector sum program where each thread computes one element of z = x + y. */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void get_args(int argc, char *argv[], int *n_p, int *bc_p, int *tpb_p);
void allocate_vectors(float **hx_p, float **hy_p, float **hz_p, float **dx_p,
                      float **dy_p, float **dz_p, int n);
void init_vectors(float *x_p, float *y_p, int n);
void print_result(float *x_p, float *y_p, float *z_p, int n);
void free_vectors(float *hx_p, float *hy_p, float *hz_p, float *dx_p,
                  float *dy_p, float *dz_p);

__global__ void vector_add(const float x[], const float y[], float z[],
                           const int n) {
  int my_elt = blockDim.x * blockIdx.x + threadIdx.x;

  if (my_elt < n) {
    z[my_elt] = x[my_elt] + y[my_elt];
  }
} /* vector_add */

int main(int argc, char *argv[]) {
  int n, block_count, thread_per_block;
  float *hx, *hy, *hz;
  float *dx, *dy, *dz;

  get_args(argc, argv, &n, &block_count, &thread_per_block);
  allocate_vectors(&hx, &hy, &hz, &dx, &dy, &dz, n);

  init_vectors(hx, hy, n);
  cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy, n * sizeof(float), cudaMemcpyHostToDevice);

  vector_add<<<block_count, thread_per_block>>>(dx, dy, dz, n);
  cudaMemcpy(hz, dz, n * sizeof(float), cudaMemcpyDeviceToHost);
  print_result(hx, hy, hz, n);

  free_vectors(hx, hy, hz, dx, dy, dz);
  return 0;
} /* main */

void get_args(int argc, char *argv[], int *n_p, int *bc_p, int *tpb_p) {
  if (argc != 4) {
    fprintf(stderr, "Invalid argument list.\n");
    fprintf(stderr,
            "Usage: ./program <vec_size> <block_count> <threads_per_block>\n");
    exit(EXIT_FAILURE);
  }
  *n_p = strtol(argv[1], NULL, 10);
  *bc_p = strtol(argv[2], NULL, 10);
  *tpb_p = strtol(argv[3], NULL, 10);
} /* get_args */

void allocate_vectors(float **hx_p, float **hy_p, float **hz_p, float **dx_p,
                      float **dy_p, float **dz_p, int n) {
  *hx_p = (float *)malloc(n * sizeof(float));
  *hy_p = (float *)malloc(n * sizeof(float));
  *hz_p = (float *)malloc(n * sizeof(float));

  cudaMalloc(dx_p, n * sizeof(float));
  cudaMalloc(dy_p, n * sizeof(float));
  cudaMalloc(dz_p, n * sizeof(float));
} /* allocate_vectors */

void init_vectors(float *x_p, float *y_p, int n) {
  srand(time(NULL));
  for (int i = 0; i < n; i++) {
    x_p[i] = rand() % 100;
    y_p[i] = rand() % 100;
  }
} /* init_vectors */

void print_result(float *x_p, float *y_p, float *z_p, int n) {
  for (int i = 0; i < n; i++) {
    printf("%.2f + %.2f = %.2f\n", x_p[i], y_p[i], z_p[i]);
  }
} /* print_result */

void free_vectors(float *hx_p, float *hy_p, float *hz_p, float *dx_p,
                  float *dy_p, float *dz_p) {
  cudaFree(dx_p);
  cudaFree(dy_p);
  cudaFree(dz_p);
  free(hx_p);
  free(hy_p);
  free(hz_p);
} /* free_vectors */
