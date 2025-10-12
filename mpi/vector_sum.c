/*
 * Each process computes a local element-wise sum of two vectors,
 * and results are gathered on the root process using MPI.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

void get_input(int rank, int *n, double *a_p[], double *b_p[]);
static inline void vector_sum(double local_x[], double local_y[],
                              double local_z[], int local_n);
void print_vector(double *arr, int n);

int main(int argc, char *argv[]) {
  int my_rank, comm_sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int n;
  double *a = NULL, *b = NULL;
  get_input(my_rank, &n, &a, &b);

  int base_n = n / comm_sz;
  int remainder_n = n % comm_sz;

  int *send_counts = malloc(comm_sz * sizeof(int));
  int *displacements = malloc(comm_sz * sizeof(int));

  for (int i = 0; i < comm_sz; i++) {
    send_counts[i] = base_n + (i < remainder_n ? 1 : 0);
    displacements[i] = (i == 0) ? 0 : displacements[i - 1] + send_counts[i - 1];
  }

  int local_n = send_counts[my_rank];
  double *local_a = malloc(local_n * sizeof(double));
  double *local_b = malloc(local_n * sizeof(double));
  MPI_Scatterv(a, send_counts, displacements, MPI_DOUBLE, local_a, local_n,
               MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
  MPI_Scatterv(b, send_counts, displacements, MPI_DOUBLE, local_b, local_n,
               MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  if (my_rank == ROOT) {
    free(a);
    free(b);
    a = b = NULL;
  }
  double *local_c = malloc(local_n * sizeof(double));
  vector_sum(local_a, local_b, local_c, local_n);

  double *result;
  if (my_rank == ROOT) {
    result = malloc(n * sizeof(double));
  }

  MPI_Gatherv(local_c, local_n, MPI_DOUBLE, result, send_counts, displacements,
              MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
  if (my_rank == ROOT) {
    print_vector(result, n);
    free(result);
  }

  free(send_counts);
  free(displacements);
  free(local_a);
  free(local_b);
  free(local_c);

  MPI_Finalize();
  return 0;
} /* main */

void get_input(int rank, int *n, double *a_p[], double *b_p[]) {
  if (rank == 0) {
    printf("Enter the size of the vectors:\n");
    scanf("%d", n);

    *a_p = malloc(*n * sizeof(double));
    printf("Input elements of first vector:\n");
    for (int i = 0; i < *n; i++) {
      scanf("%lf", &(*a_p)[i]);
    }
    *b_p = malloc(*n * sizeof(double));
    printf("Input elements of second vector:\n");
    for (int i = 0; i < *n; i++) {
      scanf("%lf", &(*b_p)[i]);
    }
  }
  MPI_Bcast(n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
} /* get_input */

void print_vector(double *arr, int n) {
  for (int i = 0; i < n; i++) {
    printf("%.2lf ", arr[i]);
  }
  printf("\n");
} /* print_vector */

static inline void vector_sum(double local_x[], double local_y[],
                              double local_z[], int local_n) {
  for (int local_i = 0; local_i < local_n; local_i++) {
    local_z[local_i] = local_x[local_i] + local_y[local_i];
  }
} /* vector_sum */
