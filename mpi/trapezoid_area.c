/*
 * Each process computes a local trapezoidal estimate of the integral of f(x)
 * over its subinterval, and results are reduced via MPI_Reduce.
 */

#include <mpi.h>
#include <stdio.h>

void build_mpi_type(double *a_p, double *b_p, int *n_p,
                    MPI_Datatype *input_mpi_t);
void get_input(int rank, int comm_sz, double *a_p, double *b_p, int *n_p);
double f(double x);
double area(double a, double b, int n, double h);

int main(int argc, char *argv[]) {
  int n;
  double a, b;
  double local_integral, total_integral, local_a, local_b;

  int aggregator_rank = 0;
  int my_rank, comm_sz;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  get_input(my_rank, comm_sz, &a, &b, &n);

  double h = (b - a) / n;
  int base_n = n / comm_sz;
  int remainder_n = n % comm_sz;
  int local_n = my_rank < remainder_n ? base_n + 1 : base_n;

  int start_index =
      my_rank * base_n + (my_rank < remainder_n ? my_rank : remainder_n);
  local_a = a + start_index * h;
  local_b = local_a + local_n * h;

  local_integral = area(local_a, local_b, local_n, h);

  MPI_Reduce(&local_integral, &total_integral, 1, MPI_DOUBLE, MPI_SUM,
             aggregator_rank, MPI_COMM_WORLD);

  if (my_rank == 0) {
    printf("With n = %d trapezoids, estimated integral = %.10f\n", n,
           total_integral);
  }

  MPI_Finalize();

  return 0;
} /* main */

void build_mpi_type(double *a_p, double *b_p, int *n_p,
                    MPI_Datatype *input_mpi_t) {
  int array_of_blocklenghts[3] = {1, 1, 1};

  MPI_Aint a_addr, b_addr, n_addr;
  MPI_Get_address(a_p, &a_addr);
  MPI_Get_address(b_p, &b_addr);
  MPI_Get_address(n_p, &n_addr);
  MPI_Aint array_of_displacements[3] = {0};
  array_of_displacements[1] = b_addr - a_addr;
  array_of_displacements[2] = n_addr - a_addr;

  MPI_Datatype array_of_types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};

  MPI_Type_create_struct(3, array_of_blocklenghts, array_of_displacements,
                         array_of_types, input_mpi_t);
  MPI_Type_commit(input_mpi_t);
} /* build_mpi_type */

void get_input(int rank, int comm_sz, double *a_p, double *b_p, int *n_p) {
  int input_source = 0;
  if (rank == 0) {
    printf("Enter a, b, and n:\n");
    scanf("%lf %lf %d", a_p, b_p, n_p);
  }

  MPI_Datatype input_mpi_t;
  build_mpi_type(a_p, b_p, n_p, &input_mpi_t);

  MPI_Bcast(a_p, 1, input_mpi_t, input_source, MPI_COMM_WORLD);

  MPI_Type_free(&input_mpi_t);
} /* get_input */

double f(double x) { return x; } /* f */

double area(double a, double b, int n, double h) {
  double x;
  double estimate = (f(a) + f(b)) / 2.0;
  for (int i = 1; i < n; i++) {
    x = a + i * h;
    estimate += f(x);
  }
  estimate *= h;
  return estimate;
} /* area */
