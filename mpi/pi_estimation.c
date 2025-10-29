/*
 * Estimating pi using Monte Carlo method.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int ROOT = 0;

void get_input(int rank, long long int *size);
double random_double(double min, double max);
long long int count_circle_hits(long long int throws);

int main(int argc, char *argv[]) {
  int my_rank, comm_sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  srand48(time(NULL) + my_rank);
  long long int total_throws;
  get_input(my_rank, &total_throws);

  long long int local_throws = total_throws / comm_sz;
  if (my_rank < (total_throws % comm_sz)) {
    local_throws += 1;
  }

  long long int local_circle_hits = count_circle_hits(local_throws);

  long long int total_circle_hits;
  MPI_Reduce(&local_circle_hits, &total_circle_hits, 1, MPI_LONG_LONG_INT,
             MPI_SUM, ROOT, MPI_COMM_WORLD);
  if (my_rank == 0) {
    double pi_estimate =
        (4.0 * (double)total_circle_hits) / ((double)total_throws);
    printf("Estimated PI: %f\n", pi_estimate);
  }

  MPI_Finalize();

  return 0;
} /* main */

void get_input(int rank, long long int *size) {
  if (rank == ROOT) {
    printf("Sample size:\n");
    scanf("%lld", size);
  }
  MPI_Bcast(size, 1, MPI_LONG_LONG_INT, ROOT, MPI_COMM_WORLD);
} /* get_input */

double random_double(double min, double max) {
  return min + drand48() * (max - min);
} /* random_double */

long long int count_circle_hits(long long int throws) {
  double x, y;
  double squared_distance;
  long long int circle_hits = 0;

  for (long long int dart = 0; dart < throws; dart++) {
    x = random_double(-1, 1);
    y = random_double(-1, 1);
    squared_distance = x * x + y * y;
    if (squared_distance <= 1) {
      circle_hits++;
    }
  }
  return circle_hits;
} /* count_circle_hits */
