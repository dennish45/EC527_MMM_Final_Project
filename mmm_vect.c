/************************************************************************/
// gcc -O1 -fopenmp test_mmm_inter_omp.c -lrt -o test_mmm_inter_omp
// OMP_NUM_THREADS=4 ./test_mmm_inter_omp

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define GIG 1.0e9
/* We do *not* use CPG (cycles per gigahertz) because when multiple
   cores are each executing with their own clock speeds, sometimes overlapping
   in time, measuring "how many cycles" a program takes does not reflect
   how much time it takes. We care about time more than about cycles. */

#define BASE  0
#define ITERS 1 //4
//#define ARRSIZE 8 //29, 49, 69, 89, 109, 129 row len = (4 + 1) x 29, array size = 145 x 145 <= L2 cache
//           252300    720300
// for bigger than L3 cache with 3 arrays, ITERS = 4 , ARRSIZE = 143, 5 x 149 = 715, 715^2 <= 6144k

#define BLOCKSIZE 4
#define INNERBLOCKSIZE 4
#define KU 4


#define OPTIONS 4
#define IDENT 0

typedef float data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int rowlen;
  data_t *data;
} matrix_rec, *matrix_ptr;


/* This define is only used if you do not set the environment variable
   OMP_NUM_THREADS as instructed above, and if OpenMP also does not
   automatically detect the hardware capabilities.

   If you have a machine with lots of cores, you may wish to test with
   more threads, but make sure you also include results for THREADS=4
   in your report. */
#define THREADS 4

void detect_threads_setting()
{
  long int i, ognt;
  char * env_ONT;
/* If you prefix your command like this:

     OMP_NUM_THREADS=10 ./test_omp

   then run this program, it will detect the setting and use it.
 */
#pragma omp parallel for
  for(i=0; i<1; i++) { ognt = omp_get_num_threads(); }
  //printf("omp's default number of threads is %d\n", ognt);
  /* If this is illegal (0 or less), default to the "#define THREADS"
     value that is defined above */
  if (ognt <= 0) {
    if (THREADS != ognt) {
      //printf("Overriding with #define THREADS value %d\n", THREADS);
      ognt = THREADS;
    }
  }
  omp_set_num_threads(ognt);
#pragma omp parallel for
  for(i=0; i<1; i++) { ognt = omp_get_num_threads(); }
  //printf("Using %d threads for OpenMP\n", ognt);
}

/************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_stamp[OPTIONS][ITERS+1];
  int clock_gettime(clockid_t clk_id, struct timespec *tp);
  matrix_ptr new_matrix(long int rowlen);
  int set_matrix_rowlen(matrix_ptr m, long int rowlen);
  long int get_matrix_rowlen(matrix_ptr m);
  int init_matrix(matrix_ptr m, long int rowlen);
  int zero_matrix(matrix_ptr m, long int rowlen);
  void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_ijk_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_kij_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void bkij(matrix_ptr A, matrix_ptr B, matrix_ptr C);
  void bbkij(matrix_ptr A, matrix_ptr B, matrix_ptr C);
  void mmm_vect(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void printMat(matrix_ptr A);
  data_t getChecksum(matrix_ptr C);
  void resetResult(matrix_ptr C);

  long int i, j, k, ognt;
  long int time_sec, time_ns;
  long int MAXSIZE = ARRSIZE; // MAXSIZE = ITERS * ARRSIZE = rowlen, array = rowlen*rowlen

  //printf("Hello World -- OpenMP Matrix Multiply\n");

  detect_threads_setting();

  // declare and initialize the matrix structure
  matrix_ptr a0 = new_matrix(MAXSIZE);
  init_matrix(a0, MAXSIZE);
  matrix_ptr b0 = new_matrix(MAXSIZE);
  init_matrix(b0, MAXSIZE);
  matrix_ptr c0 = new_matrix(MAXSIZE);
  zero_matrix(c0, MAXSIZE);

	// set values to matrix?
	set_matrix_rowlen(a0,ARRSIZE);
	set_matrix_rowlen(b0,ARRSIZE);
	set_matrix_rowlen(c0,ARRSIZE);

	clock_gettime(CLOCK_REALTIME, &time1);
	mmm_vect(a0, b0, c0);
	clock_gettime(CLOCK_REALTIME, &time2);

	//printMat(c0);
	printf("Checksum: %f\n", getChecksum(c0));
	long int timeElapsedNs = time2.tv_nsec - time1.tv_nsec;
	int timeElapsed = time2.tv_sec - time1.tv_sec;
	resetResult(c0);

  //printf("\nAll times are in seconds\n");
  printf("%4g", timeElapsed + timeElapsedNs / GIG);
} /* end main */

/**********************************************/

/* Create matrix of specified length */
matrix_ptr new_matrix(long int rowlen)
{
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->rowlen = rowlen;

  /* Allocate and declare array */
  if (rowlen > 0) {
    data_t *data = (data_t *) calloc(rowlen*rowlen, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("COULD NOT ALLOCATE %ld BYTES STORAGE \n",
                                      rowlen * rowlen * sizeof(data_t));
      exit(-1);
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set row length of matrix */
int set_matrix_rowlen(matrix_ptr m, long int rowlen)
{
  m->rowlen = rowlen;
  return 1;
}

/* Return row length of matrix */
long int get_matrix_rowlen(matrix_ptr m)
{
  return m->rowlen;
}

/* initialize matrix */
int init_matrix(matrix_ptr m, long int rowlen)
{
  long int i;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen*rowlen; i++)
      m->data[i] = (data_t)(i);
    return 1;
  }
  else return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int rowlen)
{
  long int i,j;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen*rowlen; i++)
      m->data[i] = (data_t)(IDENT);
    return 1;
  }
  else return 0;
}

data_t *get_matrix_start(matrix_ptr m)
{
  return m->data;
}

/*************************************************/

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

/*************************************************/

data_t getChecksum(matrix_ptr C) {
	int i, j;
	data_t sum;
  long int length = get_matrix_rowlen(C);
  data_t *c0 = get_matrix_start(C);
	
	for (i = 0; i < length; i++) {
		for (j = 0; j < length; j++) {
			sum += c0[i*length+j];
		}
	}
	
	return sum;
}

void resetResult(matrix_ptr C) {
	int i, j;
	data_t sum;
  long int length = get_matrix_rowlen(C);
  data_t *c0 = get_matrix_start(C);
	
	for (i = 0; i < length; i++) {
		for (j = 0; j < length; j++) {
			c0[i*length+j] = 0.0;
		}
	}
}

void printMat(matrix_ptr A) {
	int i, j;
	data_t sum;
	long int length = get_matrix_rowlen(A);
	data_t *a0 = get_matrix_start(A);
	
	for (i = 0; i < length; i++) {
		for (j = 0; j < length; j++) {
			printf("%05f\t", a0[i*length+j]);
		}
		printf("\n");
	}


}

/* MMM ijk w/ OMP 
void mmm_ijk_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int row_length = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;
  long int part_rows = row_length/4;
#pragma omp parallel shared(a0,b0,c0,row_length) private(i,j,k,sum) 
//#pragma omp parallel shared(a0,b0,c0,row_length,i,j,k) private(sum)
  {
#pragma omp for
    for (i = 0; i < row_length; i++) {
      for (j = 0; j < row_length; j++) {
        sum = IDENT;
//	#pragma omp for
        for (k = 0; k < row_length; k++)
          sum += a0[i*row_length+k] * b0[k*row_length+j];
        c0[i*row_length+j] += sum;
      }
    }
  }
}

*/

// from http://richardstartin.uk/mmm-revisited/
// currently segfaults if matrix size > 128
// accurate for sizes 64 and 128 only, apparently -- I'm guessing it would be good for larger sizes if not for the segfaults too
void mmm_vect(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
    //printf("start\n");
    long int i, j, k, ii, jj, kk;
    const long int n = get_matrix_rowlen(c);
    float *left = get_matrix_start(a);
    float *right = get_matrix_start(b);
    float *result = get_matrix_start(c);

    int column_offset, row_offset;
    const int block_width = n >= 256 ? 512 : 256;
    const int block_height = n >= 512 ? 8 : n >= 256 ? 16 : 32;

    //printf("made it here\n");

    //printf("n: %d\n", n);
    //printf("block_width: %d\n", block_width);
    //printf("block_height: %d\n", block_height);

    for (column_offset = 0; column_offset < n; column_offset += block_width) {
        for (row_offset = 0; row_offset < n; row_offset += block_height) {
            //printf("(%d, %d)\n", row_offset, column_offset);
            for (i = 0; i < n; ++i) {
                for (j = column_offset; j < column_offset + block_width && j < n; j += 64) {
                    //printf("i=%d j=%d\n", i, j);
                    //printf("%d\n", i*n+j);
                    __m256 sum1 = _mm256_loadu_ps(result + i * n + j);
                    //printf("2\n");
                    __m256 sum2 = _mm256_loadu_ps(result + i * n + j + 8);
                    //printf("3\n");
                    __m256 sum3 = _mm256_loadu_ps(result + i * n + j + 16);
                    //printf("4\n");
                    __m256 sum4 = _mm256_loadu_ps(result + i * n + j + 24);
                    //printf("5\n");
                    __m256 sum5 = _mm256_loadu_ps(result + i * n + j + 32);
                    __m256 sum6 = _mm256_loadu_ps(result + i * n + j + 40);
                    __m256 sum7 = _mm256_loadu_ps(result + i * n + j + 48);
                    __m256 sum8 = _mm256_loadu_ps(result + i * n + j + 56);
                    for (k = row_offset; k < row_offset + block_height && k < n; ++k) {
                        __m256 multiplier = _mm256_set1_ps(left[i * n + k]);
                        sum1 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j), sum1);
                        sum2 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 8), sum2);
                        sum3 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 16), sum3);
                        sum4 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 24), sum4);
                        sum5 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 32), sum5);
                        sum6 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 40), sum6);
                        sum7 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 48), sum7);
                        sum8 = _mm256_fmadd_ps(multiplier, _mm256_load_ps(right + k * n + j + 56), sum8);
                    }
                    _mm256_storeu_ps(result + i * n + j, sum1);
                    _mm256_storeu_ps(result + i * n + j + 8, sum2);
                    _mm256_storeu_ps(result + i * n + j + 16, sum3);
                    _mm256_storeu_ps(result + i * n + j + 24, sum4);
                    _mm256_storeu_ps(result + i * n + j + 32, sum5);
                    _mm256_storeu_ps(result + i * n + j + 40, sum6);
                    _mm256_storeu_ps(result + i * n + j + 48, sum7);
                    _mm256_storeu_ps(result + i * n + j + 56, sum8);
                }
            }

            //printf("(%d, %d)\n", row_offset, column_offset);
        }
    }

}
