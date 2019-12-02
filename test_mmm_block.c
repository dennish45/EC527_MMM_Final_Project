/*****************************************************************************/
// gcc -O1 -o test_mmm_block.out test_mmm_block.c -lrt

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define GIG 1000000000
#define CPG 3.3           // Cycles per GHz -- Adjust to your computer

#define BASE  0
#define ITERS 1 //originally 20
#define DELTA 8 //originally 113

#define BLOCKSIZE 2

#define OPTIONS 2
#define IDENT 0

typedef float data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int len;
  data_t *data;
} matrix_rec, *matrix_ptr;

/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_stamp[OPTIONS][ITERS+1];
  int clock_gettime(clockid_t clk_id, struct timespec *tp);
  matrix_ptr new_matrix(long int len);
  int set_matrix_length(matrix_ptr m, long int index);
  long int get_matrix_length(matrix_ptr m);
  int init_matrix(matrix_ptr m, long int len);
  int zero_matrix(matrix_ptr m, long int len);
  void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_jki(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void bijk(matrix_ptr A, matrix_ptr B, matrix_ptr C, int n, int bsize);
  void bbijk(matrix_ptr A, matrix_ptr B, matrix_ptr C, int n, int bsize);
  void printMat(matrix_ptr A);

  long int i, j, k;
  long int time_sec, time_ns;
  long int MAXSIZE = BASE+(ITERS+1)*DELTA;

  printf("\n Hello World -- MMM \n");

  // declare and initialize the matrix structure
  matrix_ptr a0 = new_matrix(MAXSIZE);
  init_matrix(a0, MAXSIZE);
  matrix_ptr b0 = new_matrix(MAXSIZE);
  init_matrix(b0, MAXSIZE);
  matrix_ptr c0 = new_matrix(MAXSIZE);
  zero_matrix(c0, MAXSIZE);

  OPTION = 0;
  for (i = 0; i < ITERS; i++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, i, BASE+(i+1)*DELTA);
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    //mmm_ijk(a0,b0,c0);
    bijk(a0, b0, c0, BASE+(i+1)*DELTA, BLOCKSIZE);
    printMat(c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }
  
  OPTION++;
  for (i = 0; i < ITERS; i++) {
    printf(" OPT %d, iter %ld, size %ld\n", OPTION, i, BASE+(i+1)*DELTA);
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    bbijk(a0,b0,c0, BASE+(i+1)*DELTA, BLOCKSIZE);
    printMat(c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }

  OPTION++;
  if (OPTIONS > 2) {
    for (i = 0; i < ITERS; i++) {
      printf(" OPT %d, iter %ld, size %ld\n", OPTION, i, BASE+(i+1)*DELTA);
      set_matrix_length(a0,BASE+(i+1)*DELTA);
      set_matrix_length(b0,BASE+(i+1)*DELTA);
      set_matrix_length(c0,BASE+(i+1)*DELTA);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
      mmm_jki(a0,b0,c0);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
      time_stamp[OPTION][i] = diff(time1,time2);
    }
  }

  printf("\nlength, bijk\n");
  for (i = 0; i < ITERS; i++) {
    printf("%ld, ", BASE+(i+1)*DELTA);
    for (j = 0; j < OPTIONS; j++) {
      if (j != 0) printf(", ");
      printf("%ld", (long int)((double)(CPG)*(double)
		 (GIG * time_stamp[j][i].tv_sec + time_stamp[j][i].tv_nsec)));
    }
    printf("\n");
  }
}/* end main */

/**********************************************/

/* Create matrix of specified length */
matrix_ptr new_matrix(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc(len*len, sizeof(data_t));
    if (!data) {
	  free((void *) result);
	  printf("\n COULDN'T ALLOCATE %ld BYTES STORAGE \n", result->len);
	  return NULL;  /* Couldn't allocate storage */
	}
	result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set length of matrix */
int set_matrix_length(matrix_ptr m, long int index)
{
  m->len = index;
  return 1;
}

/* Return length of matrix */
long int get_matrix_length(matrix_ptr m)
{
  return m->len;
}

/* initialize matrix */
int init_matrix(matrix_ptr m, long int len)
{
  long int i;

  if (len > 0) {
    m->len = len;
    for (i = 0; i < len*len; i++)
      m->data[i] = (data_t)(i);
    return 1;
  }
  else return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int len)
{
  long int i,j;

  if (len > 0) {
    m->len = len;
    for (i = 0; i < len*len; i++)
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

//blocking matrix multiply
void printMat(matrix_ptr A) {
	int i, j;
	data_t sum;
	long int get_matrix_length(matrix_ptr m);
	data_t *get_matrix_start(matrix_ptr m);
	long int length = get_matrix_length(A);
	data_t *a0 = get_matrix_start(A);
	
	for (i = 0; i < length; i++) {
		for (j = 0; j < length; j++) {
			printf("%05f\t", a0[i*length+j]);
		}
		printf("\n");
	}


}

//blocking matrix multiply
void bijk(matrix_ptr A, matrix_ptr B, matrix_ptr C, int n, int bsize) {
	int i, j, k, kk, jj;
	data_t sum;
	long int get_matrix_length(matrix_ptr m);
	data_t *get_matrix_start(matrix_ptr m);
	long int length = get_matrix_length(A);
	data_t *a0 = get_matrix_start(A);
	data_t *b0 = get_matrix_start(B);
	data_t *c0 = get_matrix_start(C);
	int en = bsize * (n/bsize); // Amount that fits evenly into blocks

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			c0[i*length+j] = 0.0;

	for (kk = 0; kk < en; kk += bsize) {
		for (jj = 0; jj < en; jj += bsize) {
			for (i = 0; i < n; i++) {
				for (j = jj; j < jj + bsize; j++) {
					sum = c0[i*length+j];
					for (k = kk; k < kk + bsize; k++) {
						sum += a0[i*length+k]*b0[k*length+j];
					}
					c0[i*length+j] = sum;
				}
			}
		}
	}
}

//blocking matrix multiply
void bbijk(matrix_ptr A, matrix_ptr B, matrix_ptr C, int n, int bsize) {
	int i, j, k, i0, j0, k0, i00, j00, k00;
	data_t sum;
	long int get_matrix_length(matrix_ptr m);
	data_t *get_matrix_start(matrix_ptr m);
	long int length = get_matrix_length(A);
	data_t *a0 = get_matrix_start(A);
	data_t *b0 = get_matrix_start(B);
	data_t *c0 = get_matrix_start(C);
	int en = bsize * (n/bsize); // Amount that fits evenly into blocks
	int innerbsize = bsize / 2;
	
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			c0[i*length+j] = 0.0;

	// MMM loop nest (j, i, k)
	for (i=0; i<length; i+=bsize)
		for (j=0; j<length; j+=bsize)
			for (k=0; k<length; k+=bsize)
			// mini-MMM loop nest (i0, j0, k0)
				for (i0=i; i0<(i + bsize); i0+=innerbsize)
					for (j0=j; j0<(j + bsize); j0+=innerbsize)
						for (k0=k; k0<(k + bsize); k0+=innerbsize)
						// micro-MMM loop nest (j00, i00)
							for (k00=k0; k00<=(k0 + innerbsize); k00++)
								for (j00=j0; j00<=(j0 + innerbsize); j00++)
									for (i00=i0; i00<=(i0 + innerbsize); i00++)
										c0[i00*length+j00] += a0[i00*length+k00] * b0[k00*length+j00];

}

/* mmm */
void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int get_matrix_length(matrix_ptr m);
  data_t *get_matrix_start(matrix_ptr m);
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

  for (i = 0; i < length; i++)
    for (j = 0; j < length; j++) {
      sum = IDENT;
      for (k = 0; k < length; k++)
	sum += a0[i*length+k] * b0[k*length+j];
      c0[i*length+j] += sum;
    }
}

/* mmm */
void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int get_matrix_length(matrix_ptr m);
  data_t *get_matrix_start(matrix_ptr m);
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  for (k = 0; k < length; k++)
    for (i = 0; i < length; i++) {
      r = a0[i*length+k];
      for (j = 0; j < length; j++)
	c0[i*length+j] += r*b0[k*length+j];
    }
}

/* mmm */
void mmm_jki(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int get_matrix_length(matrix_ptr m);
  data_t *get_matrix_start(matrix_ptr m);
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  for (j = 0; j < length; j++)
    for (k = 0; k < length; k++) {
      r = b0[k*length+j];
      for (i = 0; i < length; i++)
	c0[i*length+j] += a0[i*length+k]*r;
    }
}
