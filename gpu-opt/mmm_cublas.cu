// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// from https://github.com/sol-prog/cuda_cublas_curand_thrust
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>

#ifndef ARR_SIZE
#define ARR_SIZE 8256
#endif

float getChecksum(float* C, int length) {
	int i, j;
	float sum = 0;	

        for (i = 0; i < length; i++) {
                for (j = 0; j < length; j++) {
                        sum += C[i*length+j];
                }
        }

        return sum;
}

void initializeArrayOrdered2D(float *arr, int len) {
  long int i;


  for (i = 0; i < len*len; i++) {
    arr[i] = (float)i;
  }
}

/*
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}
*/

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// GPU Timing variables
	cudaEvent_t startOuter, stopOuter;
	float elapsed_gpu_outer;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = ARR_SIZE;
	
	float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

        initializeArrayOrdered2D(h_A, nr_rows_A);
        initializeArrayOrdered2D(h_B, nr_rows_B);

	cudaEventCreate(&startOuter);
	cudaEventCreate(&stopOuter);
	// Record event on the default stream
	cudaEventRecord(startOuter, 0);

	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

	// If you already have useful values in A and B you can copy them in GPU:
	cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);

	// Fill the arrays A and B on GPU with random numbers
	//GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	//GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);


	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
	//std::cout << "A =" << std::endl;
	//print_matrix(h_A, nr_rows_A, nr_cols_A);
	//std::cout << "B =" << std::endl;
	//print_matrix(h_B, nr_rows_B, nr_cols_B);

	// Multiply A and B on GPU
	gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

	// Copy (and print) the result on host memory
	cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
	//std::cout << "C =" << std::endl;
	//print_matrix(h_C, nr_rows_C, nr_cols_C);

	// Stop and destroy the timer
	cudaEventRecord(stopOuter,0);
	cudaEventSynchronize(stopOuter);
	cudaEventElapsedTime(&elapsed_gpu_outer, startOuter, stopOuter);
	printf("%f", elapsed_gpu_outer/1000.0);
	cudaEventDestroy(startOuter);
	cudaEventDestroy(stopOuter);

	float checksum = getChecksum(h_C, ARR_SIZE);

	fprintf(stderr, "Checksum: %f\n", checksum);


	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
