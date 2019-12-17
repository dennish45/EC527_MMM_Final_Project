/*
   Minimal CUDA program, intended just to test ability
   to compile and run a CUDA program

     nvcc cuda_test.cu -o cuda_test

   You need to follow instructions provided elsewhere, such as in the
   "SCC-for-EC527" slides, both of the SCC_Cheatsheet PDFs, and
   SCC_Getting_Started PDFs, to get onto the system where you can
   compile and run this.

   To understand the program, of course you should read the lecture notes
   (slides) that have "GPU" in the name.
*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include "cuPrintf.cu"

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

//#define NUM_THREADS_PER_BLOCK   256

#ifndef NUM_BLOCKS
#define NUM_BLOCKS         16
#endif

#define PRINT_TIME         1
#define SM_ARR_LEN        50000
#define TOL            1e-6

#ifndef ARR_SIZE
#define ARR_SIZE 192
#endif
//#define ITERS 2000

#define IMUL(a, b) __mul24(a, b)

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

void printMat(float* A, int length) {
        int i, j;
        //data_t sum;
        //long int length = get_matrix_rowlen(A);
        //data_t *a0 = get_matrix_start(A);

        for (i = 0; i < length; i++) {
                for (j = 0; j < length; j++) {
                        fprintf(stderr, "%05f\t", A[i*length+j]);
                }
                fprintf(stderr, "\n");
        }
}

// Matrix multiplication on the (CPU)
//   host in double precision
void MatrixMulOnHost(float* M, float* N, float* P, int Width) {   
  for (int i = 0; i < Width; ++i)
    for (int j = 0; j < Width; ++j) {
      float sum = 0;
      for (int k = 0; k < Width; ++k) {
        float a = M[i * Width + k];
        float b = N[k * Width + j];
        sum += a * b;
      }
      P[i * Width + j] = sum;
    }
}

//blocking matrix multiply
void MatrixMulOnHostBlocked(float* A, float* B, float* C, int Width) {
  int i, j, k, kk, jj;
  float sum;
  int bsize = 8;
  int en = bsize * (Width/bsize); // Amount that fits evenly into blocks
  

  for (kk = 0; kk < en; kk += bsize) {
    for (jj = 0; jj < en; jj += bsize) {
      for (i = 0; i < Width; i++) {
        for (j = jj; j < jj + bsize; j++) {
          sum = C[i*Width+j];
          for (k = kk; k < kk + bsize; k++) {
            sum += A[i*Width+k]*B[k*Width+j];
          }
          C[i*Width+j] = sum;
        }
      }
    }
  }
}


void initializeArrayRand2D(float *arr, int len, int seed);
void initializeArrayOrdered2D(float *arr, int len);


// Matrix multiplication kernel
//   per thread code
__global__ void MatrixMulKernelGlobal(float* Md ,   float* Nd , float* Pd, int Width) {
  //   Pvalueis used to store the element of the
  // matrix that is computed by the thread
  //cuPrintf("%f\n", Md[threadIdx.y*Width+threadIdx.x]);
  
  int Row = blockIdx.y*(ARR_SIZE/NUM_BLOCKS) + threadIdx.y;
  int Col = blockIdx.x*(ARR_SIZE/NUM_BLOCKS) + threadIdx.x;

  float Pvalue = 0;

  for (int k = 0; k < Width; ++k) {
//    float Melement = Md [threadIdx.y*Width+k];
//    float Nelement = Nd [k*Width+threadIdx.x];
    Pvalue += Md[Row*Width+k] * Nd[k*Width+Col];
  }
 
  //cuPrintf("%f\n", Pvalue);
  
  Pd[Row*Width+Col] = Pvalue;
}

__global__ void MatrixMulKernelShared(float* Md, float* Nd, float* Pd, int Width) {
  
  const int tile_width = ARR_SIZE/NUM_BLOCKS;

  __shared__ float Mds[tile_width][tile_width];  // Shared memory
  __shared__ float Nds[tile_width][tile_width];  //   declarations

  int bx = blockIdx.x;  int by = blockIdx.y;    // ID thread
  int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the Pd element to work on
	int Row = by * tile_width + ty;
	int Col = bx * tile_width + tx;
	float Pvalue = 0; // REGISTER!

	// Loop over the Md and Nd tiles required to compute the Pd element
	for (int m = 0; m < Width/tile_width; ++m) {
		// Collaborative loading of Md and Nd tiles into shared memory
		Mds[ty][tx] = Md[Row*Width + (m*tile_width + tx)];
		Nds[ty][tx] = Nd[Col + (m*tile_width + ty)*Width];
		
		__syncthreads();
		
		for (int k = 0; k < tile_width; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];
		
		__syncthreads();
	}
	Pd[Row*Width+Col] = Pvalue;
}

//from https://github.com/tpn/cuda-samples/blob/master/v8.0/0_Simple/matrixMul_nvrtc/matrixMul_kernel.cu
__global__ void matrixMulCUDA(float *C, float *A, float *B, int width)
{

    const int BLOCK_SIZE = ARR_SIZE/NUM_BLOCKS;
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = width * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + width - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * width;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + width * ty + tx];
        Bs[ty][tx] = B[b + width * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + width * ty + tx] = Csub;
}

int main(int argc, char **argv){

  // GPU Timing variables
  cudaEvent_t startOuter, startInner, /*startSerial,*/ stopOuter, stopInner/*, stopSerial*/;
  float elapsed_gpu_outer, elapsed_gpu_inner/*, elapsed_serial*/;

  // Arrays on GPU global memoryc
  float *d_x;
  float *d_y;
  float *d_result;

  // Arrays on the host memory
  float *h_x;
  float *h_y;
  float *h_result;
  //float *h_serial;

  //int i, j, errCount = 0, zeroCount = 0;

  /*
  if (argc > 1) {
    ARR_SIZE  = atoi(argv[1]);
  }
  else {
    ARR_SIZE = SM_ARR_LEN;
  }*/

  fprintf(stderr, "Length of the array = %d\n", ARR_SIZE);

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate GPU memory
  size_t allocSize = ARR_SIZE * ARR_SIZE * sizeof(float);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSize));

  // Allocate arrays on host memory
  h_x                        = (float *) malloc(allocSize);
  h_y                        = (float *) malloc(allocSize);
  h_result                   = (float *) malloc(allocSize);
  //h_serial              = (float *) malloc(allocSize);

  // Initialize the host arrays
  fprintf(stderr, "\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArrayOrdered2D(h_x, ARR_SIZE);
  initializeArrayOrdered2D(h_y, ARR_SIZE);
  

  //initializeArray1D(h_y, ARR_SIZE, 1467);
  fprintf(stderr, "\t... done\n");

  fprintf(stderr, "Creating cuda events ...");

#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&startOuter);
  cudaEventCreate(&startInner);
  cudaEventCreate(&stopOuter);
  cudaEventCreate(&stopInner);
  // Record event on the default stream
  cudaEventRecord(startOuter, 0);
#endif

  fprintf(stderr, "\t... done\n");

  fprintf(stderr, "Transferring arrays to GPU memory ...");

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, allocSize, cudaMemcpyHostToDevice));

  dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS);
  dim3 dimBlock(ARR_SIZE/NUM_BLOCKS, ARR_SIZE/NUM_BLOCKS);

  fprintf(stderr, "\t... done\n");

//  printf("Launching kernel");

  // Launch the kernel
  cudaPrintfInit();

  fprintf(stderr, "Kernel initialized\n");

#if PRINT_TIME
  cudaEventRecord(startInner, 0);
#endif

  //MatrixMulKernelShared<<<dimGrid, dimBlock>>>(d_x, d_y, d_result, ARR_SIZE);
  matrixMulCUDA<<<dimGrid, dimBlock>>>(d_x, d_y, d_result, ARR_SIZE);
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();


  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSize, cudaMemcpyDeviceToHost));

#if PRINT_TIME
  // Stop and destroy the timer
  cudaEventRecord(stopOuter,0);
  cudaEventRecord(stopInner,0);
  cudaEventSynchronize(stopOuter);
  cudaEventSynchronize(stopInner);
  cudaEventElapsedTime(&elapsed_gpu_outer, startOuter, stopOuter);
  cudaEventElapsedTime(&elapsed_gpu_inner, startInner, stopInner);
  //printf("\nGPU time (start-to-finish): %f (msec)\n", elapsed_gpu_outer);
  //printf("GPU time (kernel only):     %f (msec)\n", elapsed_gpu_inner);
  printf("%f", elapsed_gpu_outer/1000.0);
  cudaEventDestroy(startOuter);
  cudaEventDestroy(startInner);
  cudaEventDestroy(stopOuter);
  cudaEventDestroy(stopInner);
#endif

  fprintf(stderr, "Grid size: %d\nBlock size: %d\n", NUM_BLOCKS, ARR_SIZE/NUM_BLOCKS);

  float checksumGPU = 0;
  //float checksumSerial = 0;

  // get checksum
  checksumGPU = getChecksum(h_result, ARR_SIZE);
  
  if (ARR_SIZE <= 8) {
    fprintf(stderr, "\n");
    printMat(h_result, ARR_SIZE);
  }

  fprintf(stderr, "GPU checksum: %f\n", checksumGPU);
  /*
  cudaEventCreate(&startSerial);
  cudaEventCreate(&stopSerial);
  cudaEventRecord(startSerial, 0);

  MatrixMulOnHostBlocked(h_x, h_y, h_serial, ARR_SIZE); 

  cudaEventRecord(stopSerial,0);
  cudaEventSynchronize(stopSerial);
  cudaEventElapsedTime(&elapsed_serial, startSerial, stopSerial);
  printf("Blocked serial time : %f (msec)\n", elapsed_serial);
  cudaEventDestroy(startSerial);
  cudaEventDestroy(stopSerial);

  checksumSerial = getChecksum(h_serial, ARR_SIZE);

  printf("Serial checksum: %f\n", checksumSerial);

  double maxDiff = 0.0;
  double diff;

  // Compare the results
  for(i = 0; i < ARR_SIZE; i++) {
    for(j = 0; j < ARR_SIZE; j++) {
      diff = fabs(h_result[i*ARR_SIZE+j] - h_serial[i*ARR_SIZE+j]);
      if (diff > maxDiff) {
        //printf("%f - %f = %f", h_result[i*ARR_SIZE+j], h_serial[i*ARR_SIZE+j], diff);
        maxDiff = diff;
      }
    }
  }

  printf("Maximum difference: %f\n", maxDiff);
  */
  /*
  for(i = 0; i < 50; i++) {
    printf("%d:\t%.8f\t%.8f\n", i, h_result_gold[i], h_result[i]);
  }
  */
  /*
  if (errCount > 0) {
    fprintf(stderr, "\n@ERROR: TEST FAILED: %d results did not matched\n", errCount);
  }
  else if (zeroCount > 0){
    fprintf(stderr, "\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
  }
  else {
    fprintf(stderr, "\nTEST PASSED: All results matched\n");
  } 
  */

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_y));
  CUDA_SAFE_CALL(cudaFree(d_result));

  free(h_x);
  free(h_y);
  free(h_result);
  //free(h_serial);

  return 0;
}

void initializeArrayRand2D(float *arr, int len, int seed) {
  int i, j;
  float randNum;
  srand(seed);

  for (i = 0; i < len; i++) {
    for (j = 0; j < len; j++) {
      
      randNum = (float) rand() / (float)(RAND_MAX/len);
      //printf("%f\n", randNum);
      arr[i * len + j] = randNum;
    }
  } 
}

void initializeArrayOrdered2D(float *arr, int len) {
  long int i;

  for (i = 0; i < len*len; i++) {
    arr[i] = (float)i;
  }
} 
