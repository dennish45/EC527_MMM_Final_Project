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
#define NUM_BLOCKS         128
#define PRINT_TIME         1
#define SM_ARR_LEN        50000
#define TOL            1e-6
#define ARR_SIZE 2048
//#define ITERS 2000

#define IMUL(a, b) __mul24(a, b)

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


void initializeArray2D(float *arr, int len, int seed);


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

int main(int argc, char **argv){

  // GPU Timing variables
  cudaEvent_t startOuter, startInner, startSerial, stopOuter, stopInner, stopSerial;
  float elapsed_gpu_outer, elapsed_gpu_inner, elapsed_serial;

  // Arrays on GPU global memoryc
  float *d_x;
  float *d_y;
  float *d_result;

  // Arrays on the host memory
  float *h_x;
  float *h_y;
  float *h_result;
  float *h_serial;

  int i, j, errCount = 0, zeroCount = 0;

  /*
  if (argc > 1) {
    ARR_SIZE  = atoi(argv[1]);
  }
  else {
    ARR_SIZE = SM_ARR_LEN;
  }*/

  printf("Length of the array = %d\n", ARR_SIZE);

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
  h_serial              = (float *) malloc(allocSize);

  // Initialize the host arrays
  printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArray2D(h_x, ARR_SIZE, 2453);
  initializeArray2D(h_y, ARR_SIZE, 1234);
  

  //initializeArray1D(h_y, ARR_SIZE, 1467);
  printf("\t... done\n");

  printf("Creating cuda events ...");

#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&startOuter);
  cudaEventCreate(&startInner);
  cudaEventCreate(&stopOuter);
  cudaEventCreate(&stopInner);
  // Record event on the default stream
  cudaEventRecord(startOuter, 0);
#endif

  printf("\t... done\n");

  printf("Transferring arrays to GPU memory ...");

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, allocSize, cudaMemcpyHostToDevice));

  dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS);
  dim3 dimBlock(ARR_SIZE/NUM_BLOCKS, ARR_SIZE/NUM_BLOCKS);

  printf("\t... done\n");

//  printf("Launching kernel");

  // Launch the kernel
  cudaPrintfInit();

  printf("Kernel initialized\n");

#if PRINT_TIME
  cudaEventRecord(startInner, 0);
#endif

  MatrixMulKernelShared<<<dimGrid, dimBlock>>>(d_x, d_y, d_result, ARR_SIZE);
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
  printf("\nGPU time (start-to-finish): %f (msec)\n", elapsed_gpu_outer);
  printf("GPU time (kernel only):     %f (msec)\n", elapsed_gpu_inner);
  cudaEventDestroy(startOuter);
  cudaEventDestroy(startInner);
  cudaEventDestroy(stopOuter);
  cudaEventDestroy(stopInner);
#endif

  printf("Grid size: %d\nBlock size: %d\n", NUM_BLOCKS, ARR_SIZE/NUM_BLOCKS);

  double checksumGPU = 0;
  double checksumSerial = 0;

  // get checksum
  for (i = 0; i < ARR_SIZE; i++) {
    for (j = i; j < ARR_SIZE; j++) {
      checksumGPU += h_result[i * ARR_SIZE + j];
    }
  }


  printf("GPU checksum: %f\n", checksumGPU);

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

  // get checksum
  for (i = 0; i < ARR_SIZE; i++) {
    for (j = i; j < ARR_SIZE; j++) {
      checksumSerial += h_serial[i * ARR_SIZE + j];
    }
  }

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

  /*
  for(i = 0; i < 50; i++) {
    printf("%d:\t%.8f\t%.8f\n", i, h_result_gold[i], h_result[i]);
  }
  */

  if (errCount > 0) {
    printf("\n@ERROR: TEST FAILED: %d results did not matched\n", errCount);
  }
  else if (zeroCount > 0){
    printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
  }
  else {
    printf("\nTEST PASSED: All results matched\n");
  }

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_y));
  CUDA_SAFE_CALL(cudaFree(d_result));

  free(h_x);
  free(h_y);
  free(h_result);
  free(h_serial);

  return 0;
}

void initializeArray2D(float *arr, int len, int seed) {
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
