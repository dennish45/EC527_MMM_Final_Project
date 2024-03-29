//THIS IS NOT MY CODE. I GOT IT HERE: https://github.com/richardstartin/cppavxbenchmarks/blob/master/mmul.cpp

#include <iostream>
#include <immintrin.h>
#include <random>
#include <chrono>

// g++ -std=c++11 -mfma -mavx2 -march=native -O3 -funroll-loops mmul.cpp -o mmul.out && ./mmul.out

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BENCHMARK SUPPORT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void fill_matrix(float *array, const int size) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_real_distribution<float> reals(-1E6, 1E6);
    for (size_t i = 0; i < size * size; ++i) {
        array[i] = (float)i;//reals(mt);
    }
}

static float* new_matrix(const int size) {
    float* matrix = (float*)aligned_alloc(64, size * size * sizeof(float));
    fill_matrix(matrix, size);
    return matrix;
}


static void clear_matrix(float *matrix, const int size) {
    for (int i = 0; i < size * size; i += 8) {
        _mm256_store_ps(matrix + i, _mm256_setzero_ps());
    }
}


static float* new_empty_matrix(const int size) {
    float* matrix = (float*)aligned_alloc(64, size * size * sizeof(float));
    clear_matrix(matrix, size);
    return matrix;
}


static void print_matrix(const float * matrix, const int size) {
    for (int i = 0; i < size * size; ++i) {
        std::cout << matrix[i] << " ";
    }
    std::cout << std::endl;
}


template<typename MMUL>
static long long time(const int repetitions,
                      const MMUL &impl,
                      const int size,
                      const float* left,
                      const float* right,
                      float* result) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        impl(size, left, right, result);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
}


template<typename MMUL>
static void throughput_benchmark(const std::string name,
                                 const int warmup,
                                 const int iterations,
                                 const int size,
                                 const MMUL &impl) {
    float* left = new_matrix(size);
    float* right = new_matrix(size);
    float* result = new_empty_matrix(size);
    const int repetitions = 1;
    // warmup
    for (int i = 0; i < warmup; ++i) {
        impl(size, left, right, result);
        fill_matrix(left, size);
        fill_matrix(right, size);
        clear_matrix(result, size);
    }
    // measurement
    long long nanos = 0;
    for (int i = 0; i < iterations; ++i) {
        nanos += time(repetitions, impl, size, left, right, result);
        fill_matrix(left, size);
        fill_matrix(right, size);
        clear_matrix(result, size);
    }
    long double seconds = nanos / 1E9;
    long double throughput_ps = repetitions * iterations / seconds;
    long double flops = 2.0 * size * size * size * iterations * repetitions;
    long double cycles = 2.6 * nanos;
    long double flops_per_cycle = flops / cycles;

    std::cout << name << "," << size << "," << throughput_ps << "," << flops_per_cycle << std::endl;
    free(left);
    free(right);
    free(result);
}

template<typename MMUL>
static void verify(const MMUL& trusted, const MMUL& untrusted, const int size) {
    float* left = new_matrix(size);
    float* right = new_matrix(size);
    float* trusted_result = new_empty_matrix(size);
    float* untrusted_result = new_empty_matrix(size);
    trusted(size, left, right, trusted_result);
    untrusted(size, left, right, untrusted_result);

    for (int i = 0; i < size * size; ++i) {
        if (std::abs(trusted_result[i] - untrusted_result[i]) > 1e-5) {
            std::cout << "diff at (" << i / size << ", " << i % size << "). Expected: " << trusted_result[i]
                      << " but found: " << untrusted_result[i] << std::endl;
        }
    }

    free(left);
    free(right);
    free(trusted_result);
    free(untrusted_result);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX IMPL
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void mmul_saxpy_avx(const int n, const float* left, const float* right, float* result) {
    int in = 0;
    for (int i = 0; i < n; ++i) {
        int kn = 0;
        for (int k = 0; k < n; ++k) {
            __m256 aik = _mm256_set1_ps(left[in + k]);
            int j = 0;
            for (; j < n; j += 8) {
                _mm256_store_ps(result + in + j, _mm256_fmadd_ps(aik, _mm256_load_ps(right + kn + j), _mm256_load_ps(result + in + j)));
            }
            for (; j < n; ++j) {
                result[in + j] += left[in + k] * right[kn + j];
            }
            kn += n;
        }
        in += n;
    }
}


static void mmul_tiled_avx_unrolled(const int n, const float *left, const float *right, float *result) {
    const int block_width = n >= 256 ? 512 : 256;
    const int block_height = n >= 512 ? 8 : n >= 256 ? 16 : 32;

    //printf("block_width: %d\n", block_width);
    //printf("block_height: %d\n", block_height);

    for (int column_offset = 0; column_offset < n; column_offset += block_width) {
        for (int row_offset = 0; row_offset < n; row_offset += block_height) {

            //printf("(%d, %d)\n", row_offset, column_offset);

            for (int i = 0; i < n; ++i) {
                for (int j = column_offset; j < column_offset + block_width && j < n; j += 64) {
                    __m256 sum1 = _mm256_load_ps(result + i * n + j);
                    __m256 sum2 = _mm256_load_ps(result + i * n + j + 8);
                    __m256 sum3 = _mm256_load_ps(result + i * n + j + 16);
                    __m256 sum4 = _mm256_load_ps(result + i * n + j + 24);
                    __m256 sum5 = _mm256_load_ps(result + i * n + j + 32);
                    __m256 sum6 = _mm256_load_ps(result + i * n + j + 40);
                    __m256 sum7 = _mm256_load_ps(result + i * n + j + 48);
                    __m256 sum8 = _mm256_load_ps(result + i * n + j + 56);
                    for (int k = row_offset; k < row_offset + block_height && k < n; ++k) {
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
                    _mm256_store_ps(result + i * n + j, sum1);
                    _mm256_store_ps(result + i * n + j + 8, sum2);
                    _mm256_store_ps(result + i * n + j + 16, sum3);
                    _mm256_store_ps(result + i * n + j + 24, sum4);
                    _mm256_store_ps(result + i * n + j + 32, sum5);
                    _mm256_store_ps(result + i * n + j + 40, sum6);
                    _mm256_store_ps(result + i * n + j + 48, sum7);
                    _mm256_store_ps(result + i * n + j + 56, sum8);
                }
            }
        }
    }
}

static void mmul_tiled_avx(const int n, const float *left, const float *right, float *result) {
    const int block_width = n >= 256 ? 512 : 256;
    const int block_height = n >= 512 ? 8 : n >= 256 ? 16 : 32;
    for (int row_offset = 0; row_offset < n; row_offset += block_height) {
        for (int column_offset = 0; column_offset < n; column_offset += block_width) {
            for (int i = 0; i < n; ++i) {
                for (int j = column_offset; j < column_offset + block_width && j < n; j += 8) {
                    __m256 sum = _mm256_load_ps(result + i * n + j);
                    for (int k = row_offset; k < row_offset + block_height && k < n; ++k) {
                        sum = _mm256_fmadd_ps(_mm256_set1_ps(left[i * n + k]), _mm256_load_ps(right + k * n + j), sum);
                    }
                    _mm256_store_ps(result + i * n + j, sum);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SCALAR IMPL
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void mmul_saxpy(const int n, const float* left, const float* right, float* result) {
    int in = 0;
    for (int i = 0; i < n; ++i) {
        int kn = 0;
        for (int k = 0; k < n; ++k) {
            float aik = left[in + k];
            for (int j = 0; j < n; ++j) {
                result[in + j] += aik * right[kn + j];
            }
            kn += n;
        }
        in += n;
    }
}


static void mmul_blocked(const int n, const float* left, const float* right, float* result) {
    int block_size = 8;
    for (int kk = 0; kk < n; kk += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int i = 0; i < n; i++) {
                for (int j = jj; j < jj + block_size; ++j) {
                    float sum = result[i * n + j];
                    for (int k = kk; k < kk + block_size; ++k) {
                        sum += left[i * n + k] * right[k * n + j]; // second read here is an unvectorisable strided access
                    }
                    result[i * n + j] = sum;
                }
            }
        }
    }
}


int main() {
    void (*saxpy)(const int, const float*, const float*, float*) = mmul_saxpy;
    void (*blocked)(const int, const float*, const float*, float*) = mmul_blocked;

    void (*saxpy_avx)(const int, const float*, const float*, float*) = mmul_saxpy_avx;
    void (*tiled_avx)(const int, const float*, const float*, float*) = mmul_tiled_avx;
    void (*tiled_avx_unrolled)(const int, const float*, const float*, float*) = mmul_tiled_avx_unrolled;

    std::cout << "name" << "," << "size" << "," << "throughput (ops/s)" << "," << "flops/cycle" << std::endl;
    for (int i = 64; i <= 1024; i += 64) {
       // verify(saxpy, tiled_avx, i);
       // verify(saxpy, tiled_avx_unrolled, i);
       // throughput_benchmark("blocked", 10, 100, i, blocked);
       // throughput_benchmark("saxpy", 10, 100, i, saxpy);
       // throughput_benchmark("saxpy_avx", 10, 100, i, saxpy_avx);
       // throughput_benchmark("tiled_avx", 10, 100, i, tiled_avx);
        throughput_benchmark("tiled_avx_unrolled", 0, 1, i, tiled_avx_unrolled);
    }
    return 0;
}
