#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cstring>

#ifdef MAGMA
  #include "magma_v2.h"
#endif


#if defined(CUDA)
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
#elif defined(HIP)
  #include <hip/hip_runtime.h>
  #include "rocblas/rocblas.h"
  #include "rocsolver/rocsolver.h"

  #define cudaMalloc                            hipMalloc
  #define cudaFree                              hipFree
  #define cudaDeviceReset                       hipDeviceReset
  #define cudaDeviceSynchronize                 hipDeviceSynchronize

  #define cudaMemcpyAsync                       hipMemcpyAsync
  #define cudaMemcpy                            hipMemcpy
  #define cudaMemcpyDeviceToDevice              hipMemcpyDeviceToDevice
  #define cudaMemcpyDeviceToHost                hipMemcpyDeviceToHost
  #define cudaMemcpyHostToDevice                hipMemcpyHostToDevice

  #define cudaStream_t                          hipStream_t
  #define cudaStreamDestroy                     hipStreamDestroy
  #define cudaStreamSynchronize                 hipStreamSynchronize
  #define cudaStreamNonBlocking                 hipStreamNonBlocking
  #define cudaStreamCreateWithFlags             hipStreamCreateWithFlags
#else
  #error "Define CUDA or HIP"
#endif


#if defined(MAGMA)
  #define uplo_t           magma_uplo_t
  #define UPLO_LOWER       MagmaLower
  #define UPLO_UPPER       MagmaUpper
  #define vec_mode_t       magma_vec_t
  #define VEC_MODE_NO      MagmaNoVec
  #define VEC_MODE_YES     MagmaVec

  template <typename T>
  magma_int_t (*magma_syevd_gpu)(magma_vec_t, magma_uplo_t, magma_int_t, T*, magma_int_t, T*, T*, magma_int_t, T*, magma_int_t, magma_int_t*, magma_int_t, magma_int_t*);

  template<>
  magma_int_t (*magma_syevd_gpu<float>)(magma_vec_t, magma_uplo_t, magma_int_t, float*, magma_int_t, float*, float*, magma_int_t, float*, magma_int_t, magma_int_t*, magma_int_t, magma_int_t*) = &magma_ssyevd_gpu;

  template<>
  magma_int_t (*magma_syevd_gpu<double>)(magma_vec_t, magma_uplo_t, magma_int_t, double*, magma_int_t, double*, double*, magma_int_t, double*, magma_int_t, magma_int_t*, magma_int_t, magma_int_t*) = &magma_dsyevd_gpu;

#elif defined(CUDA)
  #define uplo_t           cublasFillMode_t
  #define UPLO_LOWER       CUBLAS_FILL_MODE_LOWER
  #define UPLO_UPPER       CUBLAS_FILL_MODE_UPPER
  #define vec_mode_t       cusolverEigMode_t
  #define VEC_MODE_NO      CUSOLVER_EIG_MODE_NOVECTOR
  #define VEC_MODE_YES     CUSOLVER_EIG_MODE_VECTOR

  template <typename T>
  cudaDataType cusolver_dtype;

  template<>
  cudaDataType cusolver_dtype<float> = CUDA_R_32F;

  template<>
  cudaDataType cusolver_dtype<double> = CUDA_R_64F;

#else
  #define uplo_t           rocblas_fill
  #define UPLO_LOWER       rocblas_fill_lower
  #define UPLO_UPPER       rocblas_fill_upper
  #define vec_mode_t       rocblas_evect
  #define VEC_MODE_NO      rocblas_evect_none
  #define VEC_MODE_YES     rocblas_evect_original

  template <typename T>
  rocblas_status (*rocsolver_syevd)(rocblas_handle, const rocblas_evect, const rocblas_fill, const rocblas_int, T*, const rocblas_int, T*, T*, rocblas_int*);

  template<>
  rocblas_status (*rocsolver_syevd<float>)(rocblas_handle, const rocblas_evect, const rocblas_fill, const rocblas_int, float*, const rocblas_int, float*, float*, rocblas_int*) = &rocsolver_ssyevd;

  template<>
  rocblas_status (*rocsolver_syevd<double>)(rocblas_handle, const rocblas_evect, const rocblas_fill, const rocblas_int, double*, const rocblas_int, double*, double*, rocblas_int*) = &rocsolver_dsyevd;

#endif


constexpr int N_MAX_PRINT = 3;

template <typename T>
void print_matrix(const int &n, const std::vector<T> &A) {
    // Print transpose
    for (int i = 0; i < n; i++) {
        if (N_MAX_PRINT < i && i < n - N_MAX_PRINT - 1) {
            if (i == N_MAX_PRINT + 1) {
                for (int j = 0; j < (N_MAX_PRINT + 1) * 2 + 1; j++) {
                    std::printf(" %14s", "...");
                }
                std::cout << "\n";
            }
            continue;
        }
        for (int j = 0; j < n; j++) {
            if (N_MAX_PRINT < j && j < n - N_MAX_PRINT - 1) {
                if (j == N_MAX_PRINT + 1) {
                    std::printf(" %14s", "...");
                }
                continue;
            }
            std::printf(" %14.6e", A[j * n + i]);
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}


template<typename T>
struct Calculator {
    cudaStream_t stream;
    int h_info;
    int n;
    int lda;
    uplo_t uplo;
    vec_mode_t vec;

#if defined(MAGMA)
    magma_queue_t queue;
    T *h_wA;
    T *h_work;
    magma_int_t lwork;
    magma_int_t *h_iwork;
    magma_int_t liwork;

#elif defined(CUDA)
    cusolverDnHandle_t handle;
    cusolverDnParams_t params;
    int *d_info;
    void *d_work;
    size_t d_work_size;
    void *h_work;
    size_t h_work_size;
#else
    rocblas_handle handle;
    int *d_info;
    T *d_work;

#endif

    Calculator(int n, uplo_t uplo, vec_mode_t vec) : n{n}, lda{n}, uplo{uplo}, vec{vec} {

#if defined(MAGMA)
        // Initialize
        magma_init();
        magma_queue_create(0, &queue);
#if defined(CUDA)
        stream = magma_queue_get_cuda_stream(queue);
#elif defined(HIP)
        stream = magma_queue_get_hip_stream(queue);
#endif
        // Query work sizes
        T lwork_opt;
        magma_int_t liwork_opt;
        magma_syevd_gpu<T>(vec, uplo, n, nullptr, lda, nullptr, nullptr, lda, &lwork_opt, -1, &liwork_opt, -1, &h_info);
        lwork = static_cast<magma_int_t>(lwork_opt);
        liwork = liwork_opt;

        // Allocate work arrays
        h_wA = reinterpret_cast<T*>(malloc(sizeof(T) * lda*n));
        h_work = reinterpret_cast<T*>(malloc(sizeof(T) * lwork));
        h_iwork = reinterpret_cast<magma_int_t*>(malloc(sizeof(magma_int_t) * liwork));

#elif defined(CUDA)
        // Initialize
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cusolverDnCreate(&handle);
        cusolverDnSetStream(handle, stream);
        cusolverDnCreateParams(&params);

        // Query work sizes
        cusolverDnXsyevd_bufferSize(
            handle, params, vec, uplo, n, cusolver_dtype<T>, nullptr, lda,
            cusolver_dtype<T>, nullptr, cusolver_dtype<T>, &d_work_size,
            &h_work_size);

        // Allocate work arrays
        cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size);
        if (0 < h_work_size) {
            h_work = reinterpret_cast<void *>(malloc(h_work_size));
        }
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));

#elif defined(HIP)
        // Initialize
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        rocblas_create_handle(&handle);
        rocblas_set_stream(handle, stream);

        // Allocate work arrays
        cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * n);
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
#endif
    }

    ~Calculator() {
#if defined(MAGMA)
        free(h_iwork);
        free(h_wA);
        free(h_work);
        magma_queue_destroy(queue);
        magma_finalize();

#elif defined(CUDA)
        cudaFree(d_work);
        if (0 < h_work_size) {
            free(h_work);
        }
        cudaFree(d_info);
        cusolverDnDestroy(handle);
        cudaStreamDestroy(stream);

#elif defined(HIP)
        cudaFree(d_work);
        cudaFree(d_info);
        rocblas_destroy_handle(handle);
        cudaStreamDestroy(stream);

#endif
    }

    void calculate(const T* d_A_input, T* d_W, T* h_W, T* h_V = nullptr) {
        // The input array gets overwritten so we work on a copy
        T *d_A;
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * lda*n);
        cudaMemcpyAsync(d_A, d_A_input, sizeof(T) * lda*n, cudaMemcpyDeviceToDevice, stream);

#if defined(MAGMA)
        magma_syevd_gpu<T>(vec, uplo, n, d_A, lda, h_W, h_wA, lda, h_work, lwork, h_iwork, liwork, &h_info);

        // Copy eigenvectors to GPU
        cudaMemcpyAsync(d_W, h_W, sizeof(T) * n, cudaMemcpyHostToDevice, stream);

#elif defined(CUDA)
        cusolverDnXsyevd(
            handle, params, vec, uplo, n, cusolver_dtype<T>, d_A, lda,
            cusolver_dtype<T>, d_W, cusolver_dtype<T>, d_work, d_work_size,
            h_work, h_work_size, d_info);
        cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

#elif defined(HIP)
        rocsolver_syevd<T>(handle, vec, uplo, n, d_A, lda, d_W, d_work, d_info);
        cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

#endif

        if (0 > h_info) {
            std::stringstream ss;
            ss << "info int: " << -h_info;
            throw std::runtime_error(ss.str());
        }

        // Copy to host
#if !defined(MAGMA)
        if (h_W) {
            cudaMemcpyAsync(h_W, d_W, sizeof(T) * n, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
#endif
        if (h_V) {
            cudaMemcpyAsync(h_V, d_A, sizeof(T) * lda*n, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        cudaFree(d_A);
    }
};




template <typename T>
void run(int n, int repeat) {
    std::cout << "RUN"
              << " n: " << n
              << " repeat: " << repeat
              << " dtype: " << typeid(T).name()
              << std::endl;

    const int lda = n;

    std::vector<T> h_A(lda * n, 0);
    std::vector<T> h_V(lda * n, 0);
    std::vector<T> h_W(n, 0);


    std::mt19937 gen(n);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    // Build a symmetric matrix
    for (int i = 0; i < n; i++) {
        // Set off-diagonals to a value < 1
        for (int j = 0; j < n; j++) {
            T val = dis(gen);
            h_A[i * n + j] = val;
            h_A[j * n + i] = val;
        }
        // Set diagonal
        h_A[i * n + i] += i + 1;
    }

    std::cout << "Input matrix" << std::endl;
    print_matrix(n, h_A);

    T *d_A = nullptr;
    T *d_W = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * h_A.size());
    cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(T) * h_W.size());
    cudaMemcpy(d_A, h_A.data(), sizeof(T) * h_A.size(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    uplo_t uplo = UPLO_LOWER;
    vec_mode_t vec = VEC_MODE_YES;
    {
        Calculator<T> calc(n, uplo, vec);

        // Warm up
        calc.calculate(d_A, d_W, h_W.data(), h_V.data());

        std::cout << "Output matrix" << std::endl;
        print_matrix(n, h_V);

        // Run timing
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < repeat; iter++) {
            calc.calculate(d_A, d_W, h_W.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> time = t1 - t0;
        std::cout << "average time " << time.count()*1e-3 / repeat << " s" << std::endl;
    }

    {
        // Run timing recreating handles etc every time
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < repeat; iter++) {
            Calculator<T> calc(n, uplo, vec);
            calc.calculate(d_A, d_W, h_W.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> time = t1 - t0;
        std::cout << "average time " << time.count()*1e-3 / repeat << " s (including handle creation)" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_W);
}


enum class NumberType
{
    eFloat, // 32bit
    eDouble, // 64bit
    eComplexFloat, // complex<float>
    eComplexDouble // complex<double> 
};

int main(int argc, char *argv[]) {
    // Default values
    std::list<int> matrix_sizes = {10};
    int repeat = 10;
    NumberType number_type = NumberType::eDouble;

    // Parse args
    if (argc > 1) {
        matrix_sizes.clear();
        char *token = strtok(argv[1], ",");
        while (token != NULL) {
            matrix_sizes.push_back(std::stoi(token));
            token = strtok(NULL, ",");
        }
    }
    if (argc > 2) {
        repeat = std::stoi(argv[2]);
    }
    if (argc > 3) {
        const std::string in_number_type_str = std::string(argv[3]);

        if (in_number_type_str == "float") {
            number_type = NumberType::eFloat;
        }
        else if (in_number_type_str == "double") {
            number_type = NumberType::eDouble;
        }
        else if (in_number_type_str == "complex_float") {
            number_type = NumberType::eComplexFloat;
        }
        else if (in_number_type_str == "complex_double") {
            number_type = NumberType::eComplexDouble;
        }
        else {
            std::printf("Invalid number type: [%s]. Choose from: float, double, complex_float, complex_double",
                in_number_type_str.c_str());
        }

    }

    // Calculate
    for (auto n: matrix_sizes) {
        
        switch (number_type) {
            case NumberType::eFloat:
                run<float>(n, repeat);
                break;
            case NumberType::eDouble:
                run<double>(n, repeat);
                break;
            default:
                break;
        }
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
