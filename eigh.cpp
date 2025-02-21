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
#include <complex>
#include <type_traits>
#include <iomanip>
#include <cassert>
#include <map>
#include <algorithm>

#ifdef MAGMA
  #include "magma_v2.h"
  #include "magma_operators.h"
#endif

#if defined(CUDA)
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
  #include <cuComplex.h>
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

template<bool flag = false> void static_no_match() { static_assert(flag, "No match"); }

// Check if templated parameter is std::complex
template<typename T>
struct is_complex_t : public std::false_type {};

template<typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

template<typename T>
using enable_if_complex = std::enable_if_t<is_complex_t<T>::value, void>;

template<typename T>
using enable_if_real = std::enable_if_t<!is_complex_t<T>::value, void>;
//~

// Helper types for sharing template code between complex and real calculations
template<typename T, typename Enable = void>
struct maybe_complex;

template<typename T>
struct maybe_complex<T, enable_if_complex<T>> {
    using full_t = T;
    using real_t = typename T::value_type;
};

template<typename T>
struct maybe_complex<T, enable_if_real<T>> {
    using full_t = T;
    using real_t = T;
};

template <typename T, typename Enable = void>
struct solver_backend_types;

template <typename T>
struct solver_backend_types<T, enable_if_real<T>> {
    using dtype_eigval = T;
    using dtype_matrix = T;
};

template <typename T>
struct solver_backend_types<T, enable_if_complex<T>> {
    using dtype_eigval = typename T::value_type;

#if defined(MAGMA)
    using dtype_matrix = typename std::conditional<std::is_same<dtype_eigval, float>::value, magmaFloatComplex, magmaDoubleComplex>::type;
#elif defined(CUDA)
    using dtype_matrix = typename std::conditional<std::is_same<dtype_eigval, float>::value, cuFloatComplex, cuDoubleComplex>::type;
#elif defined(HIP)
    using dtype_matrix = rocblas_complex_num<dtype_eigval>;
#endif
};
//~

#if defined(MAGMA)
  #define uplo_t           magma_uplo_t
  #define UPLO_LOWER       MagmaLower
  #define UPLO_UPPER       MagmaUpper
  #define vec_mode_t       magma_vec_t
  #define VEC_MODE_NO      MagmaNoVec
  #define VEC_MODE_YES     MagmaVec

  template<typename T>
  struct MagmaHelpers {

      using matrix_dtype = typename solver_backend_types<T>::dtype_matrix;
      using real_t = typename solver_backend_types<T>::dtype_eigval;

      static real_t real_part(matrix_dtype magma_number) {

          if constexpr (is_complex_t<T>::value) {
              // ::real() for magma c-variables defined in magma_operators.h
              return ::real(magma_number);
          } else {
              return magma_number;
          }
      }

      // Common eigensolver. For real types the rwork and lrwork inputs are ignored
      static magma_int_t magma_eigsolver_gpu(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, matrix_dtype *dA,
        magma_int_t ldda, real_t *w, matrix_dtype *wA, magma_int_t ldwa, matrix_dtype *work, magma_int_t lwork,
        real_t *rwork, magma_int_t lrwork, magma_int_t *iwork, magma_int_t liwork, magma_int_t *info) {

            if constexpr (std::is_same<T, std::complex<float>>::value) {
                return magma_cheevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info);
            }
            else if constexpr (std::is_same<T, std::complex<double>>::value) {
                return magma_zheevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info);
            }
            else if constexpr (std::is_same<T, float>::value) {
                return magma_ssyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
            }
            else if constexpr (std::is_same<T, double>::value) {
                return magma_dsyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
            }
            else {
                static_no_match();
            }
        }
  };

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

  template<>
  cudaDataType cusolver_dtype<std::complex<float>> = CUDA_C_32F;

  template<>
  cudaDataType cusolver_dtype<std::complex<double>> = CUDA_C_64F;


#elif defined(HIP)
  #define uplo_t           rocblas_fill
  #define UPLO_LOWER       rocblas_fill_lower
  #define UPLO_UPPER       rocblas_fill_upper
  #define vec_mode_t       rocblas_evect
  #define VEC_MODE_NO      rocblas_evect_none
  #define VEC_MODE_YES     rocblas_evect_original

  template<typename T>
  struct RocHelpers {

      using matrix_dtype = typename solver_backend_types<T>::dtype_matrix;
      using real_t = typename solver_backend_types<T>::dtype_eigval;

      // Common eigensolver
      static rocblas_status roc_common_eigsolver(
        rocblas_handle handle,
        const rocblas_evect evect,
        const rocblas_fill uplo,
        const rocblas_int n,
        matrix_dtype* dA,
        const rocblas_int lda,
        real_t* D,
        real_t* E,
        rocblas_int* info) {

            if constexpr (std::is_same<T, std::complex<float>>::value) {
                return rocsolver_cheevd(handle, evect, uplo, n, dA, lda, D, E, info);
            }
            else if constexpr (std::is_same<T, std::complex<double>>::value) {
                return rocsolver_zheevd(handle, evect, uplo, n, dA, lda, D, E, info);
            }
            else if constexpr (std::is_same<T, float>::value) {
                return rocsolver_ssyevd(handle, evect, uplo, n, dA, lda, D, E, info);
            }
            else if constexpr (std::is_same<T, double>::value) {
                return rocsolver_dsyevd(handle, evect, uplo, n, dA, lda, D, E, info);
            }
            else {
                static_no_match();
            }
        }
  };

#endif // ~HIP


template<typename T>
std::vector<T> build_hermitian_matrix(uint32_t seed, uint32_t matrix_size) {

    static_assert(is_complex_t<T>::value, "build_hermitian_matrix<T>() needs std::complex -valued T");

    using real_t = typename T::value_type;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<real_t> dis(0.0, 1.0);
    const uint32_t n = matrix_size;

    std::vector<T> out(n* n, 0);

    for (int i = 0; i < n; i++) {
        // Set off-diagonals to a value < 1
        for (int j = 0; j < n; j++) {

            const real_t re = dis(gen);

            if (i == j) {
                // Diagonal is real
                out[i * n + j] = T(2.0 * re, 0);
            }
            else {
                const real_t im = dis(gen);
                out[i * n + j] = T(re, im);
                out[j * n + i] = T(re, -im);
            }
        }
    }

    return out;
}

// Returns a random symmetric matrix
template<typename T>
std::vector<T> build_symmetric_matrix(uint32_t seed, uint32_t matrix_size) {

    static_assert(!is_complex_t<T>::value, "build_symmetric_matrix<T>() needs real-valued T");

    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    const uint32_t n = matrix_size;

    std::vector<T> out(n* n, 0);

    for (int i = 0; i < n; i++) {
        // Set off-diagonals to a value < 1
        for (int j = 0; j < n; j++) {
            T val = dis(gen);
            out[i * n + j] = val;
            out[j * n + i] = val;
        }
        // Set diagonal
        out[i * n + i] += i + 1;
    }

    return out;
}

template<typename T>
static inline void print_number_formatted(T number) {

    if constexpr (is_complex_t<T>::value) {
        std::printf("(%14.6e, %14.6e)", number.real(), number.imag());
    }
    else {
        std::printf("%14.6e", number);
    }
};

// Stuff for test matrices
template<typename T>
struct MatrixHelpers {
    static void print_matrix(const int &n, const std::vector<T> &A) {

        constexpr int N_MAX_PRINT = 3;

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
                print_number_formatted(A[j * n + i]);
                std::cout << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::flush;
    }

    // Returns a random Hermitian matrix for complex T, and a symmetric matrix for real T
    static std::vector<T> build_test_matrix(uint32_t seed, uint32_t matrix_size) {

        if constexpr (is_complex_t<T>::value) {
            return build_hermitian_matrix<T>(seed, matrix_size);
        }
        else {
            return build_symmetric_matrix<T>(seed, matrix_size);
        }
    }


    // Rotates eigenvector matrix so that the first element each column is positive and real
    static void fix_eigenvector_phase(std::vector<T>& inOut_eigenvector_matrix, size_t matrix_size) {

        if (inOut_eigenvector_matrix.empty() || matrix_size < 1) {
            return;
        }

        assert(matrix_size*matrix_size == inOut_eigenvector_matrix.size());

        for (size_t i = 0; i < inOut_eigenvector_matrix.size(); i += matrix_size) {

            if constexpr (is_complex_t<T>::value) {

                const auto angle = std::arg(inOut_eigenvector_matrix[i]);
                if (angle == 0) continue;

                const auto rotated_angle = (angle < 0) ? M_PI - angle : -angle;
                const auto rotation = std::exp(T(0, rotated_angle));

                for (size_t j = 0; j < matrix_size; j++) {
                    inOut_eigenvector_matrix[i + j] *= rotation;
                }
            }
            else {
                // For real numbers, just flip the overall sign if the first element is negative
                if (inOut_eigenvector_matrix[i] < 0) {
                    for (size_t j = 0; j < matrix_size; j++) {
                        inOut_eigenvector_matrix[i + j] *= -1;
                    }
                }
            }
        }


    }

};
//~


template<typename T>
struct Calculator {
    cudaStream_t stream;
    int h_info;
    int n;
    int lda;
    uplo_t uplo;
    vec_mode_t vec;

    using eigval_t = typename maybe_complex<T>::real_t;

    using backend_dtype = typename solver_backend_types<T>::dtype_matrix;
    using backend_eigval_t = typename solver_backend_types<T>::dtype_eigval;

    static_assert(sizeof(backend_dtype) == sizeof(T), "Size mismatch in input matrix datatype vs backend matrix datatype");
    static_assert(sizeof(backend_eigval_t) == sizeof(eigval_t), "Size mismatch in input real type vs backend real type");

#if defined(MAGMA)
    magma_queue_t queue;
    backend_dtype *h_wA;
    backend_dtype *h_work;
    magma_int_t lwork;
    magma_int_t *h_iwork;
    magma_int_t liwork;

    // rwork and lrwork needed for complex MAGMA solver, not used by the real version
    std::vector<backend_eigval_t> rwork;
    magma_int_t lrwork = 0;

    // Find optimal workgroup sizes
    void magma_query_work_sizes(magma_int_t &lwork_opt, magma_int_t &lrwork_opt, magma_int_t &liwork_opt) {
        backend_dtype work_temp;
        backend_eigval_t rwork_temp;
        magma_int_t iwork_temp;

        MagmaHelpers<T>::magma_eigsolver_gpu(vec, uplo, n, nullptr, lda, nullptr, nullptr, lda, &work_temp, -1, &rwork_temp, -1, &iwork_temp, -1, &h_info);

        lwork_opt = static_cast<magma_int_t>(MagmaHelpers<T>::real_part(work_temp));
        lrwork_opt = static_cast<magma_int_t>(rwork_temp);
        liwork_opt = iwork_temp;
    }

#elif defined(CUDA)
    const cudaDataType cusolver_dtype_real = cusolver_dtype<eigval_t>;
    const cudaDataType cusolver_dtype_complex = cusolver_dtype<T>;

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
    backend_eigval_t *d_work;

#endif

    Calculator(int n, uplo_t uplo, vec_mode_t vec) : n{n}, lda{n}, uplo{uplo}, vec{vec} {

#if defined(MAGMA)
        magma_init();
        magma_queue_create(0, &queue);
    #if defined(CUDA)
        stream = magma_queue_get_cuda_stream(queue);
    #elif defined(HIP)
        stream = magma_queue_get_hip_stream(queue);
    #endif

        magma_query_work_sizes(lwork, lrwork, liwork);

        // Allocate work arrays
        h_wA = reinterpret_cast<backend_dtype*>(malloc(sizeof(backend_dtype) * lda*n));
        h_work = reinterpret_cast<backend_dtype*>(malloc(sizeof(backend_dtype) * lwork));
        h_iwork = reinterpret_cast<magma_int_t*>(malloc(sizeof(magma_int_t) * liwork));

        if constexpr (is_complex_t<T>::value) {
            assert(lrwork > 0 && "Invalid lrwork (complex solver)");
            rwork.resize(lrwork);
        }

#elif defined(CUDA)
        // Initialize
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cusolverDnCreate(&handle);
        cusolverDnSetStream(handle, stream);
        cusolverDnCreateParams(&params);

        // Query work sizes. The DataTypeW must always be real
        cusolverDnXsyevd_bufferSize(
            handle, params, vec, uplo, n, cusolver_dtype_complex, nullptr, lda,
            cusolver_dtype_real, nullptr, cusolver_dtype_complex, &d_work_size,
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

        // Allocate work array "E", real valued and length n
        cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * n);
        // Allocate work info: rocblas_int type
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(rocblas_int));
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

    // Solve eigensystem. Eigenvectors will be optionally copied to h_V if not null.
    void calculate(
        const backend_dtype* d_A_input,
        backend_eigval_t* d_W,
        eigval_t* h_W,
        T* h_V = nullptr) {

        // The input array gets overwritten so we work on a copy
        backend_dtype *d_A;
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(backend_dtype) * lda*n);
        cudaMemcpyAsync(d_A, d_A_input, sizeof(backend_dtype) * lda*n, cudaMemcpyDeviceToDevice, stream);

#if defined(MAGMA)

        MagmaHelpers<T>::magma_eigsolver_gpu(vec, uplo, n, d_A, lda, h_W, h_wA, lda, h_work, lwork, rwork.data(), lrwork, h_iwork, liwork, &h_info);

        // MAGMA outputs eigenvalues to host memory. Copy them to GPU
        cudaMemcpyAsync(d_W, h_W, sizeof(eigval_t) * n, cudaMemcpyHostToDevice, stream);

#elif defined(CUDA)
        cusolverDnXsyevd(
            handle, params, vec, uplo, n, cusolver_dtype_complex, d_A, lda,
            cusolver_dtype_real, d_W, cusolver_dtype_complex, d_work, d_work_size,
            h_work, h_work_size, d_info);
        cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

#elif defined(HIP)
        RocHelpers<T>::roc_common_eigsolver(handle, vec, uplo, n, d_A, lda, d_W, d_work, d_info);
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
        // Eigenvalues
        if (h_W) {
            cudaMemcpyAsync(h_W, d_W, sizeof(T) * n, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
#endif
        // Eigenvectors are now in d_A
        if (h_V) {
            cudaMemcpyAsync(h_V, d_A, sizeof(backend_dtype) * lda*n, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        cudaFree(d_A);
    }
};

struct TestResults {
    int matrix_size = 0;
    double avg_time = 0.0;
    double avg_time_including_init = 0.0;
};

template <typename T>
TestResults run(int n, int repeat, bool rerun_with_inits = true) {

    using eigval_t = typename Calculator<T>::eigval_t;

    using backend_dtype = typename Calculator<T>::backend_dtype;
    using backend_eigval_t = typename Calculator<T>::backend_eigval_t;


    std::cout << "RUN"
              << " n: " << n
              << " repeat: " << repeat
              << " dtype: " << typeid(T).name()
              << std::endl;

    const int lda = n;

    // Host eigenvectors, can be complex. Optionally copied from the device after finding solution
    std::vector<T> h_V(lda * n, 0);
    // Host eigenvalues, real
    std::vector<eigval_t> h_W(n, 0);

    // Build a test matrix. Will be symmetric for real T and Hermitian for complex T
    std::vector<T> h_A = MatrixHelpers<T>::build_test_matrix(n, n);

    std::cout << "Input matrix" << std::endl;
    MatrixHelpers<T>::print_matrix(n, h_A);

    // Device matrix (can be complex)
    backend_dtype *d_A = nullptr;
    // Device eigenvalues (real)
    backend_eigval_t *d_W = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * h_A.size());
    cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(backend_eigval_t) * h_W.size());
    cudaMemcpy(d_A, h_A.data(), sizeof(T) * h_A.size(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    TestResults results;
    results.matrix_size = n;

    uplo_t uplo = UPLO_LOWER;
    vec_mode_t vec = VEC_MODE_YES;

    {
        Calculator<T> calc(n, uplo, vec);

        // Warm up
        calc.calculate(d_A, d_W, h_W.data(), h_V.data());

        // Rotate eigenvectors to a common phase for easier comparison
        MatrixHelpers<T>::fix_eigenvector_phase(h_V, n);

        std::cout << "Output matrix (normalized)" << std::endl;
        MatrixHelpers<T>::print_matrix(n, h_V);

        // Run timing
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < repeat; iter++) {
            // Solve eigensystem, eigenvecs are also solved but not copied to host
            calc.calculate(d_A, d_W, h_W.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> time = t1 - t0;
        results.avg_time = time.count()*1e-3 / repeat;
        std::cout << "average time " << results.avg_time << " s" << std::endl;
    }

    if (rerun_with_inits)
    {
        // Run timing recreating handles etc every time
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < repeat; iter++) {
            Calculator<T> calc(n, uplo, vec);
            calc.calculate(d_A, d_W, h_W.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> time = t1 - t0;
        results.avg_time_including_init = time.count()*1e-3 / repeat;
        std::cout << "average time " << results.avg_time_including_init << " s (including handle creation)" << std::endl;
    }
    cudaFree(d_A);
    cudaFree(d_W);

    return results;
}

const std::vector<std::string> allowed_number_types { "float", "double", "complex_float", "complex_double" };

// Convenience enum to avoid awkward if-else string comparisons
enum class NumberType
{
    eFloat, // 32bit
    eDouble, // 64bit
    eComplexFloat, // complex<float>
    eComplexDouble // complex<double>
};

std::map<std::string, NumberType> number_type_names {
    {"float", NumberType::eFloat},
    {"double", NumberType::eDouble},
    {"complex_float", NumberType::eComplexFloat},
    {"complex_double", NumberType::eComplexDouble}
};

void print_usage() {
    std::cout << "Usage: <executable> <matrix_size> <repeat> <number_type> <rerun_with_inits>\n\n";
    std::cout << "Example: ./exec 3,100,800,3200 10 double 1\n";
    std::cout << "This will solve and time the eigenvalue problem for double-valued,"
        << " symmetric (Hermitian if using complex numbers) matrices of sizes 3,100,800,3200, each repeated 10 times.\n"
        << "The last argument (0 or 1) specifies if the test should be repeated with full recreation of handles etc on each iteration.\n";
    std::cout << "Choose number_type from: 'float', 'double', 'complex_float', 'complex_double'.\n";
    std::cout << std::flush;
}

int main(int argc, char *argv[]) {
    // Default values
    std::list<int> matrix_sizes = {10};
    int repeat = 10;
    NumberType number_type = NumberType::eDouble;
    bool rerun_with_inits = true;

    if (argc <= 1)
    {
        print_usage();
        return EXIT_SUCCESS;
    }

    // Parse args
    if (argc > 1) {
        matrix_sizes.clear();
        char *token = strtok(argv[1], ",");
        while (token != NULL) {
            matrix_sizes.push_back(std::stoi(token));
            token = strtok(NULL, ",");
        }
    }

    if (matrix_sizes.empty()) {
        print_usage();
        return EXIT_SUCCESS;
    }

    if (argc > 2) {
        repeat = std::stoi(argv[2]);
    }
    if (argc > 3) {
        const std::string in_number_type_str = std::string(argv[3]);

        if ( std::find(allowed_number_types.begin(), allowed_number_types.end(), in_number_type_str) == allowed_number_types.end() ) {
            std::printf("Invalid number type: [%s]. Choose from: float, double, complex_float, complex_double",
                in_number_type_str.c_str()
            );
            return EXIT_FAILURE;
        }

        assert(number_type_names.count(in_number_type_str) > 0);
        number_type = number_type_names.at(in_number_type_str);
    }
    if (argc > 4) {
        rerun_with_inits = static_cast<bool>(std::stoi(argv[4]));
    }

    std::vector<TestResults> results;
    results.reserve(matrix_sizes.size());
    // Calculate
    for (int n: matrix_sizes) {

        switch (number_type) {
            case NumberType::eFloat:
                results.push_back(run<float>(n, repeat, rerun_with_inits));
                break;
            case NumberType::eDouble:
                results.push_back(run<double>(n, repeat, rerun_with_inits));
                break;
            case NumberType::eComplexFloat:
                results.push_back(run<std::complex<float>>(n, repeat, rerun_with_inits));
                break;
            case NumberType::eComplexDouble:
                results.push_back(run<std::complex<double>>(n, repeat, rerun_with_inits));
                break;
            default:
                break;
        }
    }

    std::cout << "\n";
    std::cout << "================= SUMMARY =================\n";
    std::printf("%6s %18s %18s\n", "Size", "Avg Time", "Avg Time w/ init");
    for (const TestResults& res : results) {
        std::printf("%6d %18g %18g\n" , res.matrix_size, res.avg_time, res.avg_time_including_init);
    }
    std::cout << std::flush;

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
