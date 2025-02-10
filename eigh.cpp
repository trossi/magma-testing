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

#ifdef MAGMA
  #include "magma_v2.h"
  #include "magma_operators.h"
#endif

// Check if templated parameter is std::complex
template<typename T>
struct is_complex_t : public std::false_type {};

template<typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};
//~


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

  // Templated wrapper for MAGMA complex types
  template<typename T>
  using magma_complex_num = typename std::conditional<std::is_same<T, float>::value, magmaFloatComplex, magmaDoubleComplex>::type;

  // MAGMA complex solvers (HEEVD)
  template<typename T>
  magma_int_t (*magma_heevd_gpu)(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, magma_complex_num<T> *dA, magma_int_t ldda, T *w, magma_complex_num<T> *wA, magma_int_t ldwa, magma_complex_num<T> *work, magma_int_t lwork, T *rwork, magma_int_t lrwork, magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

  template<>
  magma_int_t (*magma_heevd_gpu<float>)(magma_vec_t, magma_uplo_t, magma_int_t, magma_complex_num<float>*, magma_int_t, float*, magma_complex_num<float>*, magma_int_t, magma_complex_num<float>*, magma_int_t, float*, magma_int_t, magma_int_t*, magma_int_t, magma_int_t*) = &magma_cheevd_gpu;

  template<>
  magma_int_t (*magma_heevd_gpu<double>)(magma_vec_t, magma_uplo_t, magma_int_t, magma_complex_num<double>*, magma_int_t, double*, magma_complex_num<double>*, magma_int_t, magma_complex_num<double>*, magma_int_t, double*, magma_int_t, magma_int_t*, magma_int_t, magma_int_t*) = &magma_zheevd_gpu;

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

#elif defined(HIP)
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

  // Eigensolvers for complex diagonalization (Hermitian matrices, eigenvalues are real)

  template <typename real_t>
  rocblas_status (*rocsolver_heevd)(rocblas_handle, const rocblas_evect, const rocblas_fill, const rocblas_int, rocblas_complex_num<real_t>*, const rocblas_int, real_t*, real_t*, rocblas_int*);

  // rocblas_float_complex
  template<>
  rocblas_status (*rocsolver_heevd<float>)(rocblas_handle, const rocblas_evect, const rocblas_fill, const rocblas_int, rocblas_complex_num<float>*, const rocblas_int, float*, float*, rocblas_int*) = &rocsolver_cheevd;

  // rocblas_double_complex
  template<>
  rocblas_status (*rocsolver_heevd<double>)(rocblas_handle, const rocblas_evect, const rocblas_fill, const rocblas_int, rocblas_complex_num<double>*, const rocblas_int, double*, double*, rocblas_int*) = &rocsolver_zheevd;

#endif // ~HIP

// Helper type for sharing template code between complex and real calculations
template<typename T>
struct d_internal_type
{
    using full_t = T;
    using real_t = T;
};

template<typename U>
struct d_internal_type<std::complex<U>>
{
    using real_t = U;

#if defined(MAGMA)
    using full_t = magma_complex_num<U>;
#elif defined(CUDA)
    //TODO
#elif defined(HIP)
    using full_t = rocblas_complex_num<U>;
#endif
};

// Define combined eigsolver: calls syevd for real matrices and heevd for complex matrices
// define rocsolver_eig() to call rocsolver_syevd() for real template arg, and rocsolver_heevd for complex arg

#if defined(MAGMA)

/*
template<typename T>
magma_int_t (*magma_eigsolver_gpu)(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, typename d_internal_type<T>::full_t *dA, magma_int_t ldda, typename d_internal_type<T>::real_t *w, typename d_internal_type<T>::full_t *wA, magma_int_t ldwa, typename d_internal_type<T>::full_t *work, magma_int_t lwork, typename d_internal_type<T>::real_t *rwork, magma_int_t lrwork, magma_int_t *iwork, magma_int_t liwork, magma_int_t *info) = magma_syevd_gpu<typename d_internal_type<T>::real_t>;

template<typename U>
magma_int_t (*magma_eigsolver_gpu<std::complex<U>>)(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, typename d_internal_type<std::complex<U>>::full_t *dA, magma_int_t ldda, typename d_internal_type<std::complex<U>>::real_t *w, typename d_internal_type<std::complex<U>>::full_t *wA, magma_int_t ldwa, typename d_internal_type<std::complex<U>>::full_t *work, magma_int_t lwork, typename d_internal_type<std::complex<U>>::real_t *rwork, magma_int_t lrwork, magma_int_t *iwork, magma_int_t liwork, magma_int_t *info) = magma_heevd_gpu<U>;
*/

template<typename T>
magma_int_t magma_eigsolver_gpu(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    typename d_internal_type<T>::full_t *dA, magma_int_t ldda, typename d_internal_type<T>::real_t *w,
    typename d_internal_type<T>::full_t *wA, magma_int_t ldwa, typename d_internal_type<T>::full_t *work, 
    magma_int_t lwork, typename d_internal_type<T>::real_t *rwork, magma_int_t lrwork, magma_int_t *iwork,
    magma_int_t liwork, magma_int_t *info)
{
    return magma_heevd_gpu<typename d_internal_type<T>::real_t>(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info);
}

template<>
magma_int_t magma_eigsolver_gpu<float>(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    float *dA, magma_int_t ldda, float *w,
    float *wA, magma_int_t ldwa, float *work, 
    magma_int_t lwork, float *rwork, magma_int_t lrwork, magma_int_t *iwork,
    magma_int_t liwork, magma_int_t *info)
{
    // does not use rwork, lrwork
    return magma_syevd_gpu<float>(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
}

template<>
magma_int_t magma_eigsolver_gpu<double>(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    double *dA, magma_int_t ldda, double *w,
    double *wA, magma_int_t ldwa, double *work, 
    magma_int_t lwork, double *rwork, magma_int_t lrwork, magma_int_t *iwork,
    magma_int_t liwork, magma_int_t *info)
{
    // does not use rwork, lrwork
    return magma_syevd_gpu<double>(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
}

#elif defined(CUDA)
// TODO

#elif defined(HIP)
template<typename T>
rocblas_status (*rocsolver_eig)(rocblas_handle, const rocblas_evect, const rocblas_fill, const rocblas_int, typename d_internal_type<T>::full_t*, const rocblas_int, typename d_internal_type<T>::real_t*, typename d_internal_type<T>::real_t*, rocblas_int*) = rocsolver_syevd<typename d_internal_type<T>::real_t>;

template<typename U>
rocblas_status (*rocsolver_eig<std::complex<U>>)(rocblas_handle, const rocblas_evect, const rocblas_fill, const rocblas_int, typename d_internal_type<std::complex<U>>::full_t*, const rocblas_int, typename d_internal_type<std::complex<U>>::real_t*, typename d_internal_type<std::complex<U>>::real_t*, rocblas_int*) = rocsolver_heevd<U>;
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
            std::cout << A[j * n + i];
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}


template<typename T>
std::vector<T> build_hermitian_matrix(uint32_t seed, uint32_t matrix_size)
{
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
std::vector<T> build_test_matrix(uint32_t seed, uint32_t matrix_size)
{
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


// Specializations for complex numbers, these return a random Hermitian matrix
// NB: did not figure out how to avoid unambiguous templates if using partial specializations like std::complex<U> 

template<>
std::vector<std::complex<float>> build_test_matrix(uint32_t seed, uint32_t matrix_size)
{    
    return build_hermitian_matrix<std::complex<float>>(seed, matrix_size);
}

template<>
std::vector<std::complex<double>> build_test_matrix(uint32_t seed, uint32_t matrix_size)
{    
    return build_hermitian_matrix<std::complex<double>>(seed, matrix_size);
}

template<typename T>
struct Calculator {
    cudaStream_t stream;
    int h_info;
    int n;
    int lda;
    uplo_t uplo;
    vec_mode_t vec;

    using d_full_type = typename d_internal_type<T>::full_t;
    using d_real_type = typename d_internal_type<T>::real_t;
    using h_real_type = d_real_type;

#if defined(MAGMA)
    magma_queue_t queue;
    T *h_wA;
    T *h_work;
    magma_int_t lwork;
    magma_int_t *h_iwork;
    magma_int_t liwork;

    // rwork and lrwork needed for complex MAGMA solver, not used otherwise
    d_real_type *rwork;
    magma_int_t lrwork;

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
    d_real_type *d_work;

#endif

#if defined(MAGMA)
    // Find optimal sizes. Default implementation is for complex matrices (heevd), specialization for real matrices is below
    void magma_query_work_sizes(magma_int_t &lwork_opt, magma_int_t &lrwork_opt, magma_int_t &liwork_opt)
    {
        d_full_type work_temp;
        d_real_type rwork_temp;
        magma_int_t iwork_temp;
        
        magma_eigsolver_gpu<T>(vec, uplo, n, nullptr, lda, nullptr, nullptr, lda, &work_temp, -1, &rwork_temp, -1, &iwork_temp, -1, &h_info);

        lwork_opt = static_cast<magma_int_t>(real(work_temp));
        lrwork_opt = static_cast<magma_int_t>(rwork_temp);
        liwork_opt = iwork_temp;
    }

#endif //~ if defined(MAGMA)

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

        // Allocate work array "E", real valued and length n
        cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(d_real_type) * n);
        // Allocate work info: rocblas_int type
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

    // Solve eigensystem. Eigenvectors will be optionally copied to h_V if not null.
    void calculate(
        const d_full_type* d_A_input,
        d_real_type* d_W,
        h_real_type* h_W,
        T* h_V = nullptr) {

        // The input array gets overwritten so we work on a copy
        d_full_type *d_A;
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(d_full_type) * lda*n);
        cudaMemcpyAsync(d_A, d_A_input, sizeof(d_full_type) * lda*n, cudaMemcpyDeviceToDevice, stream);

#if defined(MAGMA)
        magma_eigsolver_gpu<T>(vec, uplo, n, d_A, lda, h_W, h_wA, lda, h_work, lwork, rwork, lrwork, h_iwork, liwork, &h_info);

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
        rocsolver_eig<T>(handle, vec, uplo, n, d_A, lda, d_W, d_work, d_info);
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
            // Eigenvectors are now in d_A
            cudaMemcpyAsync(h_V, d_A, sizeof(T) * lda*n, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        cudaFree(d_A);
    }
};

#if defined(MAGMA)
template<>
void Calculator<float>::magma_query_work_sizes(magma_int_t &lwork_opt, magma_int_t &lrwork_opt, magma_int_t &liwork_opt)
{
    float work_temp;
    float rwork_temp;
    magma_int_t iwork_temp;
    magma_eigsolver_gpu<float>(vec, uplo, n, nullptr, lda, nullptr, nullptr, lda, &work_temp, -1, &rwork_temp, -1, &iwork_temp, -1, &h_info);
    lwork_opt = static_cast<magma_int_t>(work_temp);
    liwork_opt = iwork_temp;
}

template<>
void Calculator<double>::magma_query_work_sizes(magma_int_t &lwork_opt, magma_int_t &lrwork_opt, magma_int_t &liwork_opt)
{
    double work_temp;
    double rwork_temp;
    magma_int_t iwork_temp;
    magma_eigsolver_gpu<double>(vec, uplo, n, nullptr, lda, nullptr, nullptr, lda, &work_temp, -1, &rwork_temp, -1, &iwork_temp, -1, &h_info);
    lwork_opt = static_cast<magma_int_t>(work_temp);
    liwork_opt = iwork_temp;
}

#endif

/*
template<typename U>
void Calculator<std::complex<U>>::init_magma() {
        {
        magma_init();
        magma_queue_create(0, &queue);
    #if defined(CUDA)
        stream = magma_queue_get_cuda_stream(queue);
    #elif defined(HIP)
        stream = magma_queue_get_hip_stream(queue);
    #endif
        // Query optimal work sizes
        d_full_type lwork_opt;
        magma_int_t liwork_opt;

        d_real_type rwork_opt;

        magma_heevd_gpu<d_real_type>(vec, uplo, n, nullptr, lda, nullptr, nullptr, lda, &lwork_opt, -1, &rwork_opt, -1, &liwork_opt, -1, &h_info);

        lwork = static_cast<magma_int_t>(lwork_opt);
        liwork = liwork_opt;

        // Allocate work arrays
        h_wA = reinterpret_cast<T*>(malloc(sizeof(T) * lda*n));
        h_work = reinterpret_cast<T*>(malloc(sizeof(T) * lwork));
        h_iwork = reinterpret_cast<magma_int_t*>(malloc(sizeof(magma_int_t) * liwork));
    }
}
*/


template <typename T>
void run(int n, int repeat) {

    using real_t = typename d_internal_type<T>::real_t;

    using d_full_type = typename d_internal_type<T>::full_t;
    using d_real_type = typename d_internal_type<T>::real_t;

    std::cout << "RUN"
              << " n: " << n
              << " repeat: " << repeat
              << " dtype: " << typeid(T).name()
              << std::endl;

    const int lda = n;

    // Host eigenvectors, can be complex. Optionally copied from the device after finding solution
    std::vector<T> h_V(lda * n, 0);
    // Host eigenvalues, real
    std::vector<real_t> h_W(n, 0);

    // Build a test matrix. Will be symmetric for real T and Hermitian for complex T
    std::vector<T> h_A = build_test_matrix<T>(n, n);

    std::cout << "Input matrix" << std::endl;
    print_matrix(n, h_A);

    // Device matrix (can be complex)
    d_full_type *d_A = nullptr;
    // Device eigenvalues (real)
    d_real_type *d_W = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * h_A.size());
    cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(d_real_type) * h_W.size());
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
            // Solve eigensystem, eigenvecs are also solved but not copied to host
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
            case NumberType::eComplexFloat:
                run<std::complex<float>>(n, repeat);
                break;
            case NumberType::eComplexDouble:
                run<std::complex<double>>(n, repeat);
                break;
            default:
                break;
        }
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
