#pragma once

// Specialization of Calculator to real numbers, will work with real symmetric matrices
template<typename T>
struct Calculator<T, false> {
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

#if defined(MAGMA)

    void magma_query_work_sizes(magma_int_t &lwork_opt, magma_int_t &liwork_opt)
    {
        T work_temp;
        T rwork_temp;
        magma_int_t iwork_temp;
        magma_syevd_gpu<T>(vec, uplo, n, nullptr, lda, nullptr, nullptr, lda, &work_temp, -1, &iwork_temp, -1, &h_info);
        lwork_opt = static_cast<magma_int_t>(work_temp);
        liwork_opt = iwork_temp;
    }
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
        magma_query_work_sizes(lwork, liwork);

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
