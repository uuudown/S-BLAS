// ------------------------------------------------------------------------
// File: utility.h
// S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
// This file defines the utility functions used by other functions.
// ------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
// and Linghao Song
// GitHub repo: http://www.github.com/uuudown/S-BLAS
// PNNL-IPID: 31803-E, IR: PNNL-31803
// MIT Lincese.
// ------------------------------------------------------------------------

#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cusparse.h>
#include "config.h"

// ================================== Error Checking ====================================
/* Error Checking for CUDA API */
#define CUDA_SAFE_CALL( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

/* Error Checking for CUDA Kernel */
#define CUDA_CHECK_ERROR()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

/* Error Checking for cuSparse library */
#define CHECK_CUSPARSE( err ) __cusparseSafeCall( err, __FILE__, __LINE__ )

inline void __cusparseSafeCall( cusparseStatus_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( CUSPARSE_STATUS_SUCCESS != err )
    {
        fprintf(stderr, "CUSPARSE API failed at %s:%i : %d\n",
                 file, line, err);
        exit(-1);
    }
#endif
    return;
}

#define CHECK_NCCL(cmd) do { \
    ncclResult_t r=cmd;\
    if (r!=ncclSuccess){\
        printf("Failed, NCCL error %s: %d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));\
        exit(EXIT_FAILURE);\
    }\
} while(0) \

// ============================== Aloc and Free ================================
#define SAFE_FREE_HOST(X) if ((X) != NULL) { \
           CUDA_SAFE_CALL( cudaFreeHost((X))); \
           (X) = NULL;}

#define SAFE_FREE_GPU(X) if ((X) != NULL) { \
           CUDA_SAFE_CALL( cudaFree((X))); \
           (X) = NULL;}

#define SAFE_FREE_MULTI_GPU(X,Y) if ((X) != NULL) { \
        int current_dev = 0;\
        CUDA_SAFE_CALL( cudaGetDevice(&current_dev) ); \
        for (unsigned i=0; i<(Y); i++) \
        if (((X)[i]) != NULL){ \
            CUDA_SAFE_CALL( cudaSetDevice(i) );\
            CUDA_SAFE_CALL( cudaFree((X)[i]) );\
        }\
        CUDA_SAFE_CALL( cudaFreeHost((X)) ); \
        (X) = NULL; \
        CUDA_SAFE_CALL( cudaSetDevice(current_dev) ); \
        }

#define SAFE_ALOC_HOST(X,Y) CUDA_SAFE_CALL(cudaMallocHost((void**)&(X),(Y)));

#define SAFE_ALOC_GPU(X,Y) CUDA_SAFE_CALL(cudaMalloc((void**)&(X),(Y)));

// ================================== Print ====================================
// print 1D array
template<typename T>
void print_1d_array(T *input, int length)
{
    for (int i = 0; i < length; i++)
    {
        printf("%.3lf, ", (double)input[i]);
        if ((i+1)%10==0)
            printf("\n");
    }
    printf("\n");
}

// ================================== Timing ====================================
double get_cpu_timer()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
    //get current timestamp in milliseconds
	return (double)tp.tv_sec * 1e3 + (double)tp.tv_usec * 1e-3;
}

/* CPU Timer object definition */
typedef struct CPU_Timer
{
    CPU_Timer()
    {
        start = stop = 0.0;
    }
    void start_timer()
    {
        start = get_cpu_timer();
    }
    void stop_timer()
    {
        stop = get_cpu_timer();
    }
    double measure()
    {
        double millisconds = stop - start;
        return millisconds;
    }
    double start;
    double stop;
} cpu_timer;


/* GPU Timer object definition */
typedef struct GPU_Timer
{
    GPU_Timer()
    {
        CUDA_SAFE_CALL( cudaEventCreate(&this->start) );
        CUDA_SAFE_CALL( cudaEventCreate(&this->stop) );
    }
    void start_timer()
    {
        CUDA_SAFE_CALL( cudaEventRecord(this->start) );
    }
    void stop_timer()
    {
        CUDA_SAFE_CALL( cudaEventRecord(this->stop) );
    }
    double measure()
    {
        CUDA_SAFE_CALL( cudaEventSynchronize(this->stop) );
        float millisconds = 0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&millisconds, this->start, this->stop) ); 
        return (double)millisconds;
    }
    cudaEvent_t start;
    cudaEvent_t stop;
} gpu_timer;

// ================================== Correctness ====================================
template <typename T>
bool check_equal(const T* x, const T* y, size_t m)
{
    bool correct = true;
    for (unsigned i = 0; i<m; i++)
    {
        if (abs(x[i]-y[i]) > ERROR_BAR)
        {
            correct = false;
            /*printf("-----%lf----",abs(x[i]-y[i]));*/
        }
    }
    return correct;
}
template bool check_equal<float>(const float*, const float*, size_t);
template bool check_equal<double>(const double*, const double*, size_t);

// ================================== Random ====================================
inline double rand0to1()
{
    return ((double)rand() / (double)RAND_MAX);
}

// ================================== Sorting ====================================
template<typename T>
void swap(T *a , T *b)
{
    T tmp = *a;
    *a = *b;
    *b = tmp;
}

// quick sort key-value pair (child function)
template<typename iT, typename vT>
int partition(iT *key, vT *val, int length, int pivot_index)
{
    int i  = 0 ;
    int small_length = pivot_index;
    iT pivot = key[pivot_index];
    swap<iT>(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap<vT>(&val[pivot_index], &val[pivot_index + (length - 1)]);
    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap<iT>(&key[pivot_index+i],  &key[small_length]);
            swap<vT>(&val[pivot_index+i],&val[small_length]);
            small_length++;
        }
    }
    swap<iT>(&key[pivot_index + length - 1], &key[small_length]);
    swap<vT>(&val[pivot_index + length - 1], &val[small_length]);
    return small_length;
}

// quick sort key-value pair (main function)
template<typename iT, typename vT>
void quick_sort_key_val_pair(iT *key, vT *val, int length)
{
    if(length == 0 || length == 1) return;
    int small_length = partition<iT, vT>(key, val, length, 0) ;
    quick_sort_key_val_pair<iT, vT>(key, val, small_length);
    quick_sort_key_val_pair<iT, vT>(&key[small_length + 1], 
            &val[small_length + 1], length - small_length - 1);
}

// ================================== Reduction ====================================
template<typename vT>
__forceinline__ __device__
vT sum_32_shfl(vT sum)
{
#pragma unroll
    for(int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1)
        sum += __shfl_xor(sum, mask);
    return sum;
}

// segmented sum
template<typename vT, typename bT>
void segmented_sum(vT *input, bT *bit_flag, int length)
{
    if(length == 0 || length == 1) return;
    for (int i = 0; i < length; i++)
    {
        if (bit_flag[i])
        {
            int j = i + 1;
            while (!bit_flag[j] && j < length)
            {
                input[i] += input[j];
                j++;
            }
        }
    }
}

// reduce sum
template<typename T>
T reduce_sum(T *input, int length)
{
    if(length == 0) return 0;
    T sum = 0;
    for (int i = 0; i < length; i++) sum += input[i];
    return sum;
}

// ================================== Scan ====================================
// in-place exclusive scan
template<typename IdxType, typename DataType>
void exclusive_scan(DataType *input, IdxType length)
{
    if(length == 0 || length == 1)
        return;
    DataType old_val, new_val;
    old_val = input[0];
    input[0] = 0;
    for (IdxType i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}

// ================================== search ====================================
template<typename IdxType>
IdxType csr_findRowIdxUsingNnzIdx(const IdxType* rowPtr, IdxType height, IdxType nnzIdx)
{
    for (IdxType i=0; i<height; i++)
    {
        if ((rowPtr[i] <= nnzIdx) && (nnzIdx < rowPtr[i+1]))
            return i;
    }
    return -1;
}

#endif


