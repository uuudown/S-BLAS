// ------------------------------------------------------------------------
// File: spmm.h
// S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
// This file implements the Sparse-Matrix-Dense-Matrix multiplication (SPMM).
// ------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
// and Linghao Song
// GitHub repo: http://www.github.com/uuudown/S-BLAS
// PNNL-IPID: 31803-E, IR: PNNL-31803
// MIT Lincese.
// ------------------------------------------------------------------------

#ifndef SPMM_H
#define SPMM_H

#include <iostream>
#include <assert.h>
#include <omp.h>
#include <cusparse.h>
#include <nccl.h>
#include "utility.h"
#include "matrix.h"

using namespace std;
//========================== CPU baseline version ============================
template <typename IdxType, typename DataType>
void sblas_spmm_csr_cpu(
        CsrSparseMatrix<IdxType,DataType>* pA, 
        DenseMatrix<IdxType,DataType>* pB, 
        DenseMatrix<IdxType,DataType>* pC, 
        DataType alpha,
        DataType beta)
{
    assert((pA->width) == (pB->height));
    assert((pA->height) == (pC->height));
    assert((pB->width) == (pC->width));
    if (pB->order == row_major)
    {
        cerr << "SBLAS_SPMM_CSR_CPU: B should be in column major!" << endl;
        exit(-1);
    }
    if (pC->order == row_major)
    {
        for (IdxType i=0; i<pA->height; i++)
        {
            for(IdxType n=0; n<pB->width; n++)
            {
                DataType sum = 0;
                for (IdxType j=pA->csrRowPtr[i]; j<pA->csrRowPtr[i+1]; j++)
                {
                    IdxType col_A = pA->csrColIdx[j];
                    DataType val_A = pA->csrVal[j];
                    DataType val_B = pB->val[n*(pB->height)+col_A];
                    sum += val_A * val_B;
                }
                pC->val[i*(pC->width)+n] = beta*(pC->val[n*(pC->height)+i]) + alpha*sum;
            }
        }
    }
    else
    {
        for (IdxType i=0; i<pA->height; i++)
        {
            for(IdxType n=0; n<pB->width; n++)
            {
                DataType sum = 0;
                for (IdxType j=pA->csrRowPtr[i]; j<pA->csrRowPtr[i+1]; j++)
                {
                    IdxType col_A = pA->csrColIdx[j];
                    DataType val_A = pA->csrVal[j];
                    DataType val_B = pB->val[n*(pB->height)+col_A];
                    sum += val_A * val_B;
                }
                pC->val[n*(pC->height)+i] = beta*(pC->val[n*(pC->height)+i]) + alpha*sum;
            }
        }
    }
}

/** Compute Sparse-Matrix-Dense-Matrix Multiplication using multi-GPUs.
  * Since A and B are allocated on unified memory, there is no need for memcpy.
  * The idea is to reuse A on each GPU and parition B, then each GPU calls 
  * cuSparse single-GPU spMM to compute its own share. For this method, there is 
  * no explicit inter-GPU communication required.
  
  * ---------  C = A * B -----------
  * A[m*k] in CSR sparse format
  * B[k*n] in column major dense format
  * C[m*n] in column major dense format
  */
template <typename IdxType, typename DataType>
void sblas_spmm_csr_v1(
        CsrSparseMatrix<IdxType,DataType>* pA, 
        DenseMatrix<IdxType,DataType>* pB, 
        DenseMatrix<IdxType,DataType>* pC, 
        DataType alpha,
        DataType beta,
        unsigned n_gpu)
{
    assert((pA->width) == (pB->height));
    assert((pA->height) == (pC->height));
    assert((pB->width) == (pC->width) );
    if (pB->order == row_major)
    {
        cerr << "SBLAS_SPMM_CSR_V1: B should be in column major!" << endl;
        exit(-1);
    }
    if (pC->order == row_major)
    {
        cerr << "SBLAS_SPMM_CSR_V1: C should be in column major!" << endl;
        exit(-1);
    }
//Start OpenMP
#pragma omp parallel num_threads (n_gpu)
{
    int i_gpu = omp_get_thread_num();
    CUDA_SAFE_CALL( cudaSetDevice(i_gpu) );
    cusparseHandle_t handle;
    cusparseMatDescr_t  mat_A;
    cusparseStatus_t cusparse_status;
    CHECK_CUSPARSE( cusparseCreate(&handle) );
    CHECK_CUSPARSE( cusparseCreateMatDescr(&mat_A) );
    CHECK_CUSPARSE( cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL) ); 
    CHECK_CUSPARSE( cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO) ); 
    printf("gpu-%d m:%d,n:%ld,k:%d\n",i_gpu, pA->height, pB->get_dim_gpu_num(i_gpu), pA->width);
    CHECK_CUSPARSE( cusparseDcsrmm(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
            pA->height, 
            pB->get_dim_gpu_num(i_gpu),
            pA->width,
            pA->nnz,
            &alpha,
            mat_A,
            pA->csrVal_gpu[i_gpu],
            pA->csrRowPtr_gpu[i_gpu],
            pA->csrColIdx_gpu[i_gpu],
            pB->val_gpu[i_gpu],
            pA->width,
            &beta,
            pC->val_gpu[i_gpu],
            pC->height) );		 	 	
    pC->sync2cpu(i_gpu);
#pragma omp barrier
    CHECK_CUSPARSE( cusparseDestroyMatDescr(mat_A) );
    CHECK_CUSPARSE( cusparseDestroy(handle) );
} //end of omp

}

template <typename IdxType, typename DataType>
void sblas_spmm_csr_v2(
        CsrSparseMatrix<IdxType,DataType>* pA, 
        DenseMatrix<IdxType,DataType>* pB, 
        DenseMatrix<IdxType,DataType>* pC, 
        DataType alpha,
        DataType beta,
        unsigned n_gpu)
{
    assert((pA->width) == (pB->height));
    assert((pA->height) == (pC->height));
    assert((pB->width) == (pC->width) );
    if (pB->order == row_major)
    {
        cerr << "SBLAS_SPMM_CSR_V2: B should be in column major!" << endl;
        exit(-1);
    }
    if (pC->order == row_major)
    {
        cerr << "SBLAS_SPMM_CSR_V2: C should be in col major!" << endl;
        exit(-1);
    }
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm[n_gpu];
    DenseMatrix<IdxType,DataType> C_copy(pC->height, pC->width, 0., row_major);
    C_copy.sync2gpu(n_gpu, replicate);
//Start OpenMP
#pragma omp parallel num_threads (n_gpu) shared (comm, id)
{
    int i_gpu = omp_get_thread_num();
    CUDA_SAFE_CALL( cudaSetDevice(i_gpu) );
    CHECK_NCCL(ncclCommInitRank(&comm[i_gpu], n_gpu, id, i_gpu));
    cusparseHandle_t handle;
    cusparseMatDescr_t  mat_A;
    cusparseStatus_t cusparse_status;
    CHECK_CUSPARSE( cusparseCreate(&handle) );
    CHECK_CUSPARSE( cusparseCreateMatDescr(&mat_A) );
    CHECK_CUSPARSE( cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL) ); 
    CHECK_CUSPARSE( cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO) ); 
    DataType dummy_alpha = 1.0;
    DataType dummy_beta = 1.0;
    CHECK_CUSPARSE( cusparseDcsrmm(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
            pA->get_gpu_row_ptr_num(i_gpu)-1,
            pB->width,
            pA->width,
            pA->nnz_gpu[i_gpu],
            &dummy_alpha,
            mat_A,
            pA->csrVal_gpu[i_gpu],
            pA->csrRowPtr_gpu[i_gpu],
            pA->csrColIdx_gpu[i_gpu],
            pB->val_gpu[i_gpu],
            pB->height,
            &dummy_beta,
            /*C_copy.val_gpu[i_gpu],*/
            &(C_copy.val_gpu[i_gpu])[(pA->starting_row_gpu[i_gpu])],
            /*&(C_copy.val_gpu[i_gpu])[(pA->starting_row_gpu[i_gpu])*(pB->width)],*/
            /*C_copy.val_gpu[i_gpu] + 1,*/
            //pC->width) );		 	 	
            pC->height) );		 	 	
#pragma omp barrier
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#pragma omp barrier
    gpu_timer nccl_timer;
    nccl_timer.start_timer();
    CHECK_NCCL( ncclAllReduce(C_copy.val_gpu[i_gpu], C_copy.val_gpu[i_gpu], 
                C_copy.get_mtx_num(), ncclDouble, ncclSum, comm[i_gpu], 0) );
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#pragma omp barrier
    nccl_timer.stop_timer();
    cout << "GPU-" << i_gpu << " NCCL Time: " << nccl_timer.measure() << "ms." << endl;
    CHECK_CUSPARSE( cusparseDestroyMatDescr(mat_A) );
    CHECK_CUSPARSE( cusparseDestroy(handle) );
    CHECK_NCCL(ncclCommDestroy(comm[i_gpu]));
}
    CUDA_CHECK_ERROR();
    pC->plusDenseMatrixGPU(C_copy, alpha, beta);
}

#endif

