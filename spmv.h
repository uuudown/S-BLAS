#ifndef SPMV_H
#define SPMV_H

#include <stdio.h>
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
void sblas_spmv_csr_cpu(
        CsrSparseMatrix<IdxType,DataType>* pA, 
        DenseVector<IdxType,DataType>* pB,
        DenseVector<IdxType,DataType>* pC,
        DataType alpha,
        DataType beta)
{
    assert((pA->width) == (pB->length) );
    assert((pB->length) == (pC->length));
    for (IdxType i=0; i<pA->height; i++)
    {
        DataType sum = 0;
        for (IdxType j=pA->csrRowPtr[i]; j<pA->csrRowPtr[i+1]; j++)
        {
            IdxType col_A = pA->csrColIdx[j];
            DataType val_A = pA->csrVal[j];
            DataType val_B = pB->val[col_A];
            sum += val_A * val_B;
        }
        pC->val[i] = beta*(pC->val[i]) + alpha*sum;
    }
}


//========================== GPU V1 ============================
template <typename IdxType, typename DataType>
void sblas_spmv_csr_v1(
        CsrSparseMatrix<IdxType,DataType>* pA, 
        DenseVector<IdxType,DataType>* pB, 
        DenseVector<IdxType,DataType>* pC, 
        DataType alpha,
        DataType beta,
        unsigned n_gpu)
{
    assert((pA->width == pB->length));
    assert((pA->height) == (pC->length));

    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm[n_gpu];

    DenseVector<IdxType,DataType> C_copy(pC->get_vec_length(),0.);
    C_copy.sync2gpu(n_gpu, replicate);



//Start OpenMP
#pragma omp parallel num_threads (n_gpu) shared (comm, id)
{

    cpu_timer nccl_timer;
    nccl_timer.start_timer();
    
    int i_gpu = omp_get_thread_num();
    CUDA_SAFE_CALL( cudaSetDevice(i_gpu) );

    CHECK_NCCL(ncclCommInitRank(&comm[i_gpu], n_gpu, id, i_gpu));

    cusparseHandle_t handle;
    cusparseMatDescr_t mat_A;
    cusparseStatus_t cusparse_status;

    CHECK_CUSPARSE( cusparseCreate(&handle) );
    CHECK_CUSPARSE( cusparseCreateMatDescr(&mat_A) );
    CHECK_CUSPARSE( cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL) ); 
    CHECK_CUSPARSE( cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO) ); 

    nccl_timer.stop_timer();
    cout << "GPU-" << i_gpu << " NCCL Time: " << nccl_timer.measure() << "ms." << endl;

    DataType dummy_alpha = 1.0;
    DataType dummy_beta = 1.0;


    CHECK_CUSPARSE( cusparseDcsrmv(handle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                pA->get_gpu_row_ptr_num(i_gpu)-1,
                pA->width,
                pA->nnz_gpu[i_gpu],
                &dummy_alpha,
                mat_A,
                pA->csrVal_gpu[i_gpu],
                pA->csrRowPtr_gpu[i_gpu],
                pA->csrColIdx_gpu[i_gpu],
                pB->val_gpu[i_gpu],
                &dummy_beta,
                &(C_copy.val_gpu[i_gpu])[pA->starting_row_gpu[i_gpu]]  ) ); 

    //we can add a shift to result matrix to recover its correct row, then perform all 
    //reduce, since NCCL allows inplace all-reduce, we won't need extra buffer.

#pragma omp barrier


    CHECK_CUSPARSE( cusparseDestroyMatDescr(mat_A) );
    CHECK_CUSPARSE( cusparseDestroy(handle) );

    CHECK_NCCL( ncclAllReduce(C_copy.val_gpu[i_gpu], C_copy.val_gpu[i_gpu], 
                C_copy.get_vec_length(), ncclDouble, ncclSum, comm[i_gpu], 0) );

    CHECK_NCCL(ncclCommDestroy(comm[i_gpu]));


    /*CUDA_CHECK_ERROR();*/

}

    pC->plusDenseVectorGPU(C_copy, alpha, beta);
}






#endif
