// ------------------------------------------------------------------------
// File: matrix.h
// S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
// This file define the sparse CSR, COO and CSC formats. 
// It also defines the dense matrix and dense vector.
// ------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
// and Linghao Song
// GitHub repo: http://www.github.com/uuudown/S-BLAS
// PNNL-IPID: 31803-E, IR: PNNL-31803
// MIT Lincese.
// ------------------------------------------------------------------------

#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "mmio.h"
#include "mmio_highlevel.h"
#include "utility.h"
#include "kernel.h"

using namespace std;

//Multi-GPU sparse data sharing policy:
// - none: no gpu allocation, just on cpu
// - replicate: duplicate the copy across all gpus
// - segment: scatter across all gpus
enum GpuSharePolicy {none=0, replicate=1, segment=2};

//Dense data storage format:
enum MajorOrder {row_major=0, col_major=1};

//Conversion from CSR format to CSC format on CPU
template <typename IdxType, typename DataType>
void CsrToCsc(const IdxType m, const IdxType n, const IdxType nnz,
        const IdxType *csrRowPtr, const IdxType *csrColIdx, const DataType *csrVal, 
        IdxType *cscRowIdx, IdxType *cscColPtr, DataType *cscVal)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(IdxType) * (n+1));
    for (IdxType i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }
    // prefix-sum scan to get the column pointer
    exclusive_scan<IdxType,IdxType>(cscColPtr, n + 1);
    IdxType *cscColIncr=(IdxType *)malloc(sizeof(IdxType) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(IdxType) * (n+1));
    // insert nnz to csc
    for (IdxType row = 0; row < m; row++)
    {
        for (IdxType j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            IdxType col = csrColIdx[j];
            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }
    free (cscColIncr);
}

//Conversion from CSC format to CSR format on CPU
template <typename IdxType, typename DataType>
void CscToCsr(const IdxType n, const IdxType m, const IdxType nnz,
        const IdxType *cscColPtr, const IdxType *cscRowIdx,
        const DataType *cscVal, IdxType *csrColIdx, IdxType *csrRowPtr,
        DataType *csrVal)
{
    // histogram in column pointer
    memset (csrRowPtr, 0, sizeof(IdxType) * (m+1));
    for (IdxType i = 0; i < nnz; i++)
    {
        csrRowPtr[cscRowIdx[i]]++;
    }
    // prefix-sum scan to get the column pointer
    exclusive_scan<IdxType,IdxType>(csrRowPtr, m + 1);
    IdxType *csrRowIncr = (IdxType *)malloc(sizeof(IdxType) * (m+1));
    memcpy (csrRowIncr, csrRowPtr, sizeof(IdxType) * (m+1));
    // insert nnz to csr
    for (IdxType col = 0; col < n; col++)
    {
        for (IdxType j = cscColPtr[col]; j < cscColPtr[col+1]; j++)
        {
            IdxType row = cscRowIdx[j];
            csrColIdx[csrRowIncr[row]] = col;
            csrVal[csrRowIncr[row]] = cscVal[j];
            csrRowIncr[row]++;
        }
    }
    free (csrRowIncr);
}

// ================================== COO Matrix ====================================
// Matrix per data element
template<typename IdxType, typename DataType>
struct CooElement
{
    IdxType row;
    IdxType col;
    DataType val;
};
// Function for value comparision
template<typename IdxType, typename DataType>
int cmp_func(const void *aa, const void *bb) 
{
    struct CooElement<IdxType,DataType>* a = (struct CooElement<IdxType,DataType>*) aa;
    struct CooElement<IdxType,DataType>* b = (struct CooElement<IdxType,DataType>*) bb;
    if (a->row > b->row) return +1;
    if (a->row < b->row) return -1;
    if (a->col > b->col) return +1;
    if (a->col < b->col) return -1;
    return 0;
}
//COO matrix definition
template <typename IdxType,typename DataType>
class CooSparseMatrix
{
public:
   CooSparseMatrix() : nnz(0), height(0), width(0), n_gpu(0), policy(none)
   {
       this->cooRowIdx = NULL;
       this->cooColIdx = NULL;
       this->cooVal = NULL;
       this->cooRowIdx_gpu = NULL;
       this->cooColIdx_gpu = NULL;
       this->cooVal_gpu = NULL;
       this->nnz_gpu = NULL;
   }
   ~CooSparseMatrix()
   {
       SAFE_FREE_HOST(this->cooRowIdx);
       SAFE_FREE_HOST(this->cooColIdx);
       SAFE_FREE_HOST(this->cooVal);
       if (n_gpu != 0 and policy != none)
       {
           SAFE_FREE_MULTI_GPU(cooRowIdx_gpu, n_gpu);
           SAFE_FREE_MULTI_GPU(cooColIdx_gpu, n_gpu);
           SAFE_FREE_MULTI_GPU(cooVal_gpu, n_gpu);
           SAFE_FREE_HOST(nnz_gpu);
       }
   }
   CooSparseMatrix(const char* filename, unsigned n_gpu=0, enum GpuSharePolicy policy=none)
   {
       MM_typecode matcode;
       FILE *f;
       cout << "Loading input matrix from '" << filename << "'." << endl;
       if ((f = fopen(filename, "r")) == NULL) 
       {
           cerr << "Error openning file " << filename << endl;
           exit(-1);
       }
       if (mm_read_banner(f, &matcode) != 0) 
       {
           cerr << "Could not process Matrix Market banner." << endl;
           exit(-1);
       }
       int m, n, nz;
       if ((mm_read_mtx_crd_size(f, &m, &n, &nz)) !=0) 
       {
           cerr << "Error reading matrix crd size." << endl;
           exit(-1);
       }
       this->height = (IdxType)m;
       this->width = (IdxType)n;
       this->nnz = (IdxType)nz;

       cout << "Height: " << height << " Width: " << width << " nnz: " << nnz << endl;
       SAFE_ALOC_HOST(this->cooRowIdx, get_nnz_idx_size());
       SAFE_ALOC_HOST(this->cooColIdx, get_nnz_idx_size());
       SAFE_ALOC_HOST(this->cooVal, get_nnz_val_size());

       for (unsigned i=0; i<nnz; i++)
       {
           int row, col;
           double val;
           fscanf(f, "%d %d %lg\n", &row, &col, &val);
           cooRowIdx[i] = row - 1;//we're using coo, count from 0  
           cooColIdx[i] = col - 1;
           cooVal[i] = val;
       }
       //Sorting to ensure COO format
       sortByRow();
       fclose(f);
       if (n_gpu != 0 and policy != none)
       {
           cooRowIdx_gpu = new IdxType*[n_gpu];
           cooColIdx_gpu = new IdxType*[n_gpu];
           cooVal_gpu = new DataType*[n_gpu];
           //this is to replicate arrays on each GPU
           if (policy == replicate)
           {
               for (unsigned i=0; i<n_gpu; i++)
               {
                   CUDA_SAFE_CALL( cudaSetDevice(i) );
                   SAFE_ALOC_GPU(cooRowIdx_gpu[i], get_nnz_idx_size());
                   SAFE_ALOC_GPU(cooColIdx_gpu[i], get_nnz_idx_size());
                   SAFE_ALOC_GPU(cooVal_gpu[i], get_nnz_val_size());
                   CUDA_SAFE_CALL( cudaMemcpy( cooRowIdx_gpu[i], cooRowIdx, 
                               get_nnz_idx_size(), cudaMemcpyHostToDevice) );
                   CUDA_SAFE_CALL( cudaMemcpy( cooColIdx_gpu[i], cooColIdx, 
                               get_nnz_idx_size(), cudaMemcpyHostToDevice) );
                   CUDA_SAFE_CALL( cudaMemcpy( cooVal_gpu[i], cooVal, 
                               get_nnz_val_size(), cudaMemcpyHostToDevice) );
               }
           }
           if (policy == segment)
           {
               SAFE_ALOC_HOST(nnz_gpu, n_gpu*sizeof(IdxType))
               IdxType avg_nnz = ceil(nnz/n_gpu);
               for (unsigned i=0; i<n_gpu; i++)
               {
                   CUDA_SAFE_CALL( cudaSetDevice(i) );
                   nnz_gpu[i] = min( (i+1)*avg_nnz, nnz ) - i*avg_nnz;
                   SAFE_ALOC_GPU(cooRowIdx_gpu[i], get_gpu_nnz_idx_size(i));
                   SAFE_ALOC_GPU(cooColIdx_gpu[i], get_gpu_nnz_idx_size(i));
                   SAFE_ALOC_GPU(cooVal_gpu[i], get_gpu_nnz_val_size(i));
                   CUDA_SAFE_CALL( cudaMemcpy( cooRowIdx_gpu[i], 
                               &cooRowIdx[i*avg_nnz], 
                               get_gpu_nnz_idx_size(i), cudaMemcpyHostToDevice) );
                   CUDA_SAFE_CALL( cudaMemcpy( cooColIdx_gpu[i], 
                               &cooColIdx[i*avg_nnz], 
                               get_gpu_nnz_idx_size(i), cudaMemcpyHostToDevice) );
                   CUDA_SAFE_CALL( cudaMemcpy( cooVal_gpu[i], 
                               &cooVal[i*avg_nnz], 
                               get_gpu_nnz_val_size(i), cudaMemcpyHostToDevice) );
               }
           }
       }
   }
   void sortByRow() 
   {
       struct CooElement<IdxType,DataType>* coo_arr = 
           new struct CooElement<IdxType, DataType> [nnz];
       unsigned size = sizeof(struct CooElement<IdxType,DataType>);
       for (unsigned i = 0; i < nnz; i++)
       {
           coo_arr[i].row = cooRowIdx[i];
           coo_arr[i].col = cooColIdx[i];
           coo_arr[i].val = cooVal[i];
       }
       qsort(coo_arr, nnz, size, cmp_func<IdxType,DataType>);
       for (unsigned i = 0; i < nnz; i++)
       {
           cooRowIdx[i] = coo_arr[i].row;
           cooColIdx[i] = coo_arr[i].col;
           cooVal[i] = coo_arr[i].val;
       }
       delete [] coo_arr;
   }
   size_t get_gpu_nnz_idx_size(unsigned i_gpu)
   {
       assert(i_gpu < n_gpu);
       if (nnz_gpu != NULL)
           return nnz_gpu[i_gpu] * sizeof(IdxType);
       else
           return 0;
   }
   size_t get_gpu_nnz_val_size(unsigned i_gpu)
   {
       assert(i_gpu < n_gpu);
       if (nnz_gpu != NULL)
           return nnz_gpu[i_gpu] * sizeof(DataType);
       else
           return 0;
   }
   size_t get_nnz_idx_size() { return nnz * sizeof(IdxType); }
   size_t get_nnz_val_size() { return nnz * sizeof(DataType); }
public:
   IdxType *cooRowIdx;
   IdxType *cooColIdx;
   DataType *cooVal;

   IdxType **cooRowIdx_gpu;
   IdxType **cooColIdx_gpu;
   DataType **cooVal_gpu;
   IdxType *nnz_gpu; //number of nnz per GPU

   IdxType nnz;
   IdxType height;
   IdxType width;
   unsigned n_gpu;
   enum GpuSharePolicy policy; 
};

// ================================== CSR Matrix ====================================
template <typename IdxType, typename DataType>
class CsrSparseMatrix
{
public:
   CsrSparseMatrix() : nnz(0), height(0), width(0), n_gpu(0), policy(none)
   {
       this->csrRowPtr = NULL;
       this->csrColIdx = NULL;
       this->csrVal = NULL;
       this->csrRowPtr_gpu = NULL;
       this->csrColIdx_gpu = NULL;
       this->csrVal_gpu = NULL;
       this->nnz_gpu = NULL;
       this->starting_row_gpu = NULL;
       this->stoping_row_gpu = NULL;
   }
   ~CsrSparseMatrix()
   {
       SAFE_FREE_HOST(csrRowPtr);
       SAFE_FREE_HOST(csrColIdx);
       SAFE_FREE_HOST(csrVal);
       SAFE_FREE_MULTI_GPU(csrRowPtr_gpu, n_gpu);
       SAFE_FREE_MULTI_GPU(csrColIdx_gpu, n_gpu);
       SAFE_FREE_MULTI_GPU(csrVal_gpu, n_gpu);
       SAFE_FREE_HOST(nnz_gpu);
       SAFE_FREE_HOST(starting_row_gpu);
       SAFE_FREE_HOST(stoping_row_gpu);
   }
   CsrSparseMatrix(const char* filename) : policy(none), n_gpu(0)
   {
       int m=0, n=0, nnzA=0, isSymmetricA;
       mmio_info(&m, &n, &nnzA, &isSymmetricA, filename);
       this->height = (IdxType)m;
       this->width = (IdxType)n;
       this->nnz = (IdxType)nnzA;
       SAFE_ALOC_HOST(this->csrRowPtr,get_row_ptr_size());
       SAFE_ALOC_HOST(this->csrColIdx,get_col_idx_size());
       SAFE_ALOC_HOST(this->csrVal,get_val_size());
       int* csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
       int* csrColIdxA = (int *)malloc(nnzA * sizeof(int));
       double* csrValA = (double *)malloc(nnzA * sizeof(double));
       mmio_data(csrRowPtrA, csrColIdxA, csrValA, filename);
       printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);
       for (int i=0; i<m+1; i++)
       {
           this->csrRowPtr[i] = (IdxType)csrRowPtrA[i];
       }
       for (int i=0; i<nnzA; i++)
       {
           this->csrColIdx[i] = (IdxType)csrColIdxA[i];
           this->csrVal[i] = (DataType)csrValA[i];
       }
       free(csrRowPtrA);
       free(csrColIdxA);
       free(csrValA);
       this->csrRowPtr_gpu = NULL;
       this->csrColIdx_gpu = NULL;
       this->csrVal_gpu = NULL;
       this->nnz_gpu = NULL;
       this->starting_row_gpu = NULL;
       this->stoping_row_gpu = NULL;
   }
   void sync2gpu(unsigned _n_gpu, enum GpuSharePolicy _policy)
   {
        this->n_gpu = _n_gpu;
        this->policy = _policy;
        assert(this->n_gpu != 0);
        assert(this->policy != none);
       if (n_gpu != 0 and policy != none)
       {
           SAFE_ALOC_HOST(csrRowPtr_gpu, n_gpu*sizeof(IdxType*));
           SAFE_ALOC_HOST(csrColIdx_gpu, n_gpu*sizeof(IdxType*));
           SAFE_ALOC_HOST(csrVal_gpu, n_gpu*sizeof(DataType*));
           //this is to replicate arrays on each GPU
           if (policy == replicate)
           {
               for (unsigned i=0; i<n_gpu; i++)
               {
                   CUDA_SAFE_CALL( cudaSetDevice(i) );
                   SAFE_ALOC_GPU(csrRowPtr_gpu[i], get_row_ptr_size());
                   SAFE_ALOC_GPU(csrColIdx_gpu[i], get_col_idx_size());
                   SAFE_ALOC_GPU(csrVal_gpu[i], get_val_size());
                   CUDA_SAFE_CALL( cudaMemcpy( csrRowPtr_gpu[i], csrRowPtr, 
                               get_row_ptr_size(), cudaMemcpyHostToDevice) );
                   CUDA_SAFE_CALL( cudaMemcpy( csrColIdx_gpu[i], csrColIdx, 
                               get_col_idx_size(), cudaMemcpyHostToDevice) );
                   CUDA_SAFE_CALL( cudaMemcpy( csrVal_gpu[i], csrVal, 
                               get_val_size(), cudaMemcpyHostToDevice) );
               }
           }
           else if (policy == segment)
           {
               SAFE_ALOC_HOST(nnz_gpu, n_gpu*sizeof(IdxType));
               SAFE_ALOC_HOST(starting_row_gpu, n_gpu*sizeof(IdxType));
               SAFE_ALOC_HOST(stoping_row_gpu, n_gpu*sizeof(IdxType));
               IdxType avg_nnz = ceil((float)nnz/n_gpu);
               for (unsigned i=0; i<n_gpu; i++)
               {
                   CUDA_SAFE_CALL( cudaSetDevice(i) );
                   IdxType row_starting_nnz = i*avg_nnz;
                   IdxType row_stoping_nnz = min( (i+1)*avg_nnz, nnz ) - 1;
                   nnz_gpu[i] = row_stoping_nnz - row_starting_nnz + 1;
                   starting_row_gpu[i] = csr_findRowIdxUsingNnzIdx(csrRowPtr,
                           height, row_starting_nnz);
                   stoping_row_gpu[i] = csr_findRowIdxUsingNnzIdx(csrRowPtr,
                           height, row_stoping_nnz);
                   IdxType* fixedRowPtr = NULL;
                   SAFE_ALOC_HOST(fixedRowPtr, get_gpu_row_ptr_size(i));
                   fixedRowPtr[0] = 0;
                   for (int k=1; k<get_gpu_row_ptr_num(i)-1; k++)
                       fixedRowPtr[k] = csrRowPtr[starting_row_gpu[i]+k] - i*avg_nnz;
                   fixedRowPtr[get_gpu_row_ptr_num(i)-1] = nnz_gpu[i];
                   SAFE_ALOC_GPU(csrRowPtr_gpu[i], get_gpu_row_ptr_size(i));
                   SAFE_ALOC_GPU(csrColIdx_gpu[i], get_gpu_col_idx_size(i));
                   SAFE_ALOC_GPU(csrVal_gpu[i], get_gpu_nnz_val_size(i));
                   CUDA_SAFE_CALL( cudaMemcpy( csrRowPtr_gpu[i], 
                               fixedRowPtr, get_gpu_row_ptr_size(i), cudaMemcpyHostToDevice) );
                   CUDA_SAFE_CALL( cudaMemcpy( csrColIdx_gpu[i], 
                               &csrColIdx[i*avg_nnz], 
                               get_gpu_col_idx_size(i), cudaMemcpyHostToDevice) );
                   CUDA_SAFE_CALL( cudaMemcpy( csrVal_gpu[i], 
                               &csrVal[i*avg_nnz], 
                               get_gpu_nnz_val_size(i), cudaMemcpyHostToDevice) );
                   printf("gpu-%d,start-row:%d,stop-row:%d,num-rows:%ld,num-nnz:%d\n",
                           i, starting_row_gpu[i], stoping_row_gpu[i], 
                           get_gpu_row_ptr_num(i), nnz_gpu[i]);
                   /*printf("=======RowPtr==========\n");*/
                   /*print_1d_array(fixedRowPtr,get_gpu_row_ptr_num(i));*/
                   SAFE_FREE_HOST(fixedRowPtr);
               }
           }
       }
   }
   size_t get_gpu_row_ptr_num(unsigned i_gpu)
   {
       assert(i_gpu < n_gpu);
       if (nnz_gpu != NULL)
           return (stoping_row_gpu[i_gpu]-starting_row_gpu[i_gpu]+2);
       else return 0;
   }
   size_t get_gpu_row_ptr_size(unsigned i_gpu) //how many rows on this gpu
   {
       return get_gpu_row_ptr_num(i_gpu) * sizeof(IdxType);
   }
   size_t get_gpu_col_idx_num(unsigned i_gpu)
   {
       assert(i_gpu < n_gpu);
       if (nnz_gpu != NULL) return nnz_gpu[i_gpu];
       else return 0;
   }
   size_t get_gpu_col_idx_size(unsigned i_gpu)
   {
       return get_gpu_col_idx_num(i_gpu) * sizeof(IdxType);
   }
   size_t get_gpu_nnz_val_num(unsigned i_gpu)
   {
       assert(i_gpu < n_gpu);
       if (nnz_gpu != NULL) return nnz_gpu[i_gpu];
       else return 0;
   }
   size_t get_gpu_nnz_val_size(unsigned i_gpu)
   {
       return get_gpu_nnz_val_num(i_gpu) * sizeof(DataType);
   }
   size_t get_row_ptr_size() { return (height+1) * sizeof(IdxType); }
   size_t get_col_idx_size() { return nnz * sizeof(IdxType); }
   size_t get_val_size() { return nnz * sizeof(DataType); }
public:
   IdxType *csrRowPtr;
   IdxType *csrColIdx;
   DataType *csrVal;

   IdxType **csrRowPtr_gpu;
   IdxType **csrColIdx_gpu;
   DataType **csrVal_gpu;

   IdxType *nnz_gpu; //number of nnzs coverred by this GPU
   IdxType *starting_row_gpu; //since we partition on elements, it is possible a row
   IdxType *stoping_row_gpu; // is shared by two GPUs

   IdxType nnz;
   IdxType height;
   IdxType width;

   unsigned n_gpu;
   enum GpuSharePolicy policy; 
};

// ================================== CSC Matrix ====================================
template <typename IdxType, typename DataType>
class CscSparseMatrix
{
public:
   CscSparseMatrix() : nnz(0), height(0), width(0)
   {
       this->cscColPtr = NULL;
       this->cscRowIdx = NULL;
       this->cscVal = NULL;
   }
   ~CscSparseMatrix()
   {
       if (this->cscColPtr != NULL)
       {
           CUDA_SAFE_CALL( cudaFreeHost(this->cscColPtr) );
           this->cscColPtr = NULL;
       }
       if (this->cscRowIdx != NULL)
       {
           CUDA_SAFE_CALL( cudaFreeHost(this->cscRowIdx) );
           this->cscRowIdx = NULL;
       }
       if (this->cscVal != NULL)
       {
           CUDA_SAFE_CALL( cudaFreeHost(this->cscVal) );
           this->cscVal = NULL;
       }
   }
   CscSparseMatrix(const CsrSparseMatrix<IdxType, DataType>* csr)
   {
       this->height = csr->height;
       this->width = csr->width;
       this->nnz = csr->nnz;

       cout << "Building csc matrix from a csr matrix." << endl;
       cout << "Height: " << height << " Width: " << width << " nnz: " << nnz << endl;
       SAFE_ALOC_HOST(this->cscColPtr, get_col_ptr_size());
       SAFE_ALOC_HOST(this->cscRowIdx,get_row_idx_size());
       SAFE_ALOC_HOST(this->cscVal,get_val_size());

       CsrToCsc<IdxType, DataType>(height, width, nnz, csr->csrRowPtr, 
               csr->csrColIdx, csr->csrVal, this->cscRowIdx, this->cscColPtr, this->cscVal);
   }

   size_t get_col_ptr_size() { return (width+1) * sizeof(IdxType); }
   size_t get_row_idx_size() { return nnz * sizeof(IdxType); }
   size_t get_val_size() { return nnz * sizeof(DataType); }

public:
   IdxType *cscRowIdx;
   IdxType *cscColPtr;
   DataType *cscVal;
   IdxType nnz;
   IdxType height;
   IdxType width;
};

// =============================== Dense Matrix =================================
template <typename IdxType, typename DataType>
class DenseMatrix
{
public:
    DenseMatrix() : height(0), width(0), order(row_major), n_gpu(0)
    {
        val = NULL;
        val_gpu = NULL;
        dim_gpu = NULL;
        policy = none;
    }

    DenseMatrix(IdxType _height, IdxType _width, enum MajorOrder _order) 
        : height(_height), width(_width), order(_order), n_gpu(0) 
    {
       SAFE_ALOC_HOST(val, get_mtx_size());
       srand(RAND_INIT_SEED);
       for (IdxType i=0; i<get_mtx_num(); i++)
           val[i] = (DataType)rand0to1();
       val_gpu = NULL;
       dim_gpu = NULL;
       policy = none;
    }
    DenseMatrix(IdxType _height, IdxType _width, DataType _val, enum MajorOrder _order) 
        : height(_height), width(_width), order(_order), n_gpu(0)
    {
       SAFE_ALOC_HOST(val, get_mtx_size());
       srand(RAND_INIT_SEED);
       for (IdxType i=0; i<get_mtx_num(); i++)
           val[i] = (DataType)_val;
       val_gpu = NULL;
       dim_gpu = NULL;
       policy = none;
    }

    void sync2gpu(unsigned _n_gpu, enum GpuSharePolicy _policy)
    {
        this->n_gpu = _n_gpu;
        this->policy = _policy;
        assert(this->n_gpu != 0);
        assert(this->policy != none);
        if (policy == replicate)
        {
            SAFE_ALOC_HOST(val_gpu, n_gpu*sizeof(DataType*));
            for (unsigned i=0; i<n_gpu; i++)
            {
                CUDA_SAFE_CALL( cudaSetDevice(i) );
                SAFE_ALOC_GPU( val_gpu[i], get_mtx_size() );
                CUDA_SAFE_CALL( cudaMemcpy( val_gpu[i], val, 
                            get_mtx_size(), cudaMemcpyHostToDevice) );
            }
        }
        else if (policy == segment)
        {
            SAFE_ALOC_HOST(val_gpu, n_gpu*sizeof(DataType*));
            SAFE_ALOC_HOST(dim_gpu, n_gpu*sizeof(IdxType));
            IdxType first_order = (order==row_major)?height:width;
            IdxType second_order = (order==row_major)?width:height;

            IdxType avg_val = ceil((double)first_order/n_gpu);
            for (unsigned i=0; i<n_gpu; i++)
            {
                CUDA_SAFE_CALL( cudaSetDevice(i) );
                dim_gpu[i] = min( (i+1)*avg_val, first_order ) - i*avg_val;
                SAFE_ALOC_GPU(val_gpu[i], second_order * get_dim_gpu_size(i));
                CUDA_SAFE_CALL(cudaMemcpy( val_gpu[i], &val[(i*avg_val)*second_order],
                            second_order * get_dim_gpu_size(i), cudaMemcpyHostToDevice) );
            }
        }
    }
    ~DenseMatrix()
    {
        SAFE_FREE_HOST(val);
        SAFE_FREE_HOST(dim_gpu);
        SAFE_FREE_MULTI_GPU(val_gpu, n_gpu);
    }

    DenseMatrix* transpose()
    {
        assert(n_gpu == 0);//currently only allow transpose when no GPU copy
        assert(policy == none);
        DenseMatrix* trans_mtx = 
            new DenseMatrix(height, width, (order==row_major?col_major:row_major));

        if (order == row_major)
        {
            for (IdxType i=0; i<height; i++)
                for (IdxType j=0; j<width; j++)
                    trans_mtx->val[j*height+i] = this->val[i*width+j];
        }
        else
        {
            for (IdxType i=0; i<height; i++)
                for (IdxType j=0; j<width; j++)
                    trans_mtx->val[i*width+j] = this->val[j*height+i];
        }
        return trans_mtx;
    }

    void sync2cpu(unsigned i_gpu)
    {
        assert(val_gpu != NULL);
        assert(i_gpu < n_gpu);

        if (policy == segment)
        {
            CUDA_SAFE_CALL( cudaSetDevice(i_gpu) );
            IdxType first_order = (order==row_major)?height:width;
            IdxType second_order = (order==row_major)?width:height;
            IdxType avg_val = ceil((double)first_order/n_gpu);
            CUDA_SAFE_CALL( cudaMemcpy( &val[i_gpu*avg_val*second_order], val_gpu[i_gpu], 
                        second_order*get_dim_gpu_size(i_gpu), cudaMemcpyDeviceToHost) );
        }
        else if (policy == replicate)
        {
            CUDA_SAFE_CALL( cudaSetDevice(i_gpu) );
            CUDA_SAFE_CALL( cudaMemcpy( val, val_gpu[i_gpu], 
                        get_mtx_size(), cudaMemcpyDeviceToHost) );
        }
    }
    void plusDenseMatrixGPU (DenseMatrix const &dm, DataType alpha, DataType beta)
    {
        if (n_gpu != 0 && policy != none)
        {
            dim3 blockDim(NUM_THREADS_PER_BLK);
            dim3 gridDim(((get_mtx_num()-1)/NUM_THREADS_PER_BLK)+1);
            for (unsigned i=0; i<n_gpu; i++)
            {
                CUDA_SAFE_CALL( cudaSetDevice(i) );
                denseVector_plusEqual_denseVector<<<gridDim,blockDim>>>(
                        val_gpu[i], dm.val_gpu[i], alpha, beta, get_mtx_num());
            }
            CUDA_CHECK_ERROR();
        }
    }
   size_t get_dim_gpu_size(unsigned i_gpu) 
   { 
       return get_dim_gpu_num(i_gpu)*sizeof(DataType);
   }
   size_t get_dim_gpu_num(unsigned i_gpu)
   {
       assert(i_gpu < n_gpu);
       return (size_t)dim_gpu[i_gpu];
   }
   size_t get_row_size() { return width * sizeof(DataType); }
   size_t get_col_size() { return height * sizeof(DataType); }
   size_t get_mtx_size() { return width * height * sizeof(DataType); }
   size_t get_mtx_num() { return width * height; }
public:
   IdxType height;
   IdxType width;
   DataType* val;
   DataType** val_gpu;
   //num of rows or cols (leading dim) per gpu depending on row-major or col-major
   IdxType *dim_gpu; 
   unsigned n_gpu;
   enum GpuSharePolicy policy; 
   enum MajorOrder order;
};

// =============================== Dense Vector =================================
template <typename IdxType, typename DataType>
class DenseVector
{
public:
    DenseVector() : length(0)
    {
        this->val = NULL;
        this->val_gpu = NULL;
    }
    DenseVector(IdxType _length) : length(_length), n_gpu(0), policy(none)
    {
        this->val_gpu = NULL;
        SAFE_ALOC_HOST(this->val, get_vec_size());
        srand(RAND_INIT_SEED);
        for (IdxType i=0; i<this->get_vec_length(); i++)
            (this->val)[i] = (DataType)rand0to1();
    }
    DenseVector(IdxType _length, DataType _val) : length(_length), n_gpu(0), policy(none)
    {
        this->val_gpu = NULL;
        SAFE_ALOC_HOST(this->val, get_vec_size());
        srand(RAND_INIT_SEED);
        for (IdxType i=0; i<this->get_vec_length(); i++)
            (this->val)[i] = _val;
    }
    DenseVector(const DenseVector& dv) : length(dv.length), n_gpu(dv.n_gpu), policy(dv.policy)
    {
        SAFE_ALOC_HOST(val, get_vec_size());
        memcpy(val, dv.val, get_vec_size());
        if (n_gpu != 0 && policy != none)
        {
            SAFE_ALOC_HOST(val_gpu, n_gpu*sizeof(DataType*));
            for (unsigned i=0; i<n_gpu; i++)
            {
                CUDA_SAFE_CALL( cudaSetDevice(i) );
                SAFE_ALOC_GPU( val_gpu[i], get_vec_size() );
                CUDA_SAFE_CALL( cudaMemcpy( val_gpu[i], dv.val_gpu[i], 
                            get_vec_size(), cudaMemcpyDeviceToDevice) );
            }
        }
    }
    void sync2gpu(unsigned _n_gpu, enum GpuSharePolicy _policy)
    {
        this->n_gpu = _n_gpu;
        this->policy = _policy;
        assert(this->n_gpu != 0);
        assert(this->policy != none);
        assert(this->policy != segment);//assume now vector does not need partition
        if (policy == replicate)
        {
            SAFE_ALOC_HOST(val_gpu, n_gpu*sizeof(DataType*));
            for (unsigned i=0; i<n_gpu; i++)
            {
                CUDA_SAFE_CALL( cudaSetDevice(i) );
                SAFE_ALOC_GPU( val_gpu[i], get_vec_size() );
                CUDA_SAFE_CALL( cudaMemcpy( val_gpu[i], val, 
                            get_vec_size(), cudaMemcpyHostToDevice) );
            }
        }
    }
    void sync2cpu(unsigned i_gpu) //all gpus have the same res vector, pick from any one
    {
        assert(i_gpu < n_gpu);
        assert(val_gpu != NULL);
        CUDA_SAFE_CALL( cudaSetDevice(i_gpu) );
        CUDA_SAFE_CALL( cudaMemcpy( val, val_gpu[i_gpu], 
                    get_vec_size(), cudaMemcpyDeviceToHost) );
    }
    void plusDenseVectorGPU (DenseVector const &dv, DataType alpha, DataType beta)
    {
        if (n_gpu != 0 && policy != none)
        {
            dim3 blockDim(NUM_THREADS_PER_BLK);
            dim3 gridDim(((get_vec_length()-1)/NUM_THREADS_PER_BLK)+1);
            for (unsigned i=0; i<n_gpu; i++)
            {
                CUDA_SAFE_CALL( cudaSetDevice(i) );
                denseVector_plusEqual_denseVector<<<gridDim,blockDim>>>(
                        val_gpu[i], dv.val_gpu[i], alpha, beta, get_vec_length());
            }
            CUDA_CHECK_ERROR();
        }
    }
    ~DenseVector()
    {
        SAFE_FREE_HOST(this->val);
        SAFE_FREE_MULTI_GPU(val_gpu, n_gpu);
    }
    size_t get_vec_size() { return (size_t)length * sizeof(DataType); }
    size_t get_vec_length() { return (size_t)length; }
public:
   IdxType length;
   DataType* val;
   DataType** val_gpu;
   unsigned n_gpu;
   enum GpuSharePolicy policy; 
};

#endif
