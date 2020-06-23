// ------------------------------------------------------------------------
// File: kernel.h
// S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
// This file define the utility kernels.
// ------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
// and Linghao Song
// GitHub repo: http://www.github.com/uuudown/S-BLAS
// PNNL-IPID: 31803-E, IR: PNNL-31803
// MIT Lincese.
// ------------------------------------------------------------------------

#ifndef KERNEL_H
#define KERNEL_H

template <typename IdxType, typename DataType>
__global__ void denseVector_plusEqual_scalar(DataType* vec, DataType val,
        DataType beta, IdxType n)
{
    int tid = blockIdx.x*gridDim.x + threadIdx.x;
    for (IdxType i=tid; i<n; i+=gridDim.x*blockDim.x)
        vec[i] = vec[i]*beta + val;
}

template <typename IdxType, typename DataType>
__global__ void denseVector_plusEqual_denseVector(DataType* vec0, const DataType* vec1,
        DataType alpha, DataType beta, IdxType n)
{
    //Y = Y*beta + X*alpha
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for (IdxType i=tid; i<n; i+=gridDim.x*blockDim.x)
    {
        vec0[i] = vec0[i]*beta + vec1[i] * alpha;
        /*printf("%lf,%lf,alpha:%lf, %lf\n",vec0[i],vec1[i], alpha, beta);*/
    }
}

#endif
