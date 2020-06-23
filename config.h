// ------------------------------------------------------------------------
// File: config.h
// S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
// This file include the definitions for the settings.
// ------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
// and Linghao Song
// GitHub repo: http://www.github.com/uuudown/S-BLAS
// PNNL-IPID: 31803-E, IR: PNNL-31803
// MIT Lincese.
// ------------------------------------------------------------------------

#ifndef CONFIG_H
#define CONFIG_H

//whether checking CUDA API and kenrel error
#define CUDA_ERROR_CHECK
//error bar for verification
#define ERROR_BAR (1e-3)
//random initialization seed
#define RAND_INIT_SEED 211
//GPU warp size
#define WARP_SIZE 32
//Default number of threads per thread block
#define NUM_THREADS_PER_BLK 256

#endif
