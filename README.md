S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
=================

**Developers:** Ang Li, Chenhao Xie, Jieyang Chen, Shuaiwen Song, Linghao Song, Jiajia Li, Jesun Firoz

**General Area or Topic of Investigation:** Implementing and optimizing sparse Basic Linear Algebra Subprograms (BLAS) on modern multi-GPU systems.

**Release Number:** 0.1

**License:** MIT License, PNNL-IPID: 31803-E, IR: PNNL-31803

Installation Guide
==================

The following sections detail the compilation, packaging, and installation of the software. Also included are test data and scripts to verify the installation was successful.

Environment Requirements
------------------------

**Programming Language:** CUDA C/C++

**Operating System & Version:** Ubuntu 16.04

**Required Disk Space:** 2.5MB (additional space is required for storing test input matrix files).

**Required Memory:** Varies with different tests.

**Nodes / Cores Used:** One node with one or more Nvidia GPUs. Using NSHMEM (sptrsv_v3) requires the GPUs are directly P2P connected by NVLink/NVSwitch/NV-SLI.

Dependencies
------------
| Name | Version | Download Location | Country of Origin | Special Instructions |
| ---- | ------- | ----------------- | ----------------- | -------------------- |
| GCC | 5.4.0 | [https://gcc.gnu.org/](https://gcc.gnu.org/) | USA | None |  
| CUDA | 10.0 or newer | [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) | USA | None |  
| OpenMP | 3.1 or newer |[https://www.openmp.org/](https://www.openmp.org/) | USA | None |  
| NCCL | 2.0 or newer | [https://github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl) | USA | None |

Distribution Files
------------------
| File | Description |
| ---- | ------- |
| config.h | Configure the settings |
| kernel.h | Utility kernels functions |
| matrix.h | CSR, CSC, COO-format sparse matrix, dense matrix and dense vector definition |
| mmio.h   | Matrix markert I/O library for ANSI C |
| mmio_highlevel.h | MMIO high-level API |
| sblas.h | S-BLAS unified header file |
| spmm.h  | Sparse-matrix-dense-matrix multiplication (SPMM) implementation |
| spmm_test.h | Testing file for SPMM |
| spmv.h  | Sparse-matrix-dense-vector multiplication (SPMV) implementation |
| unit_test.cu | Unit test file for matrix.h, spmv.h and spmm.h |
| ash85.mtx | An example matrix-market (.mtx) sparse matrix |
| Makefile  | Makefile for s-blas, update it for target platforms |
| LICENSE   | MIT License file |
| NOTICE    | A notice file for release |
| README.md | This file |


Compilation
-------------------------

(1) Modify ```Makefile```, update the path variables for CUDA_PATH and CUDA_SAMPLE_PATH. Update GPU compute capability. Add NCCL and cuSparse path if necessary.

(2) Type ```make``` in this directory to compile the library. 

(3) Update ```unit_test.cu``` if performing unit test.

Execution
----------

We have two implmentations for SPMM: 1-Partition the dense matrix B, 2-Partition the sparse matrix A. Select the method 1 or 2.

```shell
$ ./unit_test
$ ./spmm 1 ./ash85.mtx 64 1.0 1.0 4
```

We verify the running results using the CPU implmentation. Please see the code for details.

License
----------
This project is licensed under the MIT License, see [LICENSE](./LICENSE) file for details.

Acknowledgments
----------
This work was supported by the S-BLAS project under PNNL's High-Performance-Data-Analytics (HPDA) program. The Pacific Northwest National Laboratory (PNNL) is operated by Battelle for the U.S. Department of Energy (DOE) under contract DE-AC05-76RL01830.
