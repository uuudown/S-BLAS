# ------------------------------------------------------------------------
# File: Makefile
# S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
# ------------------------------------------------------------------------
# Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
# and Linghao Song
# GitHub repo: http://www.github.com/uuudown/S-BLAS
# PNNL-IPID: 31803-E, IR: PNNL-31803
# MIT Lincese.
# ------------------------------------------------------------------------

# environment parameters
CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_SAMPLES_PATH ?= /usr/local/cuda/samples

#compiler
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
CC = g++

#nvcc parameters
NVCC_FLAGS = -O3 -w -m64 -gencode=arch=compute_70,code=compute_70 

#debugging
#NVCC_FLAGS = -O0 -g -G -m64 -gencode=arch=compute_70,code=compute_70 

CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SAMPLES_PATH)/common/inc
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse -Xcompiler -fopenmp -lnccl

all: unit_test spmm_test

unit_test: unit_test.cu matrix.h spmv.h spmm.h mmio_highlevel.h kernel.h utility.h
	$(NVCC) -ccbin $(CC) $(NVCC_FLAGS) unit_test.cu  $(CUDA_INCLUDES) $(CUDA_LIBS) -o $@

spmm_test: spmm_test.cu matrix.h spmv.h spmm.h mmio_highlevel.h kernel.h utility.h
	$(NVCC) -ccbin $(CC) $(NVCC_FLAGS) spmm_test.cu  $(CUDA_INCLUDES) $(CUDA_LIBS) -o $@

clean:
	rm unit_test spmm_test

