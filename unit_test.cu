// ------------------------------------------------------------------------
// File: utility.h
// S-BLAS: A Scalable Sparse-BLAS Kernel Library for Multi-GPUs.
// This file defines the unit_test functions.
// ------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// Other PNNL Developers: Chenhao Xie, Jieyang Chen, Jiajia Li, Jesun Firoz
// and Linghao Song
// GitHub repo: http://www.github.com/uuudown/S-BLAS
// PNNL-IPID: 31803-E, IR: PNNL-31803
// MIT Lincese.
// ------------------------------------------------------------------------

#include "matrix.h"
#include "sblas.h"
#include "spmm.h"

/* This unit test-function tests COO Matrix */
bool cooMatrixTest()
{
    //============= COO =============
    //default construction function
    CooSparseMatrix<int,double> cooMtx1;
    //load from file
    CooSparseMatrix<int,double> cooMtx2("./ash85.mtx");
    return true;
}

/* This unit test-function tests CSR Matrix */
bool csrMatrixTest()
{
    //============= CSR =============
    //default construction function
    CsrSparseMatrix<unsigned,double> csrMtx1;
    //load from file
    CsrSparseMatrix<unsigned,double> csrMtx2("./ash85.mtx");
    return true;
}

/* This unit test-function tests CSC Matrix */
bool cscMatrixTest()
{
    //============= CSC =============
    //default construction function
    CscSparseMatrix<unsigned,double> cscMtx1;
    //obtain from csr matrix
    CsrSparseMatrix<unsigned,double> csrMtx1("./ash85.mtx");
    CscSparseMatrix<unsigned,double> cscMtx2(&csrMtx1);
    return true;
}

/* This unit test-function tests Dense Matrix */
bool denseMatrixTest()
{
    //============= Dense =============
    //default construction function
    DenseMatrix<unsigned,double> denseMtx1;
    //random generate
    DenseMatrix<unsigned,double> denseMtx2(2048, 2048, row_major);
    DenseMatrix<unsigned,double>* p_denseMtx3 = NULL;
    p_denseMtx3 = denseMtx2.transpose();
    return true;
}

bool spmmCsrTest(unsigned b_width, double alpha, double beta, unsigned n_gpu)
{
    cpu_timer load_timer, run_timer, run_cpu_timer;
    load_timer.start_timer();
    CsrSparseMatrix<int, double> A("./ash85.mtx");
    DenseMatrix<int,double> B(A.width, b_width, col_major);
    DenseMatrix<int,double> C(A.height, b_width, 1, col_major);
    DenseMatrix<int,double> C_cpu(A.height, b_width, 1, col_major);
    //Partition and Distribute
    A.sync2gpu(n_gpu, replicate);
    B.sync2gpu(n_gpu, segment);
    C.sync2gpu(n_gpu, segment);
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    load_timer.stop_timer();
    run_timer.start_timer();
    sblas_spmm_csr_v1<int, double>(&A, &B, &C, alpha, beta, n_gpu);
    CUDA_CHECK_ERROR();
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    run_timer.stop_timer();
    run_cpu_timer.start_timer();
    sblas_spmm_csr_cpu<int, double>(&A, &B, &C_cpu, alpha, beta);
    CUDA_CHECK_ERROR();
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    run_cpu_timer.stop_timer();
    //print_1d_array(C_cpu.val,C_cpu.get_mtx_num());
    //print_1d_array(C.val,C.get_mtx_num());
    bool correct = check_equal(C_cpu.val, C.val, C.get_mtx_num());
    cout << "Validation = " << (correct?"True":"False") << endl;
    cout << "Load Time: " << load_timer.measure() << "ms." << endl;
    cout << n_gpu << "-GPUs Run Time: " << run_timer.measure() << " ms." << endl;
    cout << "CPU Run Time: " << run_cpu_timer.measure() << " ms." << endl;
    return correct;
}

bool spmmCsrTest2(unsigned b_width, double alpha, double beta, unsigned n_gpu)
{
    cpu_timer load_timer, run_timer, run_cpu_timer;
    load_timer.start_timer();
    CsrSparseMatrix<int, double> A("./ash85.mtx");
    DenseMatrix<int,double> B(A.width, b_width, col_major);
    DenseMatrix<int,double> C(A.height, b_width, 1, col_major);
    DenseMatrix<int,double> C_cpu(A.height, b_width, 1, col_major);

    //Partition and Distribute
    A.sync2gpu(n_gpu, segment);
    B.sync2gpu(n_gpu, replicate);
    C.sync2gpu(n_gpu, replicate);
    
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    load_timer.stop_timer();
    run_timer.start_timer();
    sblas_spmm_csr_v2<int, double>(&A, &B, &C, alpha, beta, n_gpu);
    CUDA_CHECK_ERROR();
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    run_timer.stop_timer();
    
    run_cpu_timer.start_timer();
    sblas_spmm_csr_cpu<int, double>(&A, &B, &C_cpu, alpha, beta);
    CUDA_CHECK_ERROR();
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    run_cpu_timer.stop_timer();
    //get data back to CPU
    C.sync2cpu(0);
    //print_1d_array(C.val,C.get_mtx_num());
    //print_1d_array(C_cpu.val,C_cpu.get_mtx_num());
    bool correct = check_equal(C_cpu.val, C.val, C.get_mtx_num());
    cout << "Validation = " << (correct?"True":"False") << endl;
    cout << "Load Time: " << load_timer.measure() << "ms." << endl;
    cout << n_gpu << "-GPUs Run Time: " << run_timer.measure() << " ms." << endl;
    cout << "CPU Run Time: " << run_cpu_timer.measure() << " ms." << endl;
    return correct;
}

bool spmvCsrTest(double alpha, double beta, unsigned n_gpu)
{
    cpu_timer load_timer, run_timer, run_cpu_timer;
    load_timer.start_timer();
    //Correct
    CsrSparseMatrix<int, double> A("./ash85.mtx");
    DenseVector<int,double> B(A.width,1.);
    DenseVector<int,double> C(A.height,1.);
    DenseVector<int,double> C_cpu(A.height,1.);
    //Partition and Distribute
    A.sync2gpu(n_gpu, segment);
    B.sync2gpu(n_gpu, replicate);
    C.sync2gpu(n_gpu, replicate);
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    load_timer.stop_timer();
    //CPU Baseline
    run_cpu_timer.start_timer();
    sblas_spmv_csr_cpu<int, double>(&A, &B, &C_cpu, alpha, beta);
    CUDA_CHECK_ERROR();
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    run_cpu_timer.stop_timer();
    run_timer.start_timer();
    sblas_spmv_csr_v1<int, double>(&A, &B, &C, alpha, beta, n_gpu);
    CUDA_CHECK_ERROR();
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    run_timer.stop_timer();
    //get data back to CPU
    C.sync2cpu(0);
    //print_1d_array(C.val,C.get_vec_length());
    //print_1d_array(C_cpu.val,C_cpu.get_vec_length());
    bool correct = check_equal(C_cpu.val, C.val, C.get_vec_length());
    cout << "Validation = " << (correct?"True":"False") << endl;
    cout << "Load Time: " << load_timer.measure() << "ms." << endl;
    cout << "CPU Run Time: " << run_cpu_timer.measure() << " ms." << endl;
    cout << n_gpu << "-GPUs Run Time: " << run_timer.measure() << " ms." << endl;
    return correct;
}

int main()
{
    cooMatrixTest();
    csrMatrixTest();
    cscMatrixTest();
    denseMatrixTest();
    spmmCsrTest(256,3.0,4.0,4);
    spmmCsrTest2(256,3.0,4.0,4);
    spmvCsrTest(3.0,4.0,4);
    return 0;
}



