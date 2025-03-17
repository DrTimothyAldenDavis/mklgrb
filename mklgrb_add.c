// benchmark MKL sparse and SuiteSparse:GraphBLAS

// usage:
//      mklgrb_add filename.lagraph

#ifndef MKL_ILP64
#define MKL_ILP64
#endif

#include "mklgrb.h"

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // startup LAGraph and load the sparse matrix S in CSR format
    //--------------------------------------------------------------------------

    startup_LAGraph ( ) ;
    CHECK (argc >= 2) ;
    char *filename = argv [1] ;
    GrB_Matrix S = NULL, A = NULL, D = NULL, T = NULL, B2 = NULL ;
    load_matrix ("add", filename, &S, &D) ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, S)) ;
    OK (GrB_Matrix_ncols (&ncols, S)) ;

    //--------------------------------------------------------------------------
    // get the # of threads for each trial
    //--------------------------------------------------------------------------

    int nthreads_default = omp_get_max_threads ( ) ;
    fprintf (stderr, "Default # threads: %d\n", nthreads_default) ;
    printf ("Default # threads: %d\n", nthreads_default) ;
    int nt [4], nthreads_trials ;
    char msg [LAGRAPH_MSG_LEN ] ;

#if 0
    nthreads_trials = 3 ;
    nt [0] = 1 ;
    nt [1] = nthreads_default / 2 ;
    nt [2] = nthreads_default ;
#else
    // just run with default # of threads
    nthreads_trials = 1 ;
    nt [0] = nthreads_default ;
#endif

    //--------------------------------------------------------------------------
    // run some benchmarks with the matrix in CSR format
    //--------------------------------------------------------------------------

    GrB_Type type [4] = { GrB_FP64, GrB_FP32, GxB_FC64, GxB_FC32 } ;
    just_practicing = true ;

    // for (int k = 0 ; k < 4 ; k++)    // FIXME: run all types
    for (int k = 0 ; k <= 0 ; k++)    // FIXME: just run GrB_FP64
    {
        // run some benchmarks with A as a sparse CSR matrix of type [k]
        convert_matrix_to_type (&A, type [k], D, S) ;

        // create a matrix B2 of the same type and size as A but with 1/10th
        // the entries
        GrB_Index nvals2 ;
        OK (GrB_Matrix_nvals (&nvals2, A)) ;
        double d = (nvals2/10) / ((double) nrows * ((double) ncols)) ;
        OK (LAGraph_Random_Matrix (&T, GrB_FP64, ncols, nrows, d, 0, msg)) ;
        OK (GrB_Matrix_new (&B2, type [k], nrows, ncols)) ;
        OK (GrB_assign (B2, NULL, NULL, T, GrB_ALL, nrows, GrB_ALL, ncols,
            NULL)) ;
        GrB_free (&T) ;
        OK (GxB_set (B2, GxB_FORMAT, GxB_BY_ROW)) ;
        OK (GxB_set (B2, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
        OK (GrB_Matrix_nvals (&nvals2, B2)) ;
        fprintf (stderr,
            "nnz(B) for sparse add: %10.3fM\n", 1e-6 * (double) nvals2) ;
        printf (
            "nnz(B) for sparse add: %10.3fM\n", 1e-6 * (double) nvals2) ;

        if (just_practicing)
        {
            set_nthreads (nthreads_default) ;
            benchmark_add ("B+A warmup ", 1.0, B2, A, false, false) ;
            just_practicing = false ;
        }

        for (int i = 0 ; i < nthreads_trials ; i++)
        {
            fprintf (stderr, "\nCSR ================ # threads: %d\n", nt [i]) ;
            printf ("\nCSR ================ # threads: %d\n", nt [i]) ;
            set_nthreads (nt [i]) ;

            // sparse add, from easier to harder cases
            benchmark_add ("C=B+A      ", 1.0, B2, A, false, false) ;
            benchmark_add ("C=B+A      ", 1.0, B2, A, false, true) ;

            benchmark_add ("C=2*B+A    ", 2.0, B2, A, false, false) ;
            benchmark_add ("C=2*B+A    ", 2.0, B2, A, false, true) ;

            benchmark_add ("C=B'+A     ", 1.0, B2, A, true, false) ;
            benchmark_add ("C=B'+A     ", 1.0, B2, A, true, true) ;

            benchmark_add ("C=2*B'+A   ", 2.0, B2, A, true, false) ;
            benchmark_add ("C=2*B'+A   ", 2.0, B2, A, true, true) ;

            benchmark_add ("C=A+B      ", 1.0, A, B2, false, false) ;
            benchmark_add ("C=A+B      ", 1.0, A, B2, false, true) ;

            benchmark_add ("C=2*A+B    ", 2.0, A, B2, false, false) ;
            benchmark_add ("C=2*A+B    ", 2.0, A, B2, false, true) ;

            benchmark_add ("C=A+A      ", 1.0, A, A,  false, false) ;
            benchmark_add ("C=A+A      ", 1.0, A, A,  false, true) ;

            benchmark_add ("C=2*A+A    ", 2.0, A, A,  false, false) ;
            benchmark_add ("C=2*A+A    ", 2.0, A, A,  false, true) ;

            benchmark_add ("C=A'+B     ", 1.0, A, B2, true, false) ;
            benchmark_add ("C=A'+B     ", 1.0, A, B2, true, true) ;

            benchmark_add ("C=2*A'+B   ", 2.0, A, B2, true, false) ;
            benchmark_add ("C=2*A'+B   ", 2.0, A, B2, true, true) ;

            benchmark_add ("C=A'+A     ", 1.0, A, A,  true, false) ;
            benchmark_add ("C=A'+A     ", 1.0, A, A,  true, true) ;

            benchmark_add ("C=2*A'+A   ", 2.0, A, A,  true, false) ;
            benchmark_add ("C=2*A'+A   ", 2.0, A, A,  true, true) ;

        }
        GrB_free (&A) ;
        GrB_free (&B2) ;
    }

    //--------------------------------------------------------------------------
    // free everything and finalize LAGraph
    //--------------------------------------------------------------------------

    set_nthreads (nthreads_default) ;
    GrB_free (&S) ;
    GrB_free (&D) ;
    finish_LAGraph ( ) ;
}

