// benchmark MKL sparse and SuiteSparse:GraphBLAS: spmmd

// usage:
//      mklgrb_spmmd filename.lagraph

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
    GrB_Matrix S = NULL, A = NULL, D = NULL, Right = NULL, T = NULL ;
    load_matrix ("spmmd", filename, &S, &D) ;
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

        GrB_Index nvals2 = 1000 ;
        GrB_Index thin = 8 ;

        // create a ncols-by-thin matrix Right with ~nvals2 entries
        double d = nvals2 / ((double) ncols * thin) ;
        OK (LAGraph_Random_Matrix (&T, GrB_FP64, ncols, thin, d, 0, msg)) ;
        OK (GrB_Matrix_new (&Right, type [k], ncols, thin)) ;
        OK (GrB_assign (Right, NULL, NULL, T, GrB_ALL, ncols, GrB_ALL, thin,
            NULL)) ;
        GrB_free (&T) ;
        OK (GxB_set (Right, GxB_FORMAT, GxB_BY_ROW)) ;
        OK (GxB_set (Right, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
        // GxB_print (Right, 2) ;

        if (just_practicing)
        {
            set_nthreads (nthreads_default) ;
            benchmark_spmmd ("A*R warmup ", A, Right, false, false) ;
            just_practicing = false ;
        }

        for (int i = 0 ; i < nthreads_trials ; i++)
        {
            fprintf (stderr, "\nCSR ================ # threads: %d\n", nt [i]) ;
            printf ("\nCSR ================ # threads: %d\n", nt [i]) ;
            set_nthreads (nt [i]) ;

            // sparse times sparse where one matrix is narrow
            if (nrows == ncols)
            {
            benchmark_spmmd ("C=A'*R     ", A, Right, true, true) ;
            benchmark_spmmd ("C=A'*R     ", A, Right, true, false) ;
            }
            benchmark_spmmd ("C=A*R      ", A, Right, false, true) ;
            benchmark_spmmd ("C=A*R      ", A, Right, false, false) ;

        }
        GrB_free (&A) ;
        GrB_free (&Right) ;
    }

    //--------------------------------------------------------------------------
    // convert S to sparse CSC
    //--------------------------------------------------------------------------

#if 1

    set_nthreads (nthreads_default) ;
    double tconvert = omp_get_wtime ( ) ;
    convert_matrix_format (S, GxB_BY_COL) ;
    tconvert = omp_get_wtime ( ) - tconvert ;

    //--------------------------------------------------------------------------
    // run some benchmarks with the matrix in CSC format
    //--------------------------------------------------------------------------

    // for (int k = 0 ; k < 4 ; k++)    // FIXME: run all types
    for (int k = 0 ; k <= 0 ; k++)  // FIXME: just run GrB_FP64
    {
        // run some benchmarks with A as a sparse CSC matrix of type [k]
        convert_matrix_to_type (&A, type [k], D, S) ;

        GrB_Index nvals2 = 1000 ;
        GrB_Index thin = 8 ;
        double d ;

        // create a ncols-by-thin matrix Right with ~nvals2 entries
        d = nvals2 / ((double) ncols * thin) ;
        OK (LAGraph_Random_Matrix (&T, GrB_FP64, ncols, thin, d, 0, msg)) ;
        OK (GrB_Matrix_new (&Right, type [k], ncols, thin)) ;
        OK (GrB_assign (Right, NULL, NULL, T, GrB_ALL, ncols, GrB_ALL, thin,
            NULL)) ;
        GrB_free (&T) ;
        OK (GxB_set (Right, GxB_FORMAT, GxB_BY_COL)) ;
        OK (GxB_set (Right, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
        // GxB_print (Right, 2) ;

        bool C_is_csr = false ;

        for (int i = 0 ; i < nthreads_trials ; i++)
        {
            fprintf (stderr, "\nCSC ================ # threads: %d\n", nt [i]) ;
            printf ("\nCSC ================ # threads: %d\n", nt [i]) ;
            set_nthreads (nt [i]) ;

            if (nrows == ncols)
            {
            benchmark_spmmd ("C=A'*R     ", A, Right, true, true) ;
            benchmark_spmmd ("C=A'*R     ", A, Right, true, false) ;
            }
            benchmark_spmmd ("C=A*R      ", A, Right, false, true) ;
            benchmark_spmmd ("C=A*R      ", A, Right, false, false) ;

        }
        GrB_free (&A) ;
        GrB_free (&Right) ;
    }
#endif

    //--------------------------------------------------------------------------
    // free everything and finalize LAGraph
    //--------------------------------------------------------------------------

    set_nthreads (nthreads_default) ;
    GrB_free (&S) ;
    GrB_free (&D) ;
    finish_LAGraph ( ) ;
}

