// benchmark SuiteSparse:GraphBLAS iterator

// usage:
//      test_iter filename.lagraph

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
    GrB_Matrix S = NULL, A = NULL, D = NULL, T = NULL, B = NULL, C = NULL ;
    load_matrix ("mv and mm", filename, &S, &D) ;
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
    set_nthreads (nthreads_default) ;

    // for (int k = 0 ; k < 4 ; k++)    // FIXME: run all types
    for (int k = 0 ; k <= 0 ; k++)    // FIXME: just run GrB_FP64
    {
        // run some benchmarks with A as a sparse CSR matrix of type [k]
        convert_matrix_to_type (&A, type [k], D, S) ;

        GrB_Index nvals ;
        GrB_Matrix_nvals (&nvals, A) ;
        GrB_Index nvals2 = nvals / 10 ;

        // try the iterator

        printf ("\nA -----------------------------------------------\n") ;
        // GxB_print (A, 2) ;
        OK (GxB_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
        benchmark_iter ("y=A*x (sparse)", A) ;
        OK (GxB_set (A, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
        benchmark_iter ("y=A*x (hyper) ", A) ;
        GrB_free (&A) ;

        // create a nrows-by-ncols matrix B with nvals(A)/10 entries
        double d = nvals2 / ((double) nrows * ((double) ncols)) ;
        OK (LAGraph_Random_Matrix (&B, GrB_FP64, nrows, ncols, d, 1, msg)) ;
        OK (GxB_set (B, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
        benchmark_iter ("y=B*x (sparse)", B) ;
        OK (GxB_set (B, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
        benchmark_iter ("y=B*x (hyper) ", B) ;
        GrB_free (&B) ;

        // create a nrows-by-ncols matrix C with about 1000 entries
        d = 1000 / ((double) nrows * ((double) ncols)) ;
        OK (LAGraph_Random_Matrix (&C, GrB_FP64, nrows, ncols, d, 1, msg)) ;
        OK (GxB_set (C, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
        benchmark_iter ("y=C*x (sparse)", C) ;
        OK (GxB_set (C, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
        benchmark_iter ("y=C*x (hyper) ", C) ;
        GrB_free (&C) ;
    }

    //--------------------------------------------------------------------------
    // convert S to sparse CSC
    //--------------------------------------------------------------------------

#if 0
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

        for (int i = 0 ; i < nthreads_trials ; i++)
        {
            fprintf (stderr, "\nCSC ================ # threads: %d\n", nt [i]) ;
            printf ("\nCSC ================ # threads: %d\n", nt [i]) ;
            set_nthreads (nt [i]) ;

            // sparse times dense vector (GrB uses saxpy4 method)
            // displayed as y ... S|*x
#if 1
            benchmark_mv ("y+=A*x     ", A, 1, 1, false) ;     // y += A*x
#endif
#if 0
            benchmark_mv ("y=A*x      ", A, 1, 0, false) ;     // y = A*x
            benchmark_mv ("y=a*A*x    ", A, 0.5, 0, false) ;   // y = 0.5*A*x
            benchmark_mv ("y=a*A*x+b*y", A, 0.5, 0.2, false) ; // y = a*A*x+b*y
#endif
#if 1
            // sparse times dense vector (MKL gets hints)
            benchmark_mv ("y+=A*x(opt)", A, 1, 1, true) ;
#endif

            // sparse times dense matrix
            // A is CSC, C and B are by column (GrB uses saxpy4 method)
//          fprintf (stderr, "\n") ;
            // displayed as C+=S|*B (|n)
#if 1
            benchmark_mm ("C+=A*B     ", A, 1, 1, 4, true, false) ;
            benchmark_mm ("C+=A*B(opt)", A, 1, 1, 4, true, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 8, true, false) ;
            benchmark_mm ("C+=A*B(opt)", A, 1, 1, 8, true, true) ;
#endif

#if 0
            benchmark_mm ("C=A*B      ", A, 1, 0, 4, true) ;         // C = A*B
            benchmark_mm ("C=a*A*B    ", A, 0.5, 0, 4, true) ;       // C = 0.5*A*B
            benchmark_mm ("C=a*A*B+b*C", A, 0.5, 0.2, 4, true) ;     // C = a*A*B+b*C
#endif
#if 0
            benchmark_mm ("C+=A*B     ", A, 1, 1, 1, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 2, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 4, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 5, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 8, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 16, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 17, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 32, true) ;
#endif

            // A is CSC, C and B are by row (GrB varies by matrix.  It must
            // explicitly transpose the sparse matrix A or the dense matrix B,
            // and then use either bitmap saxpy if transposing A (which does
            // not exploit the accumulator but is still pretty fast) or saxpy5
            // if transposing B.  It usually decides to transpose B then use
            // saxpy5.
            // displayed as C+=S|*B (_n)
//          fprintf (stderr, "\n") ;
#if 1
            benchmark_mm ("C+=A*B     ", A, 1, 1, 4, false, false) ;
            benchmark_mm ("C+=A*B(opt)", A, 1, 1, 4, false, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 8, false, false) ;
            benchmark_mm ("C+=A*B(opt)", A, 1, 1, 8, false, true) ;
#endif
#if 0
            benchmark_mm ("C=A*B      ", A, 1, 0, 4, false) ;        // C = A*B
            benchmark_mm ("C=a*A*B    ", A, 0.5, 0, 4, false) ;      // C = 0.5*A*B
            benchmark_mm ("C=a*A*B+b*C", A, 0.5, 0.2, 4, false) ;    // C = a*A*B+b*C
#endif
#if 0
            benchmark_mm ("C+=A*B     ", A, 1, 1, 1, false) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 2, false) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 4, false) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 5, false) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 8, false) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 16, false) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 17, false) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 32, false) ;
#endif
#if 0
            for (int ncol = 1 ; ncol <= 17 ; ncol++)
            {
                benchmark_mm ("C+=A*B     ", A, 1, 1, ncol, false) ;
            }
            benchmark_mm ("C+=A*B     ", A, 1, 1, 32, false) ;
#endif

        }
        GrB_free (&A) ;
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

