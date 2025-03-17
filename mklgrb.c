// benchmark MKL sparse and SuiteSparse:GraphBLAS: mv and mm

// usage:
//      mklgrb filename.lagraph

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
    GrB_Matrix S = NULL, A = NULL, D = NULL ;
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

#if 1
    // for (int k = 0 ; k < 4 ; k++)    // FIXME: run all types
    for (int k = 0 ; k <= 0 ; k++)    // FIXME: just run GrB_FP64
    {
        // run some benchmarks with A as a sparse CSR matrix of type [k]
        convert_matrix_to_type (&A, type [k], D, S) ;

        if (just_practicing)
        {
            set_nthreads (nthreads_default) ;
            benchmark_mv ("A*x warmup ", A, 1, 1, false) ;
            just_practicing = false ;
        }

        for (int i = 0 ; i < nthreads_trials ; i++)
        {
            fprintf (stderr, "\nCSR ================ # threads: %d\n", nt [i]) ;
            printf ("\nCSR ================ # threads: %d\n", nt [i]) ;
            set_nthreads (nt [i]) ;

#if 1

            // sparse times dense vector (GrB uses dot4 method)
            // displayed as y ... S|*x.  Do it twice since MKL seems
            // to have some initialization time.
#if 1
            benchmark_mv ("y+=A*x     ", A, 1, 1, false) ;
#endif
#if 0
            benchmark_mv ("y=A*x      ", A, 1, 0, false) ;     // y = A*x
            benchmark_mv ("y=a*A*x    ", A, 0.5, 0, false) ;   // y = 0.5*A*x
            benchmark_mv ("y=a*A*x+b*y", A, 0.5, 0.2, false) ; // y = a*A*x+b*y
#endif
#if 1
            // sparse times dense vector (MKL gets hints)
            benchmark_mv ("y+=A*x(opt)", A, 1, 1, true) ;     // y += A*x
#endif

            // sparse times dense matrix (GrB uses dot4 method)
            // A is CSR, C and B are by column
            // displayed as C+=S_*B (|n)

//          fprintf (stderr, "\n") ;
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

            // A is CSR, C and B are by row (GrB uses saxpy5 method)
            // displayed as C+=S_*B (_n)
//          fprintf (stderr, "\n") ;
#if 1

            benchmark_mm ("C+=A*B     ", A, 1, 1, 4, false, false) ;
            benchmark_mm ("C+=A*B(opt)", A, 1, 1, 4, false, true) ;
            benchmark_mm ("C+=A*B     ", A, 1, 1, 8, false, false) ;
            benchmark_mm ("C+=A*B(opt)", A, 1, 1, 8, false, true) ;
#endif
#if 0
            benchmark_mm ("C+=A*B     ", A, 1, 0, 4, false) ;        // C = A*B
            benchmark_mm ("C+=A*B     ", A, 0.5, 0, 4, false) ;      // C = 0.5*A*B
            benchmark_mm ("C+=A*B     ", A, 0.5, 0.2, 4, false) ;    // C = a*A*B+b*C
#endif
#if 1
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
            for (int ncol = 1 ; ncol <= 33 ; ncol++)
            {
                benchmark_mm ("C+=A*B     ", A, 1, 1, ncol, false) ;
            }
            // benchmark_mm ("C+=A*B     ", A, 1, 1, 32, false) ;
#endif

#endif

        }
        GrB_free (&A) ;
    }
#endif

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

