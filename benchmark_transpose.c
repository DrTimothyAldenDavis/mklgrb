// benchmark MKL sparse and SuiteSparse:GraphBLAS: transpose

#include "mklgrb.h"

// benchmark GrB for 3 to 100 trials, but no more than 5.0 seconds.
// Once GrB is benchmarked, MKL uses the same number of trials

/*

#define MAXTRIALS 1
#define MINTRIALS 1
#define MAXTIME 0.1

*/

#define MAXTRIALS 100
#define MINTRIALS 3
#define MAXTIME 5.0

void benchmark_transpose    // C = A'
(
    char *what,
    GrB_Matrix A,
    bool ensure_sorted  // if true, ensure C is sorted on output, and include
                        // this in the benchmarked run time
)
{

    bool report = true ;    // if true, report the summary of the trials
    bool warmup = true ;    // if true, do a warmup first

    //--------------------------------------------------------------------------
    // get the properties of A and B and their operators
    //--------------------------------------------------------------------------

    GET_MATRIX_PROPERTIES (A, A_format, A_is_csr, anrows, ancols, anvals, atype,
        abs, plus, minus, times, plus_monoid, plus_times) ;

    // only CSR input matrices are supported
    printf ("A_is_csr: %d\n", A_is_csr) ;
    CHECK (A_is_csr) ;

    //--------------------------------------------------------------------------
    // create the C matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, C0 = NULL ;
    GrB_Index cnrows = ancols ;
    GrB_Index cncols = anrows ;

    OK (GrB_wait (A, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // benchmark GraphBLAS
    //--------------------------------------------------------------------------

    #define TRIAL_SETUP ;
    #define TRIAL_FINALIZE ;

    #define WARMUP_SETUP                                            \
    {                                                               \
        OK (GxB_set (GxB_BURBLE, true)) ;                           \
    }
    #define WARMUP_FINALIZE OK (GxB_set (GxB_BURBLE, false)) ;

    char title [1024] ;
    timing_t tgrb ;

    // C = A' using GraphBLAS

    sprintf (title, "GrB_transpose: C=A'") ;
    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        /* C = A' */                                                \
        OK (GrB_free (&C)) ;                                        \
        OK (GrB_Matrix_new (&C, atype, cnrows, cncols)) ;           \
        OK (GrB_transpose (C, NULL, NULL, A, NULL)) ;               \
        /* ensure the work is finished and C is sorted */           \
        if (ensure_sorted) { OK (GrB_wait (C, GrB_MATERIALIZE)) ; } \
    }
    BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;

    // should be the same as the trial, so zero it out:
    printf ("GrB warmup %g\n", tgrb.twarmup) ;
    tgrb.twarmup = 0 ;

    printf ("GrB time %g (%g trials)\n", tgrb.tmedian, (double) tgrb.ntrials) ;

    //--------------------------------------------------------------------------
    // unpack A for MKL
    //--------------------------------------------------------------------------

    timing_t tmkl ;
    memset (&tmkl, 0, sizeof (tmkl)) ;
    GrB_Matrix E = NULL ;

    struct matrix_descr A_mkl_descr ;
    sparse_matrix_t A_mkl = NULL, C_mkl = NULL ;

    double tmove_A = omp_get_wtime ( ) ;
    tmove_A = omp_get_wtime ( ) - tmove_A ;
    printf ("move A: %g\n", tmove_A) ;

    //--------------------------------------------------------------------------
    // benchmark MKL
    //--------------------------------------------------------------------------

    // mkl_sparse_convert_csr does the transpose and then saves the transpose
    // in the input matrix, so the 2nd trial would be very fast.  As a result,
    // the MKL matrix A_mkl must be recreated each time.

    #undef  TRIAL_SETUP
    #define TRIAL_SETUP ;
    #undef  TRIAL_FINALIZE
    #define TRIAL_FINALIZE ;

    #undef  WARMUP_SETUP
    #define WARMUP_SETUP ;
    #undef  WARMUP_FINALIZE
    #define WARMUP_FINALIZE ;

    sprintf (title, "mkl_sparse_convert_csr: C=A'") ;
    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        /* create A */                                              \
        if (A_mkl != NULL) mkl_sparse_destroy (A_mkl) ;             \
        move_GrB_Matrix_to_mkl (A, false, &A_mkl, &A_mkl_descr) ;   \
        /* C = A' */                                                \
        if (C_mkl != NULL) { OK (mkl_sparse_destroy (C_mkl)) ; }    \
        C_mkl = NULL ;                                              \
        OK (mkl_sparse_convert_csr (A_mkl, SPARSE_OPERATION_TRANSPOSE, \
            &C_mkl)) ;                                              \
        /* ensure the work is finished and C is sorted */           \
        if (ensure_sorted) { OK (mkl_sparse_order (C_mkl)) ; }      \
    }
    BENCHMARK (title, tmkl, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;

    // should be the same as the trial, so zero it out:
    printf ("MKL warmup %g\n", tmkl.twarmup) ;
    tmkl.twarmup = 0 ;

    printf ("MKL time %g (%g trials),\n", tmkl.tmedian, (double) tmkl.ntrials) ;

    int nthreads ;
    OK (GxB_get (GxB_NTHREADS, &nthreads)) ;
    fprintf (stderr, "%s %2d:", what, nthreads) ;
     printf (        "%s %2d:", what, nthreads) ;

    char c ;
    if (atype == GrB_FP32) c = 's' ;
    else if (atype == GrB_FP64) c = 'd' ;
    else if (atype == GxB_FC32) c = 'c' ;
    else if (atype == GxB_FC64) c = 'z' ;
    else CHECK (false) ;    // unknown type

    fprintf (stderr, "C=A'         %s(%c)", ensure_sorted ? "sort" : "    ", c) ;
     printf (        "C=A'         %s(%c)", ensure_sorted ? "sort" : "    ", c) ;

    print_results (tmkl, tgrb) ;

    //--------------------------------------------------------------------------
    // compare C_mkl and C
    //--------------------------------------------------------------------------

    // GxB_print (C,3) ;
    bool jumbled = !ensure_sorted ;

    OK (GrB_Matrix_new (&C0, atype, cnrows, cncols)) ;
    OK (GxB_set (C0, GxB_FORMAT, GxB_BY_ROW)) ;
    double tmove_C = omp_get_wtime ( ) ;
    move_mkl_to_GrB_Matrix (C0, false, &C_mkl, true, jumbled) ;  // break opacity
    tmove_C = omp_get_wtime ( ) - tmove_C ;
    printf ("move C: %g\n", tmove_C) ;

    // the C_mkl matrix is now imported into the shallow GrB_Matrix C0 
    OK (GxB_print (C0,1)) ;

    // E = abs (C-C0)
    OK (GrB_Matrix_new (&E, atype, cnrows, cncols)) ;
    OK (GrB_eWiseAdd (E, NULL, NULL, minus, C, C0, NULL)) ;
    // E = abs (E)
    OK (GrB_apply (E, NULL, NULL, abs, E, NULL)) ;
    // err = max (E)
    double err, norm ;
    OK (GrB_reduce (&err, NULL, GrB_MAX_MONOID_FP64, E, NULL)) ;

    // E = abs (C), typecasting if necessary
    OK (GrB_apply (E, NULL, NULL, abs, C, NULL)) ;
    // norm = max (E)
    OK (GrB_reduce (&norm, NULL, GrB_MAX_MONOID_FP64, E, NULL)) ;
    if (norm == 0) norm = 1 ;
    printf ("max err: %g  rel err: %g\n", err, err / norm) ;
//  fprintf (stderr, "max err: %g  rel err: %g\n", err, err / norm) ;

    //--------------------------------------------------------------------------
    // break opacity
    //--------------------------------------------------------------------------

    // MKL owns the C_mkl matrix and the shallow content of C0, so destroy it;
    // C0 is also freed.  The content of these two matrices is identical, but
    // C0->[pix]_shallow are all true, so the double free is avoided.
    mkl_sparse_destroy (C_mkl) ;   // frees all content
    GrB_free (&C0) ;               // does not free shallow content owned by MKL

    //--------------------------------------------------------------------------
    // pack the MKL sparse matrices A and B into the GrB_Matrices A and B
    //--------------------------------------------------------------------------

    // move not needed since MKL does not take ownership of A and B
    if (A_mkl != NULL) mkl_sparse_destroy (A_mkl) ;   // frees just the header

    // GxB_print (A,2) ;
    // GxB_print (B,2) ;

    GrB_free (&E) ;
    GrB_free (&C) ;
}

