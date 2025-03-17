// benchmark MKL sparse and SuiteSparse:GraphBLAS: syrk

#include "mklgrb.h"

// benchmark GrB for 3 to 100 trials, but no more than 5.0 seconds.

/*

#define MAXTRIALS 1
#define MINTRIALS 1
#define MAXTIME 0.1

*/

#define MAXTRIALS 100
#define MINTRIALS 3
#define MAXTIME 5.0

void benchmark_syrk   // C = A'*A or A*A'
(
    char *what,
    GrB_Matrix A,
    bool A_transpose,   // if true, compute C=A'*A, otherwise C = A*A'
    bool ensure_sorted  // if true, ensure C is sorted on output, and include
                        // this in the benchmarked run time
)
{

    bool report = true ;    // if true, report the summary of the trials
    bool warmup = true ;    // if true, do a warmup first

    //--------------------------------------------------------------------------
    // get the properties of A and its operators
    //--------------------------------------------------------------------------

    // GxB_print (A, 2) ;
    GET_MATRIX_PROPERTIES (A, A_format, A_is_csr, anrows, ancols, anvals, atype,
        abs, plus, minus, times, plus_monoid, plus_times) ;

    CHECK (A_is_csr) ;

    //--------------------------------------------------------------------------
    // create the C matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, C0 = NULL, T = NULL ;
    GrB_Index cnrows = A_transpose ? ancols : anrows ;
    GrB_Index cncols = cnrows ;

    OK (GrB_wait (A, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // determine the transpose operators of A
    //--------------------------------------------------------------------------

    GrB_Descriptor desc ;
    sparse_operation_t opA ;
    if (A_transpose)
    {
        // C = A'*A
        opA = SPARSE_OPERATION_TRANSPOSE ;
        desc = GrB_DESC_T0 ;
    }
    else
    {
        // C = A*A'
        opA = SPARSE_OPERATION_NON_TRANSPOSE ;
        desc = GrB_DESC_T1 ;
    }

    GxB_Format_Value C_format = GxB_BY_ROW ;

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

    // C = A'*A or A'*A using GraphBLAS
    if (A_transpose)
    {
        sprintf (title, "C=A'*A") ;
    }
    else
    {
        sprintf (title, "C=A*A'") ;
    }
    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        OK (GrB_free (&C)) ;                                        \
        OK (GrB_Matrix_new (&C, atype, cncols, cncols)) ;           \
        OK (GrB_mxm (C, NULL, NULL, plus_times, A, A, desc)) ;      \
        /* ensure the work is finished and C is sorted */           \
        if (ensure_sorted) { OK (GrB_wait (C, GrB_MATERIALIZE)) ; } \
    }
    BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;

    GxB_print (C,2) ;

    printf ("GrB time %g (%g trials)\n", tgrb.tmedian, (double) tgrb.ntrials) ;

    //--------------------------------------------------------------------------
    // unpack A for MKL
    //--------------------------------------------------------------------------

    struct matrix_descr A_mkl_descr ;
    sparse_matrix_t A_mkl = NULL, C_mkl = NULL ;

    double tmove_A = omp_get_wtime ( ) ;
    move_GrB_Matrix_to_mkl (A, false, &A_mkl, &A_mkl_descr) ;
    tmove_A = omp_get_wtime ( ) - tmove_A ;

    printf ("move A: %g\n", tmove_A) ;

    //--------------------------------------------------------------------------
    // benchmark MKL
    //--------------------------------------------------------------------------

    GrB_Matrix E = NULL ;
    timing_t tmkl ;

    #undef  WARMUP_SETUP
    #define WARMUP_SETUP    ;
    #undef  WARMUP_FINALIZE
    #define WARMUP_FINALIZE ;
    #undef  TRIAL_SETUP
    #define TRIAL_SETUP                                             \
    {                                                               \
        if (C_mkl != NULL) { OK (mkl_sparse_destroy (C_mkl)) ; }    \
        C_mkl = NULL ;                                              \
    }
    #undef  TRIAL_FINALIZE
    #define TRIAL_FINALIZE ;

    sprintf (title, "mkl_sparse_syrk: C=A%s*A%s ",
        A_transpose ? "'" : "",
        A_transpose ? "" : "'") ;

    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        /* C = A*A' or A'*A */                                      \
        OK (mkl_sparse_syrk (opA, A_mkl, &C_mkl)) ;                 \
        /* ensure the work is finished and C is sorted */           \
        if (ensure_sorted) { OK (mkl_sparse_order (C_mkl)) ; }      \
    }
    BENCHMARK (title, tmkl, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;

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

    fprintf (stderr, "C=(A%s)*(A%s)   %s(%c)",
        A_transpose ? "'" : " ",
        A_transpose ? " " : "'",
        ensure_sorted ? "sort" : "    ", c) ;

     printf (        "C=(A%s)*(A%s)   %s(%c)",
        A_transpose ? "'" : " ",
        A_transpose ? " " : "'",
        ensure_sorted ? "sort" : "    ", c) ;

    print_results (tmkl, tgrb) ;

    //--------------------------------------------------------------------------
    // compare C_mkl and C
    //--------------------------------------------------------------------------

    bool jumbled = !ensure_sorted ;

    // C = triu (C)
    // OK (GxB_print (C,3)) ;
    OK (GrB_select (C, NULL, NULL, GrB_TRIU, C, 0, NULL)) ;

    OK (GrB_Matrix_new (&C0, atype, cnrows, cncols)) ;
    OK (GxB_set (C0, GxB_FORMAT, C_format)) ;
    double tmove_C = omp_get_wtime ( ) ;
    move_mkl_to_GrB_Matrix (C0, false, &C_mkl, true, jumbled) ;  // break opacity
    tmove_C = omp_get_wtime ( ) - tmove_C ;
    printf ("move C: %g\n", tmove_C) ;

    // the C_mkl matrix is now imported into the shallow GrB_Matrix C0 
    OK (GxB_print (C0,1)) ;

    // E = abs (C-C0)
    OK (GrB_Matrix_new (&E, atype, cnrows, cncols)) ;
    OK (GrB_eWiseAdd (E, NULL, NULL, minus, C, C0, NULL)) ;
    // OK (GxB_print (E,3)) ;
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

    //--------------------------------------------------------------------------
    // break opacity
    //--------------------------------------------------------------------------

    // MKL owns the C_mkl matrix and the shallow content of C0, so destroy it;
    // C0 is also freed.  The content of these two matrices is identical, but
    // C0->[pix]_shallow are all true, so the double free is avoided.
    mkl_sparse_destroy (C_mkl) ;   // frees all content
    GrB_free (&C0) ;               // does not free shallow content owned by MKL

    //--------------------------------------------------------------------------
    // pack the MKL sparse matrices A into the GrB_Matrices A
    //--------------------------------------------------------------------------

    // move not needed since MKL does not take ownership of A
    mkl_sparse_destroy (A_mkl) ;   // frees just the header

    // GxB_print (A,2) ;

    GrB_free (&E) ;
    GrB_free (&C) ;
}

