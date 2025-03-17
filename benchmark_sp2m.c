// benchmark MKL sparse and SuiteSparse:GraphBLAS

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

void benchmark_sp2m    // C = A*B, A'*B, A*B', A'*B'
(
    char *what,
    GrB_Matrix A,
    GrB_Matrix B,
    bool A_transpose,
    bool B_transpose,
    bool C_is_csr,
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

    GET_MATRIX_PROPERTIES (B, B_format, B_is_csr, bnrows, bncols, bnvals, btype,
        ignore1, ignore2, ignore3, ignore4, ignore5, ignore6) ;

    CHECK (atype == btype) ;

    printf ("A_is_csr: %d B_is_csr: %d\n", A_is_csr, B_is_csr) ;

    //--------------------------------------------------------------------------
    // create the C matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, C0 = NULL ;
    GrB_Index cnrows = A_transpose ? ancols : anrows ;
    GrB_Index cncols = B_transpose ? bnrows : bncols ;

    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_wait (B, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // determine the transpose operators of A and B
    //--------------------------------------------------------------------------

    GrB_Descriptor desc ;
    sparse_operation_t opA, opB ;
    if (A_transpose)
    {
        if (B_transpose)
        {
            // C = A'*B'
            desc = GrB_DESC_T0T1 ;
            opA = SPARSE_OPERATION_TRANSPOSE ;
            opB = SPARSE_OPERATION_TRANSPOSE ;
        }
        else
        {
            // C = A'*B
            desc = GrB_DESC_T0 ;
            opA = SPARSE_OPERATION_TRANSPOSE ;
            opB = SPARSE_OPERATION_NON_TRANSPOSE ;
        }
    }
    else
    {
        if (B_transpose)
        {
            // C = A*B'
            desc = GrB_DESC_T1 ;
            opA = SPARSE_OPERATION_NON_TRANSPOSE ;
            opB = SPARSE_OPERATION_TRANSPOSE ;
        }
        else
        {
            // C = A*B
            desc = NULL ;
            opA = SPARSE_OPERATION_NON_TRANSPOSE ;
            opB = SPARSE_OPERATION_NON_TRANSPOSE ;
        }
    }

    GxB_Format_Value C_format = (C_is_csr) ? GxB_BY_ROW : GxB_BY_COL ;

    //--------------------------------------------------------------------------
    // benchmark GraphBLAS
    //--------------------------------------------------------------------------

    #define TRIAL_SETUP                                             \
    {                                                               \
        /* this is not required, but it makes the computation */    \
        /* similar to what mkl_sparse_sp2m is doing */              \
        OK (GrB_free (&C)) ;                                        \
        OK (GrB_Matrix_new (&C, atype, cnrows, cncols)) ;           \
        OK (GxB_set (C, GxB_FORMAT, C_format)) ;                    \
    }
    #define TRIAL_FINALIZE ;

    #define WARMUP_SETUP                                            \
    {                                                               \
        OK (GxB_set (GxB_BURBLE, true)) ;                           \
        TRIAL_SETUP ;                                               \
    }
    #define WARMUP_FINALIZE OK (GxB_set (GxB_BURBLE, false)) ;

    char title [1024] ;
    timing_t tgrb ;

    // C = A*B, A'*B, A*B', or A'*B' using GraphBLAS
    sprintf (title, "GrB_mxm: C=%s*%s where A is sparse %s, B is sparse %s",
        A_transpose ? "A'" : "A", B_transpose ? "B'" : "B",
        A_is_csr ? "CSR" : "CSC", B_is_csr ? "CSR" : "CSC") ;
    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        /* C = A*B */                                               \
        OK (GrB_mxm (C, NULL, NULL, plus_times, A, B, desc)) ;      \
        /* ensure the work is finished and C is sorted */           \
        if (ensure_sorted) { OK (GrB_wait (C, GrB_MATERIALIZE)) ; } \
    }
    BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;

    // GxB_print (C,2) ;

    printf ("GrB time %g (%g trials)\n", tgrb.tmedian, (double) tgrb.ntrials) ;

    //--------------------------------------------------------------------------
    // unpack A and B for MKL
    //--------------------------------------------------------------------------

    struct matrix_descr A_mkl_descr, B_mkl_descr ;
    sparse_matrix_t A_mkl = NULL, B_mkl = NULL, C_mkl = NULL ;

    double tmove_A = omp_get_wtime ( ) ;
    move_GrB_Matrix_to_mkl (A, false, &A_mkl, &A_mkl_descr) ;
    tmove_A = omp_get_wtime ( ) - tmove_A ;
    double tmove_B = omp_get_wtime ( ) ;
    move_GrB_Matrix_to_mkl (B, false, &B_mkl, &B_mkl_descr) ;
    tmove_B = omp_get_wtime ( ) - tmove_B ;
    printf ("move A: %g move B: %g\n", tmove_A, tmove_B) ;

    //--------------------------------------------------------------------------
    // benchmark MKL
    //--------------------------------------------------------------------------

    GrB_Matrix E = NULL ;
    timing_t tmkl ;

    // use the same # of trials as GrB, so results will be the same

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

    sprintf (title, "mkl_sparse_sp2m: C=%s*%s where A is sparse %s,"
        " B is sparse %s",
        A_transpose ? "A'" : "A", B_transpose ? "B'" : "B",
        A_is_csr ? "CSR" : "CSC", B_is_csr ? "CSR" : "CSC") ;

    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        /* C = A*B, A'*B, A*B', or A'*B' */                         \
        OK (mkl_sparse_sp2m (                                       \
            opA, A_mkl_descr, A_mkl,                                \
            opB, B_mkl_descr, B_mkl,                                \
            SPARSE_STAGE_FULL_MULT, &C_mkl)) ;                      \
        /* ensure the work is finished and C is sorted */           \
        if (ensure_sorted) { OK (mkl_sparse_order (C_mkl)) ; }      \
    }
    BENCHMARK (title, tmkl, warmup, tgrb.ntrials, tgrb.ntrials, 1e9, report) ;

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

    fprintf (stderr, "C%s=(A%s%s)*(B%s%s)%s(%c)",
        C_is_csr ? "_" : "|",
        A_is_csr ? "_" : "|", A_transpose ? "'" : " ",
        B_is_csr ? "_" : "|", B_transpose ? "'" : " ",
        ensure_sorted ? "sort" : "    ", c) ;

     printf (        "C%s=(A%s%s)*(B%s%s)%s(%c)",
        C_is_csr ? "_" : "|",
        A_is_csr ? "_" : "|", A_transpose ? "'" : " ",
        B_is_csr ? "_" : "|", B_transpose ? "'" : " ",
        ensure_sorted ? "sort" : "    ", c) ;

    print_results (tmkl, tgrb) ;

    //--------------------------------------------------------------------------
    // compare C_mkl and C
    //--------------------------------------------------------------------------

    // GxB_print (C,3) ;
    bool jumbled = !ensure_sorted ;

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
    mkl_sparse_destroy (A_mkl) ;   // frees just the header
    mkl_sparse_destroy (B_mkl) ;   // frees just the header

    // GxB_print (A,2) ;
    // GxB_print (B,2) ;

    GrB_free (&E) ;
    GrB_free (&C) ;
}

