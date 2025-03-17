// benchmark MKL sparse and SuiteSparse:GraphBLAS

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

void benchmark_sypr   // C = A'*B*A or A*B*A'
(
    char *what,
    GrB_Matrix A,
    GrB_Matrix B,       // must be symmetric (condition not checked)
    bool A_transpose,   // if true, compute C=A'*B*A, otherwise C = A*B*A'
    GrB_Index *perm,    // if non-NULL, A is a permutation matrix and
                        // perm is the index vector,
                        // so that C=B(perm,perm)=A*B*A'
    bool ensure_sorted  // if true, ensure C is sorted on output, and include
                        // this in the benchmarked run time
)
{

    bool report = true ;    // if true, report the summary of the trials
    bool warmup = true ;    // if true, do a warmup first

    //--------------------------------------------------------------------------
    // get the properties of A and B and their operators
    //--------------------------------------------------------------------------

    // GxB_print (A, 2) ;
    GET_MATRIX_PROPERTIES (A, A_format, A_is_csr, anrows, ancols, anvals, atype,
        abs, plus, minus, times, plus_monoid, plus_times) ;

    // GxB_print (B, 2) ;
    GET_MATRIX_PROPERTIES (B, B_format, B_is_csr, bnrows, bncols, bnvals, btype,
        ignore1, ignore2, ignore3, ignore4, ignore5, ignore6) ;

    CHECK (atype == btype) ;
    CHECK (A_is_csr && B_is_csr) ;
    CHECK (atype == GrB_FP64) ;     // TODO: can easily extend to other types 

    //--------------------------------------------------------------------------
    // create the C matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, C0 = NULL, T = NULL ;
    GrB_Index cnrows = A_transpose ? ancols : anrows ;
    GrB_Index cncols = cnrows ;

    OK (GrB_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_wait (B, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // determine the transpose operators of A
    //--------------------------------------------------------------------------

    sparse_operation_t opA ;
    if (A_transpose)
    {
        // C = A'*B*A
        opA = SPARSE_OPERATION_TRANSPOSE ;
    }
    else
    {
        // C = A*B*A'
        opA = SPARSE_OPERATION_NON_TRANSPOSE ;
    }

    GxB_Format_Value C_format = GxB_BY_ROW ;

    //--------------------------------------------------------------------------
    // benchmark GraphBLAS
    //--------------------------------------------------------------------------

    #define TRIAL_SETUP ;
    {                                                               \
        /* this is not required, but it makes the computation */    \
        /* similar to what mkl_sparse_sypr is doing */              \
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

    // assume nnz(B) >> nnz(A).  Otherwise, the order of the
    // computations below should be reversed.

    if (A_transpose)
    {
        // C = A'*B*A using GraphBLAS
        sprintf (title, "C=A'*B*A") ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* T = A'*B */                                              \
            OK (GrB_free (&C)) ;                                        \
            OK (GrB_Matrix_new (&C, atype, ancols, ancols)) ;           \
            OK (GrB_free (&T)) ;                                        \
            OK (GrB_Matrix_new (&T, atype, ancols, bnrows)) ;           \
            /* OK (GxB_print (T, 2)) ; */ \
            /* OK (GxB_print (A, 2)) ; */ \
            /* OK (GxB_print (B, 2)) ; */ \
            OK (GrB_mxm (T, NULL, NULL, plus_times, A, B, GrB_DESC_T0)) ;   \
            /* C = T*A */                                               \
            OK (GrB_mxm (C, NULL, NULL, plus_times, T, A, NULL)) ;      \
            OK (GrB_free (&T)) ;                                        \
            /* ensure the work is finished and C is sorted */           \
            if (ensure_sorted) { OK (GrB_wait (C, GrB_MATERIALIZE)) ; } \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else
    {
        // C = A*B*A' using GraphBLAS
        sprintf (title, "C=A*B*A'") ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* T = A*B */                                               \
            OK (GrB_free (&C)) ;                                        \
            OK (GrB_Matrix_new (&C, atype, anrows, anrows)) ;           \
            OK (GrB_free (&T)) ;                                        \
            OK (GrB_Matrix_new (&T, atype, anrows, bncols)) ;           \
            OK (GrB_mxm (T, NULL, NULL, plus_times, A, B, NULL)) ;      \
            /* C = T*A' */                                              \
            OK (GrB_mxm (C, NULL, NULL, plus_times, T, A, GrB_DESC_T1)) ;   \
            OK (GrB_free (&T)) ;                                        \
            /* ensure the work is finished and C is sorted */           \
            if (ensure_sorted) { OK (GrB_wait (C, GrB_MATERIALIZE)) ; } \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }

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
    B_mkl_descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC ;
    B_mkl_descr.mode = SPARSE_FILL_MODE_LOWER ;
    printf ("move A: %g move B: %g\n", tmove_A, tmove_B) ;

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

    sprintf (title, "mkl_sparse_sypr: C=A%s*B*A%s ",
        A_transpose ? "'" : "",
        A_transpose ? "" : "'") ;

    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        /* C = A*B*A' or A'*B*A */                                  \
        OK (mkl_sparse_sypr (                                       \
            opA, A_mkl, B_mkl, B_mkl_descr, &C_mkl,                 \
            SPARSE_STAGE_FULL_MULT)) ;                              \
        /* ensure the work is finished and C is sorted */           \
        if (ensure_sorted) { OK (mkl_sparse_order (C_mkl)) ; }      \
    }
    BENCHMARK (title, tmkl, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;

    printf ("MKL time %g (%g trials),\n", tmkl.tmedian, (double) tmkl.ntrials) ;

    int nthreads ;
    OK (GxB_get (GxB_NTHREADS, &nthreads)) ;
    fprintf (stderr, "%s %2d:", what, nthreads) ;
     printf (        "%s %2d:", what, nthreads) ;

    char c = 'd' ;
    fprintf (stderr, "C=(A%s)*(B)*(A%s)  %s(%c)",
        A_transpose ? "'" : " ",
        A_transpose ? " " : "'",
        ensure_sorted ? "sort" : "    ", c) ;

     printf (        "C=(A%s)*(B)*(A%s)  %s(%c)",
        A_transpose ? "'" : " ",
        A_transpose ? " " : "'",
        ensure_sorted ? "sort" : "    ", c) ;

    print_results (tmkl, tgrb) ;

    //--------------------------------------------------------------------------
    // compare C_mkl and C
    //--------------------------------------------------------------------------

    // GxB_print (C,3) ;
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
    // pack the MKL sparse matrices A and B into the GrB_Matrices A and B
    //--------------------------------------------------------------------------

    // move not needed since MKL does not take ownership of A and B
    mkl_sparse_destroy (A_mkl) ;   // frees just the header
    mkl_sparse_destroy (B_mkl) ;   // frees just the header

    //--------------------------------------------------------------------------
    // compare with GrB_extract
    //--------------------------------------------------------------------------

    if (perm != NULL && !A_transpose)
    {
        // This is "cheating", since in general it would be difficult to from
        // the problem statement C=A*B*A' to tell that A is a permutation
        // matrix.  However, if A is known to be a permutation matrix already,
        // then this comparison is valid.

        timing_t tgrb_extract ;
        // GxB_print (A, 3) ;

        GrB_Matrix C2 = NULL ;
        GrB_Index n = cncols ;
        OK (GrB_Matrix_new (&C2, atype, cncols, cncols)) ;

        #undef  WARMUP_SETUP
        #define WARMUP_SETUP                                            \
        {                                                               \
            OK (GxB_set (GxB_BURBLE, true)) ;                           \
            TRIAL_SETUP ;                                               \
        }
        #undef  WARMUP_FINALIZE
        #define WARMUP_FINALIZE OK (GxB_set (GxB_BURBLE, false)) ;

        #undef  TRIAL_SETUP
        #define TRIAL_SETUP ;

        #undef  CODE
        #define CODE                                                        \
        {                                                                   \
            /* C2 = B (perm,perm), another way to compute C=A*B*A' */       \
            OK (GrB_extract (C2, NULL, NULL, B, perm, n, perm, n, NULL)) ;  \
        }
        BENCHMARK (title, tgrb_extract, warmup, MINTRIALS, MAXTRIALS, MAXTIME,
            report) ;

        // C2 = triu (C2) since this has been done to C above
        OK (GrB_select (C2, NULL, NULL, GrB_TRIU, C2, 0, NULL)) ;

        // check results
        // E = max (abs (C-C2))
        OK (GrB_eWiseAdd (E, NULL, NULL, minus, C, C2, NULL)) ;
        OK (GrB_apply (E, NULL, NULL, abs, E, NULL)) ;
        OK (GrB_reduce (&err, NULL, GrB_MAX_MONOID_FP64, E, NULL)) ;
        double tt = tgrb_extract.tmedian ;

        fprintf (stderr, "GrB_extract C=S(p,p) instead of C=P*S*P': "
            "                                "
            "   %10.4f \n", tt) ;
         printf (        "GrB_extract C=S(p,p) instead of C=P*S*P': "
            "                                "
            "   %10.4f \n", tt) ;
        printf ("GrB_extract err %g\n", err) ;
        GrB_free (&C2) ;
    }

    // GxB_print (A,2) ;
    // GxB_print (B,2) ;

    GrB_free (&E) ;
    GrB_free (&C) ;
}

