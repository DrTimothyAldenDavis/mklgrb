// benchmark MKL sparse and SuiteSparse:GraphBLAS

#include "mklgrb.h"

// benchmark GrB for 3 to 100 trials, but no more than 5.0 seconds.
// Once GrB is benchmarked, MKL uses the same number of trials
/*
#define MAXTRIALS 10
#define MINTRIALS 3
#define MAXTIME 0.1

*/

#define MAXTRIALS 100
#define MINTRIALS 3
#define MAXTIME 5.0

void benchmark_spmmd    // C = A*B or A'*B where C is dense, A and B sparse
(
    char *what,
    GrB_Matrix A,
    GrB_Matrix B,
    bool A_transpose,
    bool C_by_col           // if true, C is held by column;
                            // if false, it is held by row
)
{

    bool report = true ;    // if true, report the summary of the trials
    bool warmup = true ;    // if true, do a warmup first

    //--------------------------------------------------------------------------
    // get the properties of A and and B its operators
    //--------------------------------------------------------------------------

    GET_MATRIX_PROPERTIES (A, format, A_is_csr, anrows, ancols, nvals, atype,
        abs, plus, minus, times, plus_monoid, plus_times) ;

    GET_MATRIX_PROPERTIES (B, B_format, B_is_csr, bnrows, bncols, bnvals, btype,
        ignore1, ignore2, ignore3, ignore4, ignore5, ignore6) ;

    CHECK (atype == btype) ;
    printf ("A_tranpose: %d A_is_csr: %d B_is_csr: %d\n",
        A_transpose, A_is_csr, B_is_csr) ;
    CHECK (A_is_csr == B_is_csr) ;

    //--------------------------------------------------------------------------
    // create the dense test matrices C, C0
    //--------------------------------------------------------------------------

    GrB_Matrix C0, C ;
    GrB_Index cncols = bncols ;
    GrB_Index cnrows = (A_transpose) ? ancols : anrows ;
    OK (GrB_Matrix_new (&C0, atype, cnrows, cncols)) ;
    OK (GrB_Matrix_new (&C,  atype, cnrows, cncols)) ;

    sparse_layout_t layout ;
    if (C_by_col)
    {
        // C is held by column
        layout = SPARSE_LAYOUT_COLUMN_MAJOR ;
        OK (GxB_set (C0, GxB_FORMAT, GxB_BY_COL)) ;
        OK (GxB_set (C , GxB_FORMAT, GxB_BY_COL)) ;
    }
    else
    {
        // C is held by row
        layout = SPARSE_LAYOUT_ROW_MAJOR ;
        OK (GxB_set (C0, GxB_FORMAT, GxB_BY_ROW)) ;
        OK (GxB_set (C , GxB_FORMAT, GxB_BY_ROW)) ;
    }

    GrB_Descriptor desc ;
    sparse_operation_t op ;
    if (A_transpose)
    {
        op = SPARSE_OPERATION_TRANSPOSE ;
        desc = GrB_DESC_T0 ;
    }
    else
    {
        op = SPARSE_OPERATION_NON_TRANSPOSE ;
        desc = NULL ;
    }

    // ensure C0 is full and non-iso
    OK (GrB_assign (C0, NULL, NULL, 1, GrB_ALL, cnrows, GrB_ALL, cncols, NULL));
    OK (GrB_Matrix_setElement (C0, 0.2, 0, 0)) ;
    // OK (GxB_print (C, 2)) ;
    // OK (GxB_print (C0, 2)) ;
    // OK (GxB_print (A, 2)) ;
    // OK (GxB_print (B, 2)) ;

    //--------------------------------------------------------------------------
    // benchmark GraphBLAS
    //--------------------------------------------------------------------------

    #define WARMUP_SETUP    OK (GxB_set (GxB_BURBLE, true)) ;
    #define WARMUP_FINALIZE OK (GxB_set (GxB_BURBLE, false)) ;
    #define TRIAL_SETUP ;
    #define TRIAL_FINALIZE ;

    char title [1024] ;
    timing_t tgrb ;

    {
        // C = A*B using GraphBLAS, but ensure C is dense to mimic MKL
        sprintf (title, "GrB_mxm: C=A*B where A and B are %s, C dense "
            " matrix (by %s), ncols(C) is %d", A_is_csr ? "CSR" : "CSC",
            C_by_col ? "col" : "row", (int) cncols) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* C = 0 */                                                 \
            OK (GrB_assign (C, NULL, NULL, 0,                           \
                GrB_ALL, cnrows, GrB_ALL, cncols, NULL)) ;              \
            /* C += A*B or A'*B */                                      \
            OK (GrB_mxm (C, NULL, plus, plus_times, A, B, desc)) ;      \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }

    printf ("GrB time %g (%g trials)\n", tgrb.tmedian, (double) tgrb.ntrials) ;

    //--------------------------------------------------------------------------
    // unpack A and B for MKL
    //--------------------------------------------------------------------------

    GrB_Index Bx_size = 0, Cx_size = 0, Cx0_size = 0 ;
    MKL_INT ldc ;

    if (C_by_col)
    {
        // C held by column
        ldc = cnrows ;
    }
    else
    {
        // C held by row
        ldc = cncols ;
    }

    struct matrix_descr A_mkl_descr, B_mkl_descr ;
    sparse_matrix_t A_mkl = NULL, B_mkl = NULL ;

    // move_GrB_Matrix_to_mkl creates an MKL matrix that is a shallow copy
    // of the GrB_Matrix A.  The GrB_Matrix A is not affected, and MKL does not
    // modify it.
    move_GrB_Matrix_to_mkl (A, false, &A_mkl, &A_mkl_descr) ;
    move_GrB_Matrix_to_mkl (B, false, &B_mkl, &B_mkl_descr) ;

    timing_t tmkl ;

    char c ;

    //--------------------------------------------------------------------------
    // benchmark MKL
    //--------------------------------------------------------------------------

    if (atype == GrB_FP32)
    {
        c = 's' ;
        sprintf (title, "mkl_sparse_%c_spmd: ", c) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* C0 = A*B */                                              \
            OK (mkl_sparse_s_spmmd (op, A_mkl, B_mkl,                   \
                layout, (float *) C0->x, ldc)) ;                        \
        }
        BENCHMARK (title, tmkl, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else if (atype == GrB_FP64)
    {
        c = 'd' ;
        sprintf (title, "mkl_sparse_%c_spmd: ", c) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* C0 = A*B */                                              \
            OK (mkl_sparse_d_spmmd (op, A_mkl, B_mkl,                   \
                layout, (double *) C0->x, ldc)) ;                       \
        }
        BENCHMARK (title, tmkl, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else if (atype == GxB_FC32)
    {
        c = 'c' ;
        sprintf (title, "mkl_sparse_%c_spmd: ", c) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* C0 = A*B */                                              \
            OK (mkl_sparse_c_spmmd (op, A_mkl, B_mkl,                   \
                layout, (MKL_Complex8 *) C0->x, ldc)) ;                 \
        }
        BENCHMARK (title, tmkl, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else if (atype == GxB_FC64)
    {
        c = 'z' ;
        sprintf (title, "mkl_sparse_%c_spmd: ", c) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* C0 = A*B */                                              \
            OK (mkl_sparse_z_spmmd (op, A_mkl, B_mkl,                   \
                layout, (MKL_Complex16 *) C0->x, ldc)) ;                \
        }
        BENCHMARK (title, tmkl, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else
    {
        CHECK (false) ;
    }

    printf ("MKL time %g (%g trials),\n", tmkl.tmedian, (double) tmkl.ntrials) ;

    int nthreads ;
    OK (GxB_get (GxB_NTHREADS, &nthreads)) ;
    fprintf (stderr, "%s %2d:", what, nthreads) ;
     printf (        "%s %2d:", what, nthreads) ;

    char *rc  = C_by_col ? "|" : "_" ;
    char *AB = A_is_csr ? "_" : "|" ;
    char *tr = A_transpose ? "'" : " ";
    fprintf (stderr, "F%s=(SA%s%s)*SB%s (%2d)(%c)", rc, AB, tr, AB, (int) cncols, c) ;
     printf (        "F%s=(SA%s%s)*SB%s (%2d)(%c)", rc, AB, tr, AB, (int) cncols, c) ;

    print_results (tmkl, tgrb) ;

    //--------------------------------------------------------------------------
    // pack the results back into C0 and compare with C
    //--------------------------------------------------------------------------

    // C = C-C0
    OK (GrB_eWiseAdd (C, NULL, NULL, minus, C, C0, NULL)) ;
    // T = abs (C), typecasting if necessary
    GrB_Matrix T ;
    OK (GrB_Matrix_new (&T, GrB_FP64, cnrows, cncols)) ;
    OK (GrB_apply (T, NULL, NULL, abs, C, NULL)) ;
    // err = max (T)
    double err, norm ;
    OK (GrB_reduce (&err, NULL, GrB_MAX_MONOID_FP64, T, NULL)) ;

    // T = abs (C0), typecasting if necessary
    OK (GrB_apply (T, NULL, NULL, abs, C0, NULL)) ;
    // norm = max (T)
    OK (GrB_reduce (&norm, NULL, GrB_MAX_MONOID_FP64, C0, NULL)) ;
    if (norm == 0) norm = 1 ;
    printf ("max err: %g  rel err: %g\n", err, err / norm) ;
//  fprintf (stderr, "max err: %g  rel err: %g\n", err, err / norm) ;

    //--------------------------------------------------------------------------
    // pack the MKL sparse matrix back into the GrB_Matrix A
    //--------------------------------------------------------------------------

    // move not needed, since MKL doesn't take ownership
    OK (mkl_sparse_destroy (A_mkl)) ;
    OK (mkl_sparse_destroy (B_mkl)) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GrB_free (&T) ;
    GrB_free (&C) ;
    GrB_free (&C0) ;
}

