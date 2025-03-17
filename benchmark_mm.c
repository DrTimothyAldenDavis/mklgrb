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

void benchmark_mm    // C = alpha*A*B + beta*C
(
    char *what,
    GrB_Matrix A,
    double alpha_scalar,
    double beta_scalar,
    GrB_Index cncols,       // # of columns of C and B
    bool C_and_B_by_col,    // if true, C and B are held by column;
                            // if false, they are held by row
    bool optimize       // if true, optimize MKL with a priori hint on
                        // # of times mm will be called.
)
{

    bool report = false ;   // if true, report the summary of the trials
    bool warmup = true ;    // if true, do a warmup first

    //--------------------------------------------------------------------------
    // get the properties of A and its operators
    //--------------------------------------------------------------------------

    GET_MATRIX_PROPERTIES (A, format, grb_is_csr, anrows, ancols, nvals, atype,
        abs, plus, minus, times, plus_monoid, plus_times) ;

    //--------------------------------------------------------------------------
    // create the dense test matrices C, C0, and B
    //--------------------------------------------------------------------------

    GrB_Matrix B, C0, C, Z ;
    GrB_Index bnrows = ancols, bncols = cncols ;
    GrB_Index cnrows = anrows ;
    OK (GrB_Matrix_new (&B,  atype, bnrows, bncols)) ;
    OK (GrB_Matrix_new (&C0, atype, cnrows, cncols)) ;
    OK (GrB_Matrix_new (&Z,  atype, cnrows, cncols)) ;

    sparse_layout_t layout ;
    if (C_and_B_by_col)
    {
        // C and B are held by column
        layout = SPARSE_LAYOUT_COLUMN_MAJOR ;
        OK (GxB_set (B,  GxB_FORMAT, GxB_BY_COL)) ;
        OK (GxB_set (C0, GxB_FORMAT, GxB_BY_COL)) ;
        OK (GxB_set (Z,  GxB_FORMAT, GxB_BY_COL)) ;
    }
    else
    {
        // C and B are held by row
        layout = SPARSE_LAYOUT_ROW_MAJOR ;
        OK (GxB_set (B,  GxB_FORMAT, GxB_BY_ROW)) ;
        OK (GxB_set (C0, GxB_FORMAT, GxB_BY_ROW)) ;
        OK (GxB_set (Z,  GxB_FORMAT, GxB_BY_ROW)) ;
    }

    OK (GrB_assign (B,  NULL, NULL, 1, GrB_ALL, bncols, GrB_ALL, bncols, NULL));
    OK (GrB_assign (C0, NULL, NULL, 1, GrB_ALL, cncols, GrB_ALL, cncols, NULL));
    OK (GrB_Matrix_setElement (B,  0.2, 0, 0)) ;
    OK (GrB_Matrix_setElement (C0, 0.2, 0, 0)) ;

    unsigned seed = 1 ;
    for (int k = 0 ; k < 10000 ; k++)
    {
        int64_t i = rand_r (&seed) ;
        i = i + RAND_MAX * rand_r (&seed) ;
        i = i % bnrows ;
        int64_t j = rand_r (&seed) % bncols ;
        double x = ((double) rand_r (&seed)) / ((double) RAND_MAX) ;
        OK (GrB_Matrix_setElement (B, x, i, j)) ;
    }

    OK (GrB_wait (B,  GrB_MATERIALIZE)) ;
    OK (GrB_wait (C0, GrB_MATERIALIZE)) ;
    // GxB_print (B,2) ;
    // GxB_print (C0,2) ;
    OK (GrB_Matrix_dup (&C, C0)) ;

    //--------------------------------------------------------------------------
    // benchmark GraphBLAS
    //--------------------------------------------------------------------------

    #define WARMUP_SETUP    OK (GxB_set (GxB_BURBLE, true)) ;
    #define WARMUP_FINALIZE OK (GxB_set (GxB_BURBLE, false)) ;
    #define TRIAL_SETUP ;
    #define TRIAL_FINALIZE ;

    char title [1024] ;
    timing_t tgrb ;

    if (alpha_scalar == 1.0 && beta_scalar == 1.0)
    {
        // C = C + A*B using GraphBLAS
        sprintf (title, "GrB_mxm: C+=A*B where A is %s, C and B dense "
            " matrices (by %s), ncols(C) is %d", grb_is_csr ? "CSR" : "CSC",
            C_and_B_by_col ? "col" : "row", (int) cncols) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* C += A*B */                                              \
            OK (GrB_mxm (C, NULL, plus, plus_times, A, B, NULL)) ;      \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else if (alpha_scalar == 1.0 && beta_scalar == 0.0)
    {
        // C = A*B using GraphBLAS, but ensure B is dense to mimic MKL
        sprintf (title, "GrB_mxm: C=A*B where A is %s, C and B dense "
            " matrices (by %s), ncols(C) is %d", grb_is_csr ? "CSR" : "CSC",
            C_and_B_by_col ? "col" : "row", (int) cncols) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* C = 0 */                                                 \
            OK (GrB_assign (C, NULL, NULL, 0,                           \
                GrB_ALL, cnrows, GrB_ALL, cncols, NULL)) ;              \
            /* C += A*B */                                              \
            OK (GrB_mxm (C, NULL, plus, plus_times, A, B, NULL)) ;      \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else if (beta_scalar == 0.0)
    {
        // C = alpha*A*B using GraphBLAS, but ensure C is dense to mimic MKL
        sprintf (title, "GrB_mxm: C=(%g)*A*B where A is %s, C and B dense "
            " matrices (by %s), ncols(C) is %d",
            alpha_scalar, grb_is_csr ? "CSR" : "CSC",
            C_and_B_by_col ? "col" : "row", (int) cncols) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* C = 0 */                                                 \
            OK (GrB_assign (C, NULL, NULL, 0,                           \
                GrB_ALL, cnrows, GrB_ALL, cncols, NULL)) ;              \
            /* C += A*B */                                              \
            OK (GrB_mxm (C, NULL, plus, plus_times, A, B, NULL)) ;      \
            /* C = alpha * C */                                         \
            OK (GrB_apply (C, NULL, NULL, times, C, alpha_scalar, NULL)) ;  \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else if (alpha_scalar == 0.0)
    {
        // this is trivial, just C = beta*C, so it's not benchmarked
        CHECK (false) ;
    }
    else
    {
        // C = beta*C + alpha*A*B using GraphBLAS
        // This assumes alpha and beta are both double, and GrB typecasts them
        // internally.  If alpha is double complex, then a complex scalar can
        // be passed in instead, but this is skipped to simplify the tests.
        sprintf (title, "GrB_mxm: C=(%g)*C+(%g)*A*B where A is %s, C and B"
            " dense matrices (by %s), ncols(C) is %d",
            beta_scalar, alpha_scalar, grb_is_csr ? "CSR" : "CSC",
            C_and_B_by_col ? "col" : "row", (int) cncols) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* Z = 0 */                                                 \
            OK (GrB_assign (Z, NULL, NULL, 0,                           \
                GrB_ALL, cnrows, GrB_ALL, cncols, NULL)) ;              \
            /* Z += A*B */                                              \
            OK (GrB_mxm (Z, NULL, plus, plus_times, A, B, NULL)) ;      \
            /* Z = alpha * Z */                                         \
            OK (GrB_apply (Z, NULL, NULL, times, Z, alpha_scalar, NULL)) ;  \
            /* C = beta * C */                                              \
            OK (GrB_apply (C, NULL, NULL, times, C, beta_scalar, NULL)) ;   \
            /* C = C + Z */                                             \
            OK (GrB_eWiseAdd (C, NULL, NULL, plus, C, Z, NULL)) ;       \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }

    printf ("GrB time %g (%g trials)\n", tgrb.tmedian, (double) tgrb.ntrials) ;

    //--------------------------------------------------------------------------
    // unpack A, B, and C0 for MKL
    //--------------------------------------------------------------------------

    GrB_Index Bx_size = 0, Cx_size = 0, Cx0_size = 0 ;
    void *Bx = NULL, *Cx = NULL, *Cx0 = NULL ;
    MKL_INT ldb, ldc ;

    if (C_and_B_by_col)
    {
        // C and B are held by column
        ldb = bnrows ;
        ldc = cnrows ;
        OK (GxB_Matrix_unpack_FullC (B,  (void **) &Bx,  &Bx_size,
            /* not iso: */ NULL, NULL)) ;
        OK (GxB_Matrix_unpack_FullC (C0, (void **) &Cx0, &Cx0_size,
            /* not iso: */ NULL, NULL)) ;
    }
    else
    {
        // C and B are held by row
        ldb = bncols ;
        ldc = cncols ;
        OK (GxB_Matrix_unpack_FullR (B,  (void **) &Bx,  &Bx_size,
            /* not iso: */ NULL, NULL)) ;
        OK (GxB_Matrix_unpack_FullR (C0, (void **) &Cx0, &Cx0_size,
            /* not iso: */ NULL, NULL)) ;
    }

    struct matrix_descr A_mkl_descr ;
    sparse_matrix_t A_mkl = NULL ;

    // move_GrB_Matrix_to_mkl creates an MKL matrix that is a shallow copy
    // of the GrB_Matrix A.  The GrB_Matrix A is not affected, and MKL does not
    // modify it.
    bool A_mkl_transpose = (!grb_is_csr && C_and_B_by_col) ;
    move_GrB_Matrix_to_mkl (A, A_mkl_transpose, &A_mkl, &A_mkl_descr) ;

    //--------------------------------------------------------------------------
    // optimize MKL, if requested
    //--------------------------------------------------------------------------

    timing_t tmkl ;
    sparse_operation_t op ;
    char c ;
    if (A_mkl_transpose)
    {
        op = SPARSE_OPERATION_TRANSPOSE ;
    }
    else
    {
        op = SPARSE_OPERATION_NON_TRANSPOSE ;
    }

    double toptimize = 0 ;
    if (optimize)
    {
        toptimize = omp_get_wtime ( ) ;
        int r = (mkl_sparse_set_mm_hint (A_mkl, op, A_mkl_descr, 
            layout, cncols, 399)) ;
        printf ("mkl_sparse_set_mm_hint returned: %d\n", r) ;
        if (r == SPARSE_STATUS_NOT_SUPPORTED)
        {
            printf ("mkl_sparse_set_mm_hint is not supported for this case\n") ;
        }
        else
        {
            OK (r) ;
        }
        if (r == SPARSE_STATUS_SUCCESS)
        {
            OK (mkl_sparse_optimize (A_mkl)) ;
            toptimize = omp_get_wtime ( ) - toptimize ;
             printf (        "MKL optimize time: %g sec\n", toptimize) ;
        }
        else
        {
            toptimize = 0 ;
        }
    }

    //--------------------------------------------------------------------------
    // benchmark MKL
    //--------------------------------------------------------------------------

    // use the same # of trials as GrB, so results will be the same
    if (atype == GrB_FP32)
    {
        c = 's' ;
        float alpha = (float) alpha_scalar ;
        float beta  = (float) beta_scalar ;
        sprintf (title, "mkl_sparse_%c_mm: C=(%g)*C+(%g)*A*B where A is %s,"
            " C and B dense matrices (cncols %d, format by %s",
            c, beta_scalar, alpha_scalar, grb_is_csr ? "CSR" : "CSC",
            (int) cncols, C_and_B_by_col ? "col" : "row") ;
        #undef  CODE
        #define CODE                                                           \
        {                                                                      \
            /* C0 = beta*C0 + alpha*A*B */                                     \
            OK (mkl_sparse_s_mm (op, alpha, A_mkl, A_mkl_descr,                \
                layout, Bx, cncols, ldb, beta, Cx0, ldc)) ;                    \
        }
        BENCHMARK (title, tmkl, warmup, tgrb.ntrials, tgrb.ntrials, 1e9,
            report) ;
    }
    else if (atype == GrB_FP64)
    {
        c = 'd' ;
        double alpha = (double) alpha_scalar ;
        double beta  = (double) beta_scalar ;
        sprintf (title, "mkl_sparse_%c_mm: C=(%g)*C+(%g)*A*B where A is %s,"
            " C and B dense matrices (cncols %d, format by %s",
            c, beta_scalar, alpha_scalar, grb_is_csr ? "CSR" : "CSC",
            (int) cncols, C_and_B_by_col ? "col" : "row") ;
        #undef  CODE
        #define CODE                                                           \
        {                                                                      \
            /* C0 = beta*C0 + alpha*A*B */                                     \
            OK (mkl_sparse_d_mm (op, alpha, A_mkl, A_mkl_descr,                \
                layout, Bx, cncols, ldb, beta, Cx0, ldc)) ;                    \
        }
        BENCHMARK (title, tmkl, warmup, tgrb.ntrials, tgrb.ntrials, 1e9,
            report) ;
    }
    else if (atype == GxB_FC32)
    {
        c = 'c' ;
        MKL_Complex8 alpha = { (float) alpha_scalar, 0.0 } ;
        MKL_Complex8 beta  = { (float) beta_scalar , 0.0 } ;
        sprintf (title, "mkl_sparse_%c_mm: C=(%g)*C+(%g)*A*B where A is %s,"
            " C and B dense matrices (cncols %d, format by %s",
            c, beta_scalar, alpha_scalar, grb_is_csr ? "CSR" : "CSC",
            (int) cncols, C_and_B_by_col ? "col" : "row") ;
        #undef  CODE
        #define CODE                                                           \
        {                                                                      \
            /* C0 = beta*C0 + alpha*A*B */                                     \
            OK (mkl_sparse_c_mm (op, alpha, A_mkl, A_mkl_descr,                \
                layout, Bx, cncols, ldb, beta, Cx0, ldc)) ;                    \
        }
        BENCHMARK (title, tmkl, warmup, tgrb.ntrials, tgrb.ntrials, 1e9,
            report) ;
    }
    else if (atype == GxB_FC64)
    {
        c = 'z' ;
        MKL_Complex16 alpha = { (double) alpha_scalar, 0.0 } ;
        MKL_Complex16 beta  = { (double) beta_scalar , 0.0 } ;
        sprintf (title, "mkl_sparse_%c_mm: C=(%g)*C+(%g)*A*B where A is %s,"
            " C and B dense matrices (cncols %d, format by %s",
            c, beta_scalar, alpha_scalar, grb_is_csr ? "CSR" : "CSC",
            (int) cncols, C_and_B_by_col ? "col" : "row") ;
        #undef  CODE
        #define CODE                                                           \
        {                                                                      \
            /* C0 = beta*C0 + alpha*A*B */                                     \
            OK (mkl_sparse_z_mm (op, alpha, A_mkl, A_mkl_descr,                \
                layout, Bx, cncols, ldb, beta, Cx0, ldc)) ;                    \
        }
        BENCHMARK (title, tmkl, warmup, tgrb.ntrials, tgrb.ntrials, 1e9,
            report) ;
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

    char *rc  = C_and_B_by_col ? "|" : "_" ;
    char *AB = grb_is_csr ? "S_*F" : "S|*F" ;
    if (alpha_scalar == 1.0 && beta_scalar == 1.0)
    {
        fprintf (stderr, "C+=%s      (%s%2d)(%c)", AB, rc, (int) cncols, c) ;
         printf (        "C+=%s      (%s%2d)(%c)", AB, rc, (int) cncols, c) ;
    }
    else if (alpha_scalar == 1.0 && beta_scalar == 0.0)
    {
        fprintf (stderr, "C=%s       (%s%2d)(%c)", AB, rc, (int) cncols, c) ;
         printf (        "C=%s       (%s%2d)(%c)", AB, rc, (int) cncols, c) ;
    }
    else if (beta_scalar == 0.0)
    {
        fprintf (stderr, "C=a*%s     (%s%2d)(%c)", AB, rc, (int) cncols, c) ;
         printf (        "C=a*%s     (%s%2d)(%c)", AB, rc, (int) cncols, c) ;
    }
    else
    {
        fprintf (stderr, "C=b*C+a*%s (%s%2d)(%c)", AB, rc, (int) cncols, c) ;
         printf (        "C=b*C+a*%s (%s%2d)(%c)", AB, rc, (int) cncols, c) ;
    }

    if (toptimize > 0)
    {
        fprintf (stderr, "+") ;
    }
    else
    {
        fprintf (stderr, " ") ;
    }

    // add the optimize time to the 1st run
    tmkl.twarmup += toptimize ;

    print_results (tmkl, tgrb) ;

    //--------------------------------------------------------------------------
    // pack the results back into C0 and compare with C
    //--------------------------------------------------------------------------

    if (C_and_B_by_col)
    {
        OK (GxB_Matrix_pack_FullC (C0, (void **) &Cx0, Cx0_size, false, NULL)) ;
    }
    else
    {
        OK (GxB_Matrix_pack_FullR (C0, (void **) &Cx0, Cx0_size, false, NULL)) ;
    }

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

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GrB_free (&T) ;
    GrB_free (&B) ;
    GrB_free (&C) ;
    GrB_free (&C0) ;
    GrB_free (&Z) ;
    free (Bx) ;
    free (Cx) ;
    free (Cx0) ;
}

