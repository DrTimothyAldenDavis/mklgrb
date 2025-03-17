// benchmark MKL sparse and SuiteSparse:GraphBLAS

#include "mklgrb.h"

// benchmark GrB for 3 to 100 trials, but no more than 5.0 seconds.
// Once GrB is benchmarked, MKL uses the same number of trials
#define MAXTRIALS 100
#define MINTRIALS 3
#define MAXTIME 5.0

void benchmark_mv    // y = alpha*A*x + beta*y
(
    char *what,
    GrB_Matrix A,
    double alpha_scalar,
    double beta_scalar,
    bool optimize       // if true, optimize MKL with a priori hint on
                        // # of times mv will be called.
)
{

    bool report = false ;   // if true, report the summary of the trials
    bool warmup = true ;    // if true, do a warmup first

    //--------------------------------------------------------------------------
    // get the properties of A and its operators
    //--------------------------------------------------------------------------

    GET_MATRIX_PROPERTIES (A, format, is_csr, nrows, ncols, nvals, atype,
        abs, plus, minus, times, plus_monoid, plus_times) ;

    //--------------------------------------------------------------------------
    // create the dense test vectors X, Y, and Y0
    //--------------------------------------------------------------------------

    // GxB_print (A,2) ;
    GrB_Vector X, Y0, Y ;
    OK (GrB_Vector_new (&X,  atype, ncols)) ;
    OK (GrB_Vector_new (&Y0, atype, nrows)) ;
    OK (GrB_assign (X,  NULL, NULL, 1, GrB_ALL, ncols, NULL)) ;
    OK (GrB_assign (Y0, NULL, NULL, 1, GrB_ALL, ncols, NULL)) ;
    OK (GrB_Vector_setElement (X,  0.2, 0)) ;
    OK (GrB_Vector_setElement (Y0, 0.2, 0)) ;
    OK (GrB_wait (X,  GrB_MATERIALIZE)) ;
    OK (GrB_wait (Y0, GrB_MATERIALIZE)) ;
    OK (GrB_Vector_dup (&Y, Y0)) ;
    // GxB_print (Y,2) ;
    // GxB_print (X,2) ;

    unsigned seed = 1 ;
    for (int k = 0 ; k < 10000 ; k++)
    {
        int64_t i = rand_r (&seed) ;
        i = i + RAND_MAX * rand_r (&seed) ;
        i = i % ncols ;
        double x = ((double) rand_r (&seed)) / ((double) RAND_MAX) ;
        OK (GrB_Vector_setElement (X, x, i)) ;
    }

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
        // Y = Y + A*X using GraphBLAS
        sprintf (title, "GrB_mxv: Y+=A*X where A is %s, X and Y dense "
            " vectors", is_csr ? "CSR" : "CSC") ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* Y += A*X */                                              \
            OK (GrB_mxv (Y, NULL, plus, plus_times, A, X, NULL)) ;      \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else if (alpha_scalar == 1.0 && beta_scalar == 0.0)
    {
        // Y = A*X using GraphBLAS, but ensure Y is dense to mimic MKL
        sprintf (title, "GrB_mxv: Y=A*X where A is %s, X and Y dense "
            " vectors", is_csr ? "CSR" : "CSC") ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* Y = 0 */                                                 \
            OK (GrB_assign (Y, NULL, NULL, 0, GrB_ALL, nrows, NULL)) ;  \
            /* Y += A*X */                                              \
            OK (GrB_mxv (Y, NULL, plus, plus_times, A, X, NULL)) ;      \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else if (beta_scalar == 0.0)
    {
        // Y = alpha*A*X using GraphBLAS, but ensure Y is dense to mimic MKL
        sprintf (title, "GrB_mxv: Y=(%g)*A*X where A is %s, X and Y dense "
            " vectors", alpha_scalar, is_csr ? "CSR" : "CSC") ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* Y = 0 */                                                 \
            OK (GrB_assign (Y, NULL, NULL, 0, GrB_ALL, nrows, NULL)) ;  \
            /* Y += A*X */                                              \
            OK (GrB_mxv (Y, NULL, plus, plus_times, A, X, NULL)) ;      \
            /* Y = alpha * Y */                                         \
            OK (GrB_apply (Y, NULL, NULL, times, Y, alpha_scalar, NULL)) ;  \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    }
    else if (alpha_scalar == 0.0)
    {
        // this is trivial, just y = beta*y, so it's not benchmarked
        CHECK (false) ;
    }
    else
    {
        // Y = beta*Y + alpha*A*X using GraphBLAS
        // This assumes alpha and beta are both double, and GrB typecasts them
        // internally.  If alpha is double complex, then a complex scalar can
        // be passed in instead, but this is skipped to simplify the tests.
        sprintf (title, "GrB_mxv: Y=(%g)*Y+(%g)*A*X where A is %s, X and Y"
            " dense  vectors", beta_scalar, alpha_scalar,
            is_csr ? "CSR" : "CSC") ;
        GrB_Vector Z ;
        OK (GrB_Vector_new (&Z, atype, nrows)) ;
        #undef  CODE
        #define CODE                                                    \
        {                                                               \
            /* Z = 0 */                                                 \
            OK (GrB_assign (Z, NULL, NULL, 0, GrB_ALL, nrows, NULL)) ;  \
            /* Z += A*X */                                              \
            OK (GrB_mxv (Z, NULL, plus, plus_times, A, X, NULL)) ;      \
            /* Z = alpha * Z */                                         \
            OK (GrB_apply (Z, NULL, NULL, times, Z, alpha_scalar, NULL)) ;   \
            /* Y = beta * Y */                                              \
            OK (GrB_apply (Y, NULL, NULL, times, Y, beta_scalar, NULL)) ;   \
            /* Y = Y + Z */                                             \
            OK (GrB_eWiseAdd (Y, NULL, NULL, plus, Y, Z, NULL)) ;       \
        }
        BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
        GrB_free (&Z) ;
    }

    printf ("GrB time %g (%g trials)\n", tgrb.tmedian, (double) tgrb.ntrials) ;

    //--------------------------------------------------------------------------
    // unpack A, X, and Y for MKL
    //--------------------------------------------------------------------------

    GrB_Index x_size = 0, y_size = 0, y0_size = 0 ;
    void *x = NULL, *y = NULL, *y0 = NULL ;
    OK (GxB_Vector_unpack_Full (X,  (void **) &x,  &x_size,  NULL, NULL)) ;
    OK (GxB_Vector_unpack_Full (Y0, (void **) &y0, &y0_size, NULL, NULL)) ;

    struct matrix_descr A_mkl_descr ;
    sparse_matrix_t A_mkl = NULL ;

    // move_GrB_Matrix_to_mkl creates an MKL matrix that is a shallow copy
    // of the GrB_Matrix A.  The GrB_Matrix A is not affected, and MKL does not
    // modify it.
    move_GrB_Matrix_to_mkl (A, false, &A_mkl, &A_mkl_descr) ;

    //--------------------------------------------------------------------------
    // optimize MKL, if requested
    //--------------------------------------------------------------------------

    const sparse_operation_t op = SPARSE_OPERATION_NON_TRANSPOSE ;

    double toptimize = 0 ;
    if (optimize)
    {
        toptimize = omp_get_wtime ( ) ;
        int r = (mkl_sparse_set_mv_hint (A_mkl, op, A_mkl_descr, 399)) ;
        printf ("mkl_sparse_set_mv_hint returned: %d\n", r) ;
        if (r == SPARSE_STATUS_NOT_SUPPORTED)
        {
            printf ("mkl_sparse_set_mv_hint is not supported for this case\n") ;
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

    timing_t tmkl ;
    char c ;

    // use the same # of trials as GrB, so results will be the same
    if (atype == GrB_FP32)
    {
        c = 's' ;
        float alpha = (float) alpha_scalar ;
        float beta  = (float) beta_scalar ;
        sprintf (title, "mkl_sparse_%c_mv: Y=(%g)*Y+(%g)*A*X where A is %s,"
            " X and Y dense vectors", c, beta_scalar, alpha_scalar,
            is_csr ? "CSR" : "CSC") ;
        #undef  CODE
        #define CODE                                                           \
        {                                                                      \
            /* y0 = beta*y0 + alpha*A*x */                                     \
            OK (mkl_sparse_s_mv (op, alpha, A_mkl, A_mkl_descr, x, beta, y0)) ;\
        }
        BENCHMARK (title, tmkl, warmup, tgrb.ntrials, tgrb.ntrials, 1e9, report) ;
    }
    else if (atype == GrB_FP64)
    {
        c = 'd' ;
        double alpha = (double) alpha_scalar ;
        double beta  = (double) beta_scalar ;
        sprintf (title, "mkl_sparse_%c_mv: Y=(%g)*Y+(%g)*A*X where A is %s,"
            " X and Y dense vectors", c, beta_scalar, alpha_scalar,
            is_csr ? "CSR" : "CSC") ;
        #undef  CODE
        #define CODE                                                           \
        {                                                                      \
            /* y0 = beta*y0 + alpha*A*x */                                     \
            OK (mkl_sparse_d_mv (op, alpha, A_mkl, A_mkl_descr, x, beta, y0)) ;\
        }
        BENCHMARK (title, tmkl, warmup, tgrb.ntrials, tgrb.ntrials, 1e9, report) ;
    }
    else if (atype == GxB_FC32)
    {
        c = 'c' ;
        MKL_Complex8 alpha = { (float) alpha_scalar, 0.0 } ;
        MKL_Complex8 beta  = { (float) beta_scalar , 0.0 } ;
        sprintf (title, "mkl_sparse_%c_mv: Y=(%g)*Y+(%g)*A*X where A is %s,"
            " X and Y dense vectors", c, beta_scalar, alpha_scalar,
            is_csr ? "CSR" : "CSC") ;
        #undef  CODE
        #define CODE                                                           \
        {                                                                      \
            /* y0 = beta*y0 + alpha*A*x */                                     \
            OK (mkl_sparse_c_mv (op, alpha, A_mkl, A_mkl_descr, x, beta, y0)) ;\
        }
        BENCHMARK (title, tmkl, warmup, tgrb.ntrials, tgrb.ntrials, 1e9, report) ;
    }
    else if (atype == GxB_FC64)
    {
        c = 'z' ;
        MKL_Complex16 alpha = { (double) alpha_scalar, 0.0 } ;
        MKL_Complex16 beta  = { (double) beta_scalar , 0.0 } ;
        sprintf (title, "mkl_sparse_%c_mv: Y=(%g)*Y+(%g)*A*X where A is %s,"
            " X and Y dense vectors", c, beta_scalar, alpha_scalar,
            is_csr ? "CSR" : "CSC") ;
        #undef  CODE
        #define CODE                                                           \
        {                                                                      \
            /* y0 = beta*y0 + alpha*A*x */                                     \
            OK (mkl_sparse_z_mv (op, alpha, A_mkl, A_mkl_descr, x, beta, y0)) ;\
        }
        BENCHMARK (title, tmkl, warmup, tgrb.ntrials, tgrb.ntrials, 1e9, report) ;
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
    char *A_times_x = is_csr ? "S_*x" : "S|*x" ;

    if (alpha_scalar == 1.0 && beta_scalar == 1.0)
    {
        fprintf (stderr, "y+=%s           (%c)", A_times_x, c) ;
         printf (        "y+=%s           (%c)", A_times_x, c) ;
    }
    else if (alpha_scalar == 1.0 && beta_scalar == 0.0)
    {
        fprintf (stderr, "y=%s            (%c)", A_times_x, c) ;
         printf (        "y=%s            (%c)", A_times_x, c) ;
    }
    else if (beta_scalar == 0.0)
    {
        fprintf (stderr, "y=a*%s          (%c)", A_times_x, c) ;
         printf (        "y=a*%s          (%c)", A_times_x, c) ;
    }
    else
    {
        fprintf (stderr, "y=b*y+a*%s      (%c)", A_times_x, c) ;
         printf (        "y=b*y+a*%s      (%c)", A_times_x, c) ;
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
    // pack the results back into Y0 and compare with Y
    //--------------------------------------------------------------------------

    OK (GxB_Vector_pack_Full (Y0, (void **) &y0, y0_size, false, NULL)) ;
    // Y = Y-Y0
    OK (GrB_eWiseAdd (Y, NULL, NULL, minus, Y, Y0, NULL)) ;
    // T = abs (Y), typecasting if necessary
    GrB_Vector T ;
    OK (GrB_Vector_new (&T, GrB_FP64, nrows)) ;
    OK (GrB_apply (T, NULL, NULL, abs, Y, NULL)) ;
    // err = max (T)
    double err, norm ;
    OK (GrB_reduce (&err, NULL, GrB_MAX_MONOID_FP64, T, NULL)) ;

    // T = abs (Y0), typecasting if necessary
    OK (GrB_apply (T, NULL, NULL, abs, Y0, NULL)) ;
    // norm = max (T)
    OK (GrB_reduce (&norm, NULL, GrB_MAX_MONOID_FP64, Y0, NULL)) ;
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
    GrB_free (&X) ;
    GrB_free (&Y) ;
    GrB_free (&Y0) ;
    free (x) ;
    free (y) ;
    free (y0) ;
}

