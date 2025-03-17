
#ifndef MKL_ILP64
#define MKL_ILP64
#endif

#include <stdio.h>
#include <stdlib.h>
#include <LAGraph.h>
#include <LAGraphX.h>
#include <mkl.h>
#include <mkl_spblas.h>

//------------------------------------------------------------------------------
// break GraphBLAS opacity
//------------------------------------------------------------------------------

// MKL keeps ownershop of matrices it creates.  Moving the contents of such a
// matrix into a GrB_Matrix would then normally require a copy.  To reduce this
// to O(1), the GrB_Matrix is declared to have shallow content, not owned by
// the GrB_Matrix.  This feature is not exposed to the normal GraphBLAS user;
// so this benchmark breaks opacity to use this feature.

// grep for "opacity" to see where this is done

// #include <GraphBLAS.h>
#include "../GraphBLAS/Source/GB.h"

//------------------------------------------------------------------------------
// helper macros
//------------------------------------------------------------------------------

// STRING(x) converts x into "x"
#define STRING(x) STR(x)
#define STR(x) #x

// return true if two strings are equal
#define MATCH(s,t) (strcmp (s,t) == 0)

// assert a condition (fail if 'ok' is false)
#define CHECK(ok)                                                       \
{                                                                       \
    if (!(ok))                                                          \
    {                                                                   \
        fprintf (stderr, "Fail: line %d %s\n", __LINE__, __FILE__) ;    \
         printf (        "Fail: line %d %s\n", __LINE__, __FILE__) ;    \
        abort ( ) ;                                                     \
    }                                                                   \
}

// call a GrB, LAGraph, or MKL method and assert that it is OK
#define OK(method)                                                      \
{                                                                       \
    int res = method ;                                                  \
    if (res != 0)                                                       \
    {                                                                   \
        fprintf (stderr, "Fail: line %d: %d %s\n", __LINE__,            \
            res, STRING (method)) ;                                     \
         printf (        "Fail: line %d: %d %s\n", __LINE__,            \
            res, STRING (method)) ;                                     \
        abort ( ) ;                                                     \
    }                                                                   \
}

//------------------------------------------------------------------------------
// support utilities
//------------------------------------------------------------------------------

extern bool just_practicing ;

void startup_LAGraph (void) ;
void finish_LAGraph (void) ;

void set_nthreads (int nthreads) ;

void load_matrix
(
    char *what,
    char *filename,
    GrB_Matrix *A_handle,
    GrB_Matrix *D_handle
) ;

void convert_matrix_format (GrB_Matrix A, GxB_Format_Value format) ;

void convert_matrix_to_type // A = (atype) D*S
(
    GrB_Matrix *A,
    GrB_Type atype,
    GrB_Matrix D,
    GrB_Matrix S
) ;

void move_GrB_Matrix_to_mkl
(
    GrB_Matrix A,               // GrB_Matrix to unpack to an MKL sparse matrix
    bool A_mkl_transpose,       // if true transpose the matrix
    sparse_matrix_t *A_mkl_handle,  // newly created MKL sparse matrix
    struct matrix_descr *descr      // pointer to MKL matrix descriptor
) ;

void move_mkl_to_GrB_Matrix
(
    GrB_Matrix A,               // GrB_Matrix to pack from an MKL sparse matrix
    bool A_mkl_transpose,       // if true transpose the matrix
    sparse_matrix_t *A_mkl_handle,  // MKL sparse matrix
    bool mkl_owns_the_matrix,   // break opacity: if true then MKL owns the
    // matrix so the contents of the GrB_Matrix A must be declared as
    // shallow.  This requires opacity to be broken.
    bool jumbled
) ;

int compar (const void *x, const void *y) ;

void get_semiring_etc
(
    GrB_Matrix A,
    GrB_Type *atype,
    GrB_UnaryOp *abs,
    GrB_BinaryOp *plus,
    GrB_BinaryOp *minus,
    GrB_BinaryOp *times,
    GrB_Monoid *plus_monoid,
    GrB_Semiring *plus_times
) ;

#define GET_MATRIX_PROPERTIES(A,format,is_csr,nrows,ncols,nvals,atype,abs,plus,minus,times,plus_monoid,plus_times) \
    /* determine if A is CSR or CSC */                          \
    GxB_Format_Value format ;                                   \
    OK (GxB_get (A, GxB_FORMAT, &format)) ;                     \
    CHECK (format == GxB_BY_ROW || format == GxB_BY_COL) ;      \
    bool is_csr = (format == GxB_BY_ROW) ;                      \
    /* get the dimension and # of entries of A */               \
    GrB_Index nrows, ncols, nvals ;                             \
    OK (GrB_Matrix_nrows (&nrows, A)) ;                         \
    OK (GrB_Matrix_ncols (&ncols, A)) ;                         \
    OK (GrB_Matrix_nvals (&nvals, A)) ;                         \
    /* get the type and operators for A */                      \
    GrB_Type atype ;                                            \
    GrB_UnaryOp abs ;                                           \
    GrB_BinaryOp plus, minus, times ;                           \
    GrB_Monoid plus_monoid ;                                    \
    GrB_Semiring plus_times ;                                   \
    get_semiring_etc (A, &atype, &abs, &plus, &minus, &times,   \
        &plus_monoid, &plus_times) ;

//------------------------------------------------------------------------------
// BENCHMARK: benchmark a body of code
//------------------------------------------------------------------------------

// The code to benchmark is #define'd by the CODE macro, which must be
// #define'd before using this macro.  The CODE is first optionally exercised
// in a warmup phase.  The timing result is returned in tres, of type
// timing_t.

// The following macros must be defined:
//      CODE            the code to benchmark
//      WARMUP_SETUP    code to run before the warmup
//      WARMUP_FINALIZE code to run after the warmup
//      TRIAL_SETUP     code to run before each trial (not included in timings)
//      TRIAL_FINALIZE  code to run after each trial (not included in timings)

typedef struct
{
    double twarmup ;    // warmup time
    double tmin ;       // min time of all trials
    double tmax ;       // max time of all trials
    double tmean ;      // mean time of all trials
    double tmedian ;    // median time of all trials
    int64_t ntrials ;   // # of trials actually performed
}
timing_t ;

// BENCHMARK takes the following parameters:
//      tres:       the timing results, of time timing_t
//      warmup:     if true, do a warmup trial, wihch is not timed
//      mintrials:  the minimum number of trials to perform
//      maxtrials:  the maximum number of trials to perform
//      maxtime:    the maximum time for all trials
//      report:     if true, report the results

#define BENCHMARK(title, tres, warmup, mintrials, maxtrials, maxtime, report) \
{                                                                           \
    if (title != NULL)                                                      \
    {                                                                       \
        printf ("Benchmark: [%s]\n", title) ;                               \
    }                                                                       \
    double timings [maxtrials], timing_total = 0 ;                          \
    tres.twarmup = 0 ;                                                      \
    if (warmup)                                                             \
    {                                                                       \
        /* run the CODE once to warm it up */                               \
        tres.twarmup = omp_get_wtime ( ) ;                                  \
        WARMUP_SETUP ;                                                      \
        { CODE ; }                                                          \
        WARMUP_FINALIZE ;                                                   \
        tres.twarmup = omp_get_wtime ( ) - tres.twarmup ;                   \
    }                                                                       \
    int64_t ntrials = 0 ;                                                   \
    while (ntrials < maxtrials &&                                           \
          (ntrials < mintrials || timing_total < maxtime))                  \
    {                                                                       \
        /* setup a trial */                                                 \
        TRIAL_SETUP ;                                                       \
        double trial_time = omp_get_wtime ( ) ;                             \
        /* run a single trial */                                            \
        { CODE ; }                                                          \
        /* time the trial and finalize it */                                \
        trial_time = omp_get_wtime ( ) - trial_time ;                       \
        timings [ntrials++] = trial_time ;                                  \
        timing_total += trial_time ;                                        \
        TRIAL_FINALIZE ;                                                    \
    }                                                                       \
    /* sort the run times and compute the timing statistics */              \
    qsort (timings, ntrials, sizeof (double), compar) ;                     \
    tres.tmin = timings [0] ;                                               \
    tres.tmax = timings [ntrials-1] ;                                       \
    tres.tmean = timing_total / ((double) ntrials) ;                        \
    tres.tmedian = timings [ntrials/2] ;                                    \
    tres.ntrials = ntrials ;                                                \
    if (report)                                                             \
    {                                                                       \
        printf ("%4d: twarmup: %g sec\n"                                    \
               "    : tmin:    %g sec\n"                                    \
               "    : tmean:   %g sec\n"                                    \
               "    : tmedian: %g sec\n"                                    \
               "    : tmax:    %g sec\n"                                    \
               "    : ntrials: %g\n",                                       \
            __LINE__, tres.twarmup, tres.tmin, tres.tmean, tres.tmedian,    \
            tres.tmax, (double) tres.ntrials) ;                             \
    }                                                                       \
}

void print_results (timing_t mkl_results, timing_t grb_results) ;
void print_speedup (char *what, double mkl_time, double grb_time) ;

//------------------------------------------------------------------------------
// benchmark methods
//------------------------------------------------------------------------------

void benchmark_mv    // y = alpha*A*x + beta*y
(
    char *what,
    GrB_Matrix A,
    double alpha_scalar,
    double beta_scalar,
    bool optimize       // if true, optimize MKL with a priori hint on
                        // # of times mv will be called.
) ;

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
) ;

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
) ;

void benchmark_add    // C = alpha*A+B, alpha*A'+B
(
    char *what,
    double alpha_scalar,
    GrB_Matrix A,
    GrB_Matrix B,
    bool A_transpose,
    bool ensure_sorted  // if true, ensure C is sorted on output, and include
                        // this in the benchmarked run time
) ;

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
) ;

void benchmark_syrk   // C = A'*A or A*A'
(
    char *what,
    GrB_Matrix A,
    bool A_transpose,   // if true, compute C=A'*A, otherwise C = A*A'
    bool ensure_sorted  // if true, ensure C is sorted on output, and include
                        // this in the benchmarked run time
) ;

void benchmark_transpose    // C = A'
(
    char *what,
    GrB_Matrix A,
    bool ensure_sorted  // if true, ensure C is sorted on output, and include
                        // this in the benchmarked run time
) ;

void benchmark_spmmd    // C = A*B or A'*B where C is dense, A and B sparse
(
    char *what,
    GrB_Matrix A,
    GrB_Matrix B,
    bool A_transpose,
    bool C_by_col           // if true, C is held by column;
                            // if false, it is held by row
) ;

void benchmark_iter
(
    char *what,
    GrB_Matrix A        // must be GrB_FP64 for now
) ;

