#include "mklgrb.h"

/*

#define MAXTRIALS 1
#define MINTRIALS 1
#define MAXTIME 5.0
*/

#define MAXTRIALS 100
#define MINTRIALS 3
#define MAXTIME 5.0

//------------------------------------------------------------------------------
// matIter_mv
//------------------------------------------------------------------------------

GrB_Info matIter_mv        // y = A*x using the matrix iterator
(
    GrB_Vector y,
    GrB_Matrix A,
    GrB_Vector x
) ;

GrB_Info matIter_mv        // y = A*x using the matrix iterator
(
    GrB_Vector y,
    GrB_Matrix A,
    GrB_Vector x
)
{

    // unpack X and Y
    size_t X_size, Y_size ;
    double *X, *Y ;
    GrB_Index nrows, ncols ;
    OK (GrB_Vector_size (&nrows, y)) ;
    OK (GrB_Vector_size (&ncols, x)) ;
    OK (GxB_Vector_unpack_Full (x, (void **) &X, &X_size, NULL, NULL)) ;
    OK (GxB_Vector_unpack_Full (y, (void **) &Y, &Y_size, NULL, NULL)) ;

    for (int64_t i = 0 ; i < nrows ; i++)
    {
        Y [i] = 0 ;
    }

    // create an iterator
    GxB_Iterator iterator ;
    GxB_Iterator_new (&iterator) ;
    // attach it to the the matrix A, known to be type GrB_FP64
    GrB_Info info = GxB_Matrix_Iterator_attach (iterator, A, NULL) ;
    CHECK (info >= GrB_SUCCESS) ;
    // seek to the first entry
    info = GxB_Matrix_Iterator_seek (iterator, 0) ;
    while (info != GxB_EXHAUSTED)
    {
        // iterate over entries in all of A
        // get the entry A(i,j)
        GrB_Index i, j ;
        GxB_Matrix_Iterator_getIndex (iterator, &i, &j) ;
        double aij = GxB_Iterator_get_FP64 (iterator) ;
        // printf ("entry (%lu, %lu) = %g\n", i, j, aij) ;
        Y [i] += aij * X [j] ;
        // move to the next entry in A
        info = GxB_Matrix_Iterator_next (iterator) ;
    }
    GrB_free (&iterator) ;

    // pack X and Y
    OK (GxB_Vector_pack_Full (x, (void **) &X, X_size, false, NULL)) ;
    OK (GxB_Vector_pack_Full (y, (void **) &Y, Y_size, false, NULL)) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// rowIter_mv
//------------------------------------------------------------------------------

GrB_Info rowIter_mv        // y = A*x using the iterator
(
    GrB_Vector y,
    GrB_Matrix A,
    GrB_Vector x
) ;

GrB_Info rowIter_mv        // y = A*x using the iterator
(
    GrB_Vector y,
    GrB_Matrix A,
    GrB_Vector x
)
{

    // unpack X and Y
    size_t X_size, Y_size ;
    double *X, *Y ;
    GrB_Index nrows, ncols, nvals ;
    OK (GrB_Vector_size (&nrows, y)) ;
    OK (GrB_Vector_size (&ncols, x)) ;
    OK (GxB_Vector_unpack_Full (x, (void **) &X, &X_size, NULL, NULL)) ;
    OK (GxB_Vector_unpack_Full (y, (void **) &Y, &Y_size, NULL, NULL)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;

double t = omp_get_wtime ( ) ;
    for (int64_t i = 0 ; i < nrows ; i++)
    {
        Y [i] = 0 ;
    }
t = omp_get_wtime ( ) - t ; printf ("rowIter, clear %g\n", t) ;
t = omp_get_wtime ( ) ;

    // create an iterator
    GxB_Iterator iterator ;
    GxB_Iterator_new (&iterator) ;
    // attach it to the first row of the matrix A, known to be type GrB_FP64
    GrB_Info info = GxB_rowIterator_attach (iterator, A, NULL) ;
    CHECK (info >= GrB_SUCCESS) ;
    // seek to A(0,:)
    info = GxB_rowIterator_seekRow (iterator, 0) ;
//  if (nvals < 2000) printf ("rowIter seek 0: %d\n", info) ;
    while (info != GxB_EXHAUSTED)
    {
        // iterate over entries in A(i,:)
        GrB_Index i = GxB_rowIterator_getRowIndex (iterator) ;
//      if (nvals < 2000) printf ("rowIter at row: %d\n", i) ;
        double yi = 0 ;
        while (info == GrB_SUCCESS)
        {
            // get the entry A(i,j)
            GrB_Index j = GxB_rowIterator_getColIndex (iterator) ;
            double  aij = GxB_Iterator_get_FP64 (iterator) ;
//          if (nvals < 2000) printf ("   A(%ld,%ld) =  %g\n", i, j, aij) ;
            yi += aij * X [j] ;
            // move to the next entry in A(i,:)
            info = GxB_rowIterator_nextCol (iterator) ;
        }
        Y [i] = yi ;
        // move to the next row, A(i+1,:)
        info = GxB_rowIterator_nextRow (iterator) ;
    }
    GrB_free (&iterator) ;

t = omp_get_wtime ( ) - t ; printf ("rowIter, A*x %g\n", t) ;

    // pack X and Y
    OK (GxB_Vector_pack_Full (x, (void **) &X, X_size, false, NULL)) ;
    OK (GxB_Vector_pack_Full (y, (void **) &Y, Y_size, false, NULL)) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// unpack_mv
//------------------------------------------------------------------------------

GrB_Info unpack_mv        // y = A*x using unpack 
(
    GrB_Vector y,
    GrB_Matrix A,
    GrB_Vector x
) ;

GrB_Info unpack_mv        // y = A*x using unpack 
(
    GrB_Vector y,
    GrB_Matrix A,
    GrB_Vector x
)
{

    // unpack X and Y
    size_t X_size, Y_size ;
    double *X, *Y ;
    GrB_Index nrows, ncols ;
    OK (GrB_Vector_size (&nrows, y)) ;
    OK (GrB_Vector_size (&ncols, x)) ;
    OK (GxB_Vector_unpack_Full (x, (void **) &X, &X_size, NULL, NULL)) ;
    OK (GxB_Vector_unpack_Full (y, (void **) &Y, &Y_size, NULL, NULL)) ;

    //--------------------------------------------------------------------------
    // unpack the matrix
    //--------------------------------------------------------------------------

    size_t Ap_size, Aj_size, Ax_size ;
    GrB_Index *Ap, *Aj ;
    double *Ax ;
    bool iso ;
    OK (GxB_Matrix_unpack_CSR (A, &Ap, &Aj, (void **) &Ax, &Ap_size, &Aj_size,
        &Ax_size, &iso, NULL, NULL)) ;

    //--------------------------------------------------------------------------
    // iterate over the CSR format
    //--------------------------------------------------------------------------

    for (int64_t i = 0 ; i < nrows ; i++)
    {
        double yi = 0 ;
        for (int64_t p = Ap [i] ; p < Ap [i+1] ; p++)
        {
            GrB_Index j = Aj [p] ;
            double aij = Ax [iso ? 0: p] ;
            yi += aij * X [j] ;
        }
        Y [i] = yi ;
    }

    //--------------------------------------------------------------------------
    // pack the matrix
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_pack_CSR (A, &Ap, &Aj, (void **) &Ax, Ap_size, Aj_size,
        Ax_size, iso, false, NULL)) ;

    // pack X and Y
    OK (GxB_Vector_pack_Full (x, (void **) &X, X_size, false, NULL)) ;
    OK (GxB_Vector_pack_Full (y, (void **) &Y, Y_size, false, NULL)) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// benchmark_iter
//------------------------------------------------------------------------------

void benchmark_iter
(
    char *what,
    GrB_Matrix A        // must be GrB_FP64 for now
)
{

    bool report = true ;    // if true, report the summary of the trials
    bool warmup = true ;    // if true, do a warmup first

    GrB_BinaryOp plus = GrB_PLUS_FP64 ; 
    GrB_Semiring plus_times = GrB_PLUS_TIMES_SEMIRING_FP64 ; 

    //--------------------------------------------------------------------------
    // iterate over the matrix A
    //--------------------------------------------------------------------------

#if 0
    // create an iterator
    GxB_Iterator iterator ;
    GxB_Iterator_new (&iterator) ;
    // attach it to the first row of the matrix A, known to be type GrB_FP64
    GrB_Info info = GxB_rowIterator_attach (iterator, A, NULL) ;
    CHECK (info >= GrB_SUCCESS) ;
    // seek to A(0,:)
    info = GxB_rowIterator_seekRow (iterator, 0) ;
    while (info != GxB_EXHAUSTED)
    {
        // iterate over entries in A(i,:)
        GrB_Index i = GxB_rowIterator_getRowIndex (iterator) ;
        if (i < 40) printf ("\nRow %lu\n", i) ;
        while (info == GrB_SUCCESS)
        {
            // get the entry A(i,j)
            GrB_Index j = GxB_rowIterator_getColIndex (iterator) ;
            double  aij = GxB_Iterator_get_FP64 (iterator) ;
            if (i < 40) printf ("   (%lu,%lu) = %g\n", i, j, aij) ;
            // move to the next entry in A(i,:)
            info = GxB_rowIterator_nextCol (iterator) ;
        }
        // move to the next row, A(i+1,:)
        info = GxB_rowIterator_nextRow (iterator) ;
    }
    GrB_free (&iterator) ;
#endif

    //--------------------------------------------------------------------------
    // benchmark the iterator to compute y = A*x
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols, nvals ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;

    GrB_Vector x, y, y2, y3, y4 ;
    OK (GrB_Vector_new (&x, GrB_FP64, ncols)) ;
    OK (GrB_Vector_new (&y, GrB_FP64, nrows)) ;
    OK (GrB_assign (x, NULL, NULL, 1, GrB_ALL, ncols, NULL)) ;
    OK (GrB_apply (x, NULL, GrB_PLUS_FP64, GrB_ROWINDEX_INT64, x, 0, NULL)) ;
    OK (GrB_assign (y, NULL, NULL, 0, GrB_ALL, ncols, NULL)) ;
    OK (GrB_Vector_setElement_FP64 (y, 1, 1)) ;

    OK (GrB_wait (x, GrB_MATERIALIZE)) ;
    OK (GrB_wait (y, GrB_MATERIALIZE)) ;
    // OK (GxB_print (x, 2)) ;
    // OK (GxB_print (y, 2)) ;
    OK (GrB_Vector_dup (&y2, y)) ;
    OK (GrB_Vector_dup (&y3, y)) ;
    OK (GrB_Vector_dup (&y4, y)) ;

//  if (nvals < 2000) OK (GxB_print (A, 3)) ;
    OK (GxB_print (A, 2)) ;

    #define WARMUP_SETUP    OK (GxB_set (GxB_BURBLE, true)) ;
    #define WARMUP_FINALIZE OK (GxB_set (GxB_BURBLE, false)) ;
    #define TRIAL_SETUP ;
    #define TRIAL_FINALIZE ;

    char title [1024] ;
    timing_t titer, tgrb, tunpack, tmat ;

    // Y = A*X using GraphBLAS row iterator
    sprintf (title, "Y=A*X, A is CSR, X and Y dense vectors, row iter\n");
    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        OK (rowIter_mv (y, A, x)) ;                                 \
    }
    BENCHMARK (title, titer, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    // OK (GxB_print (y, 2)) ;
    printf ("row iter time: %g\n", titer.tmedian) ; fflush (stdout) ;
//  if (nvals < 2000) { printf ("for row iter: \n") ; OK (GxB_print (y, 3)) ; }

    // Y = A*X using GraphBLAS matrix iterator
    sprintf (title, "Y=A*X, A is CSR, X and Y dense vectors, matrix iter\n");
    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        OK (matIter_mv (y4, A, x)) ;                                \
    }
    BENCHMARK (title, tmat, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    // OK (GxB_print (y, 2)) ;
    printf ("mat iter time: %g\n", tmat.tmedian) ; fflush (stdout) ;
    // printf ("For mat iter:\n") ; OK (GxB_print (y4,2)) ;

    // Y = A*X using GraphBLAS GrB_mxv and a single thread
    int nthreads ;
    OK (GxB_get (GxB_NTHREADS, &nthreads)) ;
    OK (GxB_set (GxB_NTHREADS, 1)) ;
    sprintf (title, "Y=A*X, A is CSR, X and Y dense vectors, with GrB_mxv\n");
    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        /* Y = 0 */                                                 \
        OK (GrB_assign (y2, NULL, NULL, 0, GrB_ALL, nrows, NULL)) ; \
        /* Y += A*X */                                              \
        OK (GrB_mxv (y2, NULL, plus, plus_times, A, x, NULL)) ;     \
    }
    BENCHMARK (title, tgrb, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    printf ("mxv time (1 thread): %g\n", tgrb.tmedian) ; fflush (stdout) ;
    OK (GxB_set (GxB_NTHREADS, nthreads)) ;
//  if (nvals < 2000) { printf ("For mxv:\n") ; OK (GxB_print (y2,3)) ; }

    // Y = A*X using GraphBLAS unpack
    sprintf (title, "Y=A*X, A is CSR, X and Y dense vectors, with unpack\n");
    #undef  CODE
    #define CODE                                                    \
    {                                                               \
        OK (unpack_mv (y3, A, x)) ;                                 \
    }
    BENCHMARK (title, tunpack, warmup, MINTRIALS, MAXTRIALS, MAXTIME, report) ;
    // OK (GxB_print (y3, 2)) ;
    printf ("unpack time: %g\n", tunpack.tmedian) ; fflush (stdout) ;

    // check the result
    double err, nrm, err3, err4 ;
    // y = y - y2
    OK (GrB_eWiseAdd (y, NULL, NULL, GrB_MINUS_FP64, y, y2, NULL)) ;
    OK (GrB_select (y, NULL, NULL, GrB_VALUENE_FP64, y, 0, NULL)) ;
//  if (nvals < 2000) { printf ("rowIter - mxv:\n") ; OK (GxB_print (y,3)) ;}
    // y = abs (y)
    OK (GrB_apply (y, NULL, NULL, GrB_ABS_FP64, y, NULL)) ;
    // err = max (y)
    err = 0 ;
    OK (GrB_reduce (&err, GrB_MAX_FP64, GrB_MAX_MONOID_FP64, y, NULL)) ;

    // y3 = y3 - y2
    OK (GrB_eWiseAdd (y3, NULL, NULL, GrB_MINUS_FP64, y3, y2, NULL)) ;
    // y3 = abs (y3)
    OK (GrB_apply (y3, NULL, NULL, GrB_ABS_FP64, y3, NULL)) ;
    // err3 = max (y3)
    OK (GrB_reduce (&err3, NULL, GrB_MAX_MONOID_FP64, y3, NULL)) ;

    // y4 = y4 - y2
    OK (GrB_eWiseAdd (y4, NULL, NULL, GrB_MINUS_FP64, y4, y2, NULL)) ;
    // y4 = abs (y4)
    OK (GrB_apply (y4, NULL, NULL, GrB_ABS_FP64, y4, NULL)) ;
    // err4 = max (y4)
    OK (GrB_reduce (&err4, NULL, GrB_MAX_MONOID_FP64, y4, NULL)) ;

    // y2 = abs (y2)
    OK (GrB_apply (y2, NULL, NULL, GrB_ABS_FP64, y2, NULL)) ;
    // nrm = max (y2)
    OK (GrB_reduce (&nrm, NULL, GrB_MAX_MONOID_FP64, y2, NULL)) ;

    if (nrm == 0) nrm = 1 ;
    err = err / nrm ;
    err3 = err3 / nrm ;
    err4 = err4 / nrm ;
    fprintf (stderr,
    "%s row iter: %10.4f (%8.2f) "
    "mat iter: %10.4f (%8.2f) "
    "unpack: %10.4f (%8.2f) "
    "mxv %10.4f err: %g %g %g\n",
        what,
        titer.tmedian, titer.tmedian / tgrb.tmedian,
        tmat.tmedian, tmat.tmedian / tgrb.tmedian,
        tunpack.tmedian, tunpack.tmedian / tgrb.tmedian,
        tgrb.tmedian, err, err3, err4) ;

     printf (
    "%s row iter: %10.4f (%8.2f) "
    "mat iter: %10.4f (%8.2f) "
    "unpack: %10.4f (%8.2f) "
    "mxv %10.4f err: %g %g %g\n",
        what,
        titer.tmedian, titer.tmedian / tgrb.tmedian,
        tmat.tmedian, tmat.tmedian / tgrb.tmedian,
        tunpack.tmedian, tunpack.tmedian / tgrb.tmedian,
        tgrb.tmedian, err, err3, err4) ;

    //--------------------------------------------------------------------------
    // unpack the matrix to print it
    //--------------------------------------------------------------------------

#if 0
    if (nrows < 40 && ncols < 40)
    {

        size_t Ap_size, Aj_size, Ax_size ;
        GrB_Index *Ap, *Aj ;
        double *Ax ;
        bool iso ;
        OK (GxB_Matrix_unpack_CSR (A, &Ap, &Aj, (void **) &Ax, &Ap_size, &Aj_size,
            &Ax_size, &iso, NULL, NULL)) ;

        //----------------------------------------------------------------------
        // iterate over the CSR format
        //----------------------------------------------------------------------

        printf ("\nNow as unpacked CSR:\n") ;
        for (int64_t i = 0 ; i < nrows ; i++)
        {
            printf ("\nCSR Row %lu\n", i) ;
            for (int64_t p = Ap [i] ; p < Ap [i+1] ; p++)
            {
                GrB_Index j = Aj [p] ;
                double aij = Ax [iso ? 0: p] ;
                printf ("   (%lu,%lu) = %g\n", i, j, aij) ;
            }
        }

        //----------------------------------------------------------------------
        // pack the matrix
        //----------------------------------------------------------------------

        OK (GxB_Matrix_pack_CSR (A, &Ap, &Aj, (void **) &Ax, Ap_size, Aj_size,
            Ax_size, iso, false, NULL)) ;
    }
#endif

    GrB_free (&x) ;
    GrB_free (&y) ;
    GrB_free (&y2) ;
    GrB_free (&y3) ;
    GrB_free (&y4) ;
}
