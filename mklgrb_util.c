// benchmark MKL sparse and SuiteSparse:GraphBLAS

#ifndef MKL_ILP64
#define MKL_ILP64
#endif
#include "mklgrb.h"

bool just_practicing = false ;

//------------------------------------------------------------------------------
// startup: start LAGraph and check versions
//------------------------------------------------------------------------------

void startup_LAGraph (void)
{
    // Normally, an application such as this one could be compiled against,
    // say, SuiteSparse:GraphBLAS v6.0.1 and later linked with, say, v6.0.2.
    // However, to ensure accurate benchmarking, this benchmark must be
    // compiled with the exact same GraphBLAS.h #include file version that
    // matches the compiled runtime *.so SuiteSparse:GraphBLAS library.

    char msg [LAGRAPH_MSG_LEN] ;
    OK (LAGraph_Init (msg)) ;
    size_t none [64] ;
    memset (none, 0, 64 * sizeof (size_t)) ;
    GxB_set (GxB_MEMORY_POOL, none) ;
    OK (LAGraph_Random_Init (msg)) ;
    char *library, *date ;
    int ver [3], v ;
    OK (GxB_get (GxB_LIBRARY_NAME, &library)) ;
    OK (GxB_get (GxB_LIBRARY_VERSION, ver)) ;
    OK (GxB_get (GxB_LIBRARY_DATE, &date)) ;
    printf ("\n========================================"
            "========================================\n") ;
    printf ("GrB include: %s v%d.%d.%d (%s)\n", GxB_IMPLEMENTATION_NAME,
        GxB_IMPLEMENTATION_MAJOR, GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB, GxB_IMPLEMENTATION_DATE) ;
    printf ("GrB runtime: %s v%d.%d.%d (%s)\n", library,
        ver [0], ver [1], ver [2], date) ;
    int v_include = GxB_VERSION (GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR, GxB_IMPLEMENTATION_SUB) ;
    int v_runtime = GxB_VERSION (ver [0], ver [1], ver [2]) ;

    char *compiler ;
    int cver [3] ;
    OK (GxB_get (GxB_COMPILER_NAME, &compiler)) ;
    OK (GxB_get (GxB_COMPILER_VERSION, cver)) ;
    printf ("GraphBLAS compiled with %s (v%d.%d.%d)\n", compiler,
        cver [0], cver [1], cver [2]) ;

    if (!MATCH ("SuiteSparse:GraphBLAS", library) ||
        !MATCH ("SuiteSparse:GraphBLAS", GxB_IMPLEMENTATION_NAME) ||
        (v_include != v_runtime) || v_runtime < GxB_VERSION (6,1,3))
    {
        fprintf (stderr, "SuiteSparse:GraphBLAS v6.1.3 or later required\n") ;
        abort ( ) ;
    }

    if (!MATCH (date, GxB_IMPLEMENTATION_DATE))
    {
        fprintf (stderr, "SuiteSparse:GraphBLAS: dates do not match\n") ;
        abort ( ) ;
    }

    printf ("MKL version: %d.%d.%d, sizeof(MKL_INT): %d\n",
        __INTEL_MKL__, __INTEL_MKL_MINOR__,
        __INTEL_MKL_UPDATE__, (int) sizeof (MKL_INT)) ;

    printf ("Libraries:\n") ;
    fflush (stdout) ;
    fflush (stderr) ;
#if (defined(__apple__) || defined(__APPLE__) || defined(__MACH__))
    system ("otool -L mklgrb") ;
#else
    system ("ldd mklgrb") ;
#endif
    printf ("\n") ;

}

void finish_LAGraph (void)
{
    char msg [LAGRAPH_MSG_LEN] ;
    OK (LAGraph_Finalize (msg)) ;
    OK (LAGraph_Random_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// set_nthreads: set # of threads for MKL and GrB
//------------------------------------------------------------------------------

void set_nthreads (int nthreads)
{
    omp_set_num_threads (nthreads) ;        // any OpenMP #pragma
    GxB_set (GxB_NTHREADS, nthreads) ;      // # threads for GraphBLAS
    mkl_set_num_threads (nthreads) ;        // # threads for MKL
}

//------------------------------------------------------------------------------
// load a GrB_Matrix from a *.lagraph file, or stdin and convert to iso-1
//------------------------------------------------------------------------------

void load_matrix
(
    char *what,
    char *filename,
    GrB_Matrix *A_handle,
    GrB_Matrix *D_handle
)
{
    char msg [LAGRAPH_MSG_LEN] ;
    GrB_Matrix *Set = NULL, A = NULL, D = NULL ;
    GrB_Index nmatrices = 0 ;
    char *collection ;
    double tload = omp_get_wtime ( ) ;
    OK (LAGraph_SLoadSet (filename, &Set, &nmatrices, &collection, msg)) ;
    tload = omp_get_wtime ( ) - tload ;
    printf  (        "\nBenchmarking: %s, Matrix: %s load time: %g sec\n", what, collection, tload) ;
    fprintf (stderr, "\nBenchmarking: %s, Matrix: %s load time: %g sec\n", what, collection, tload) ;
    double tprep = omp_get_wtime ( ) ;

    // get the type and dimensions
    char type_name [GxB_MAX_NAME_LEN] ;
    OK (GxB_Matrix_type_name (type_name, Set [0])) ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, Set [0])) ;
    OK (GrB_Matrix_ncols (&ncols, Set [0])) ;

    // set all the values to 1
    OK (GrB_assign (Set [0], Set [0], NULL, 1, GrB_ALL, nrows, GrB_ALL, ncols,
        GrB_DESC_S)) ;
    // GxB_print (Set [0], 2) ;

    // ensure the matrix is GrB_FP64
    if (MATCH (type_name, "double"))
    {
        A = Set [0] ;
        Set [0] = NULL ;
    }
    else
    {
        // A = (double) Set [0]

        OK (GrB_Matrix_new (&A, GrB_FP64, nrows, ncols)) ;
        OK (GrB_assign (A, NULL, NULL, Set [0], GrB_ALL, nrows, GrB_ALL, ncols,
            NULL)) ;
    }

    LAGraph_SFreeSet (&Set, nmatrices) ;
    LAGraph_Free ((void **) &collection) ;

    // ensure A is a sparse CSR matrix with no pending work
    OK (GxB_set (A, GxB_FORMAT, GxB_BY_ROW)) ;
    OK (GxB_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GrB_wait (A, GrB_MATERIALIZE)) ;

    // compute the row degree
    GrB_Vector degree, one ;
    OK (GrB_Vector_new (&degree, GrB_FP64, nrows)) ;
    OK (GrB_Vector_new (&one, GrB_FP64, nrows)) ;
    OK (GrB_assign (one, NULL, NULL, 1, GrB_ALL, nrows, NULL)) ;
    OK (GrB_mxv (degree, NULL, NULL, GxB_PLUS_PAIR_FP64, A, one, NULL)) ;
    GrB_Index nonempty ;
    OK (GrB_Vector_nvals (&nonempty, degree)) ;
    // degree = max (one, degree), so degree is a full vector
    // (it would be sparse if A has any empty rows)
    OK (GrB_eWiseAdd (degree, NULL, NULL, GrB_MAX_FP64, one, degree, NULL)) ;
    // degree = 1./degree
    OK (GrB_eWiseAdd (degree, NULL, NULL, GrB_DIV_FP64, one, degree, NULL)) ;
    // D = diag (degree)
    OK (GrB_Matrix_new (&D, GrB_FP64, nrows, nrows)) ;
    OK (GrB_Matrix_diag (D, degree, 0)) ;
    GrB_free (&one) ;
    GrB_free (&degree) ;

    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    tprep = omp_get_wtime ( ) - tprep ;
    printf ("prep: %g sec\n", tprep) ;

    fprintf (stderr, "Matrix: n %8.3fM nn %8.3fM nvals %8.3fM, "
        "load+prep: %9.3f sec\n",
        (double) nrows / ((double) 1e6),
        (double) nonempty / ((double) 1e6),
        (double) nvals / ((double) 1e6), tload + tprep) ;
     printf (        "Matrix: n %8.3fM nn %8.3fM nvals %8.3fM, "
        "load+prep: %9.3f sec\n",
        (double) nrows / ((double) 1e6),
        (double) nonempty / ((double) 1e6),
        (double) nvals / ((double) 1e6), tload + tprep) ;

    // return the matrix A and its diagonal 1/(rowdegree) matrix D
    (*A_handle) = A ;
    (*D_handle) = D ;
}

//------------------------------------------------------------------------------
// convert a GrB_Matrix to CSC or CSC
//------------------------------------------------------------------------------

void convert_matrix_format (GrB_Matrix A, GxB_Format_Value format)
{
    double t = omp_get_wtime ( ) ;
    OK (GxB_set (A, GxB_FORMAT, format)) ;
    t = omp_get_wtime ( ) - t ;
    fprintf (stderr, "\nchange matrix to %s: %g sec\n", (format == GxB_BY_ROW) ? "CSR" : "CSC", t) ;
    printf (         "\nchange matrix to %s: %g sec\n", (format == GxB_BY_ROW) ? "CSR" : "CSC", t) ;
}

//------------------------------------------------------------------------------
// compar: compare to double values
//------------------------------------------------------------------------------

int compar (const void *x, const void *y)
{
    double a = (*((const double *) x)) ;
    double b = (*((const double *) y)) ;
    return ((a == b) ? 0 : ((a < b) ? -1 : 1)) ;
}

//------------------------------------------------------------------------------
// get_semiring_etc
//------------------------------------------------------------------------------

// Given a matrix A of type GrB_FP32, GrB_FP64, GxB_FC32, GxB_FC64, return
// the type and the corresponding operators to work on the matrix.

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
)
{
    OK (GxB_Matrix_type (atype, A)) ;
    if (*atype == GrB_FP32)
    {
        *abs = GrB_ABS_FP32 ;
        *plus = GrB_PLUS_FP32 ;
        *minus = GrB_MINUS_FP32 ;
        *times = GrB_TIMES_FP32 ;
        *plus_monoid = GrB_PLUS_MONOID_FP32 ;
        *plus_times = GrB_PLUS_TIMES_SEMIRING_FP32 ;
    }
    else if (*atype == GrB_FP64)
    {
        *abs = GrB_ABS_FP64 ;
        *plus = GrB_PLUS_FP64 ;
        *minus = GrB_MINUS_FP64 ;
        *times = GrB_TIMES_FP64 ;
        *plus_monoid = GrB_PLUS_MONOID_FP64 ;
        *plus_times = GrB_PLUS_TIMES_SEMIRING_FP64 ;
    }
    else if (*atype == GxB_FC32)
    {
        *abs = GxB_ABS_FC32 ;
        *plus = GxB_PLUS_FC32 ;
        *minus = GxB_MINUS_FC32 ;
        *times = GxB_TIMES_FC32 ;
        *plus_monoid = GxB_PLUS_FC64_MONOID ;
        *plus_times = GxB_PLUS_TIMES_FC32 ;
    }
    else if (*atype == GxB_FC64)
    {
        *abs = GxB_ABS_FC64 ;
        *plus = GxB_PLUS_FC64 ;
        *minus = GxB_MINUS_FC64 ;
        *times = GxB_TIMES_FC64 ;
        *plus_monoid = GxB_PLUS_FC64_MONOID ;
        *plus_times = GxB_PLUS_TIMES_FC64 ;
    }
    else
    {
        printf ("type not supported\n") ;
        CHECK (false) ;
    }

    // GxB_print (*atype, 3) ;
    // GxB_print (*plus, 3) ;
    // GxB_print (*plus_monoid, 3) ;
    // GxB_print (*plus_times, 3) ;
}

//------------------------------------------------------------------------------
// move_GrB_Matrix_to_mkl
//------------------------------------------------------------------------------

// Unpack the contents of a GrB_Matrix into a MKL sparse CSR or CSC matrix.
// On output, the GrB_Matrix A still exists, but with no entries.

// If the GrB_Matrix is already in sparse format, this takes O(1) time and
// space.  Otherwise, A is first converted to sparse and then unpacked.

// A must have type GrB_FP32 (float), GrB_FP64 (double), GxB_FC32 (float
// complex), or GxB_FC64 (double complex).

// If A_mkl_transpose is false, then the CSR/CSC of the matrix A is
// imported into the A_mkl matrix as-is, and A and A_mkl have the same
// dimensions.  If true, then an m-by-n GrB_Matrix A in CSR format is
// converted into an n-by-m MKL matrix in CSC format, and an m-by-n
// GrB_Matrix A in CSC format is converted into an n-by-m MKL matrix in
// CSR format.

// mkl_sparse_*_create_* does not take ownership of the arrays passed in
// to it.  For all uses of this move_GrB_Matrix_to_mkl, the content of the
// input GrB_Matrix A is owned by the GrB_Matrix itself.  That is, none of
// the content of A is shallow.

// As a result, to free the content of the matrix, it must be moved back into
// the GrB_Matrix by move_mkl_to_GrB_Matrix and then freed with GrB_free (&A).

void move_GrB_Matrix_to_mkl
(
    GrB_Matrix A,               // GrB_Matrix to unpack to an MKL matrix
    bool A_mkl_transpose,       // resulting A_mkl is the transpose of A,
                                // in the opposite CSR/CSC format
    sparse_matrix_t *A_mkl_handle,  // newly created MKL sparse matrix
    struct matrix_descr *descr      // pointer to MKL matrix descriptor
)
{

    //--------------------------------------------------------------------------
    // break opacity to check that A has no shallow components
    //--------------------------------------------------------------------------

    CHECK (!GB_is_shallow (A)) ;    // break opacity

    //--------------------------------------------------------------------------
    // unpack the contents of the GrB_Matrix
    //--------------------------------------------------------------------------

    // only 64-bit indices are supported
    CHECK (sizeof (MKL_INT) == sizeof (GrB_Index)) ;

    // iso matrices from GraphBLAS can be unpacked as non-iso, but this gives
    // an "unfair" advantage to GraphBLAS since MKL does not support them.
    // So assert the GrB_Matrix A is not iso.
    bool iso ;
    OK (GxB_Matrix_iso (&iso, A)) ;
    CHECK (!iso) ;

    // get the format of A: by row or by column
    GxB_Format_Value format ;
    OK (GxB_get (A, GxB_FORMAT, &format)) ;
    bool grb_is_csr = (format == GxB_BY_ROW) ;

    // get the name of the type of A
    char type_name [GxB_MAX_NAME_LEN] ;
    OK (GxB_Matrix_type_name (type_name, A)) ;

    // get the size of A
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    // unpack A in CSR or CSC format into Ap, Aj, and Ax
    GrB_Index *Ap = NULL ;
    GrB_Index *Aj = NULL ;
    void *Ax = NULL ;
    GrB_Index Ap_size = 0, Aj_size = 0, Ax_size = 0 ;

    //--------------------------------------------------------------------------
    // break opacity
    //--------------------------------------------------------------------------

    #if 0
    // this approach assumed that ownership of Ap, Aj, and Ax was transfered
    // to MKL by mkl_sparse_*_create_*, but that is incorrect.  MKL does not
    // take ownership.  Instead, mkl_sparse_*_create_* leaves the ownership
    // unchanged, and internally declares these arrays as the equivalant of
    / the SuiteSparse:GraphBLAS "shallow".
    if (grb_is_csr)
    {
        // unpack A in sparse CSR format
        OK (GxB_Matrix_unpack_CSR (A, &Ap, &Aj, &Ax,
            &Ap_size, &Aj_size, &Ax_size,
            NULL, NULL, NULL)) ;
    }
    else
    {
        // unpack A in sparse CSC format
        OK (GxB_Matrix_unpack_CSC (A, &Ap, &Aj, &Ax,
            &Ap_size, &Aj_size, &Ax_size,
            NULL, NULL, NULL)) ;
    }

    // A still exists but is now empty (and hypersparse).  Just to check:
    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    CHECK (nvals == 0) ;
    #endif

    Ap = (GrB_Index *) A->p ;   // break opacity
    Aj = (GrB_Index *) A->i ;   // break opacity
    Ax = (void *) A->x ;        // break opacity

    //--------------------------------------------------------------------------
    // handle the MKL transpose
    //--------------------------------------------------------------------------

    bool mkl_is_csr = grb_is_csr ;
    if (A_mkl_transpose)
    {
        // swap nrows and ncols
        MKL_INT t = nrows ;
        nrows = ncols ;
        ncols = t ;
        // flip the CSR/CSC format
        mkl_is_csr = !grb_is_csr ;
    }

    //--------------------------------------------------------------------------
    // construct the MKL matrix and its descriptor
    //--------------------------------------------------------------------------

    if (MATCH (type_name, "float"))
    {
        // create a float sparse MKL CSR or CSC matrix
        if (mkl_is_csr)
        {
            OK (mkl_sparse_s_create_csr (A_mkl_handle, SPARSE_INDEX_BASE_ZERO,
                (MKL_INT) nrows, (MKL_INT) ncols, (MKL_INT *) Ap,
                (MKL_INT *) (Ap+1), (MKL_INT *) Aj, (float *) Ax)) ;
        }
        else
        {
            OK (mkl_sparse_s_create_csc (A_mkl_handle, SPARSE_INDEX_BASE_ZERO,
                (MKL_INT) nrows, (MKL_INT) ncols, (MKL_INT *) Ap,
                (MKL_INT *) (Ap+1), (MKL_INT *) Aj, (float *) Ax)) ;
        }
    }
    else if (MATCH (type_name, "double"))
    {
        // create a double sparse MKL CSR or CSC matrix
        if (mkl_is_csr)
        {
            OK (mkl_sparse_d_create_csr (A_mkl_handle, SPARSE_INDEX_BASE_ZERO,
                (MKL_INT) nrows, (MKL_INT) ncols, (MKL_INT *) Ap,
                (MKL_INT *) (Ap+1), (MKL_INT *) Aj, (double *) Ax)) ;
        }
        else
        {
            OK (mkl_sparse_d_create_csc (A_mkl_handle, SPARSE_INDEX_BASE_ZERO,
                (MKL_INT) nrows, (MKL_INT) ncols, (MKL_INT *) Ap,
                (MKL_INT *) (Ap+1), (MKL_INT *) Aj, (double *) Ax)) ;
        }
    }
    else if (MATCH (type_name, "float complex"))
    {
        // create a float complex sparse MKL CSR matrix
        if (mkl_is_csr)
        {
            OK (mkl_sparse_c_create_csr (A_mkl_handle, SPARSE_INDEX_BASE_ZERO,
                (MKL_INT) nrows, (MKL_INT) ncols, (MKL_INT *) Ap,
                (MKL_INT *) (Ap+1), (MKL_INT *) Aj, (MKL_Complex8 *) Ax)) ;
        }
        else
        {
            OK (mkl_sparse_c_create_csc (A_mkl_handle, SPARSE_INDEX_BASE_ZERO,
                (MKL_INT) nrows, (MKL_INT) ncols, (MKL_INT *) Ap,
                (MKL_INT *) (Ap+1), (MKL_INT *) Aj, (MKL_Complex8 *) Ax)) ;
        }
    }
    else if (MATCH (type_name, "double complex"))
    {
        // create a double complex sparse MKL CSR or CSC matrix
        if (mkl_is_csr)
        {
            OK (mkl_sparse_z_create_csr (A_mkl_handle, SPARSE_INDEX_BASE_ZERO,
                (MKL_INT) nrows, (MKL_INT) ncols, (MKL_INT *) Ap,
                (MKL_INT *) (Ap+1), (MKL_INT *) Aj, (MKL_Complex16 *) Ax)) ;
        }
        else
        {
            OK (mkl_sparse_z_create_csc (A_mkl_handle, SPARSE_INDEX_BASE_ZERO,
                (MKL_INT) nrows, (MKL_INT) ncols, (MKL_INT *) Ap,
                (MKL_INT *) (Ap+1), (MKL_INT *) Aj, (MKL_Complex16 *) Ax)) ;
        }
    }
    else
    {
        // unsupported type
        CHECK (false) ;
    }

    // construct the descriptor for A
    descr->type = SPARSE_MATRIX_TYPE_GENERAL ;
    descr->mode = SPARSE_FILL_MODE_FULL ;
    descr->diag = SPARSE_DIAG_NON_UNIT ;
}

//------------------------------------------------------------------------------
// move_mkl_to_GrB_Matrix
//------------------------------------------------------------------------------

// Export the contents of an MKL sparse CSR or CSC matrix and pack them into a
// GrB_Matrix.  On input, the GrB_Matrix A exists, and its type and dimensions
// must match the type of the MKL sparse matrix A_mkl.  The contents of A are
// replaced with the contents of the MKL sparse matrix.  This method always
// takes O(1) time and space.

// The MKL matrix may either have shallow content that is already owned by
// GraphBLAS, or MKL may have created the matrix itself.  In the latter case,
// the GrB_Matrix content of A is declared shallow, so that it is not freed
// by GrB_free (&A).  To free content created by MKL, the MKL method
// mkl_sparse_destroy (A_mkl) must be used.

// This method can be used in two use cases:
//  (1) the MKL matrix was created by move_GrB_Matrix_to_mkl.  The content
//      is owned by GraphBLAS, not MKL, even when it resides inside the MKL
//      matrix.  The input parameter mkl_owns_the_matrix is false.
//  (2) MKL created the matrix itself (from the output of mkl_sparse_sp2m
//      for example).  The content is owned by MKL, even after moving it into
//      the GrB_Matrix.  The input parameter mkl_owns_the_matrix is true.

void move_mkl_to_GrB_Matrix
(
    GrB_Matrix A,               // GrB_Matrix to pack from an MKL sparse matrix
    bool A_mkl_transpose,       // if true transpose the matrix
    sparse_matrix_t *A_mkl_handle,  // MKL sparse matrix
    bool mkl_owns_the_matrix,   // break opacity: if true then MKL owns the
    // matrix so the contents of the GrB_Matrix A must be declared as
    // shallow.  This requires opacity to be broken.
    bool jumbled
)
{

    //--------------------------------------------------------------------------
    // get the type and size of A
    //--------------------------------------------------------------------------

    // get the name of the type of A
    char type_name [GxB_MAX_NAME_LEN] ;
    OK (GxB_Matrix_type_name (type_name, A)) ;

    // get the size of A
    GrB_Index nrows2, ncols2 ;
    OK (GrB_Matrix_nrows (&nrows2, A)) ;
    OK (GrB_Matrix_ncols (&ncols2, A)) ;

    // get the format of A: by row or by column
    GxB_Format_Value format ;
    OK (GxB_get (A, GxB_FORMAT, &format)) ;
    bool grb_is_csr = (format == GxB_BY_ROW) ;

    //--------------------------------------------------------------------------
    // handle the MKL transpose
    //--------------------------------------------------------------------------

    bool mkl_is_csr = grb_is_csr ;
    if (A_mkl_transpose)
    {
        // flip the CSR/CSC of the MKL matrix
        mkl_is_csr = !grb_is_csr ;
    }

    //--------------------------------------------------------------------------
    // export the contents of the MKL sparse CSR or CSC matrix
    //--------------------------------------------------------------------------

    // only 64-bit indices are supported
    CHECK (sizeof (MKL_INT) == sizeof (GrB_Index)) ;

    sparse_index_base_t indexing ;
    GrB_Index nrows, ncols ;
    GrB_Index *Ap = NULL, *Ap1 = NULL ;
    GrB_Index *Aj = NULL ;
    void *Ax = NULL ;
    size_t typesize ;

    if (MATCH (type_name, "float"))
    {
        // export a float sparse MKL CSR or CSC matrix
        typesize = sizeof (float) ;
        if (mkl_is_csr)
        {
            OK (mkl_sparse_s_export_csr (*A_mkl_handle, &indexing,
                (MKL_INT *) &nrows, (MKL_INT *) &ncols, (MKL_INT **) &Ap,
                (MKL_INT **) &(Ap1), (MKL_INT **) &Aj, (float **) &Ax)) ;
        }
        else
        {
            OK (mkl_sparse_s_export_csc (*A_mkl_handle, &indexing,
                (MKL_INT *) &nrows, (MKL_INT *) &ncols, (MKL_INT **) &Ap,
                (MKL_INT **) &(Ap1), (MKL_INT **) &Aj, (float **) &Ax)) ;
        }
    }
    else if (MATCH (type_name, "double"))
    {
        // export a double sparse MKL CSR or CSC matrix
        typesize = sizeof (double) ;
        if (mkl_is_csr)
        {
            OK (mkl_sparse_d_export_csr (*A_mkl_handle, &indexing,
                (MKL_INT *) &nrows, (MKL_INT *) &ncols, (MKL_INT **) &Ap,
                (MKL_INT **) &(Ap1), (MKL_INT **) &Aj, (double **) &Ax)) ;
        }
        else
        {
            OK (mkl_sparse_d_export_csc (*A_mkl_handle, &indexing,
                (MKL_INT *) &nrows, (MKL_INT *) &ncols, (MKL_INT **) &Ap,
                (MKL_INT **) &(Ap1), (MKL_INT **) &Aj, (double **) &Ax)) ;
        }
    }
    else if (MATCH (type_name, "float complex"))
    {
        // export a float complex sparse MKL CSR or CSC matrix
        typesize = sizeof (MKL_Complex8) ;
        if (mkl_is_csr)
        {
            OK (mkl_sparse_c_export_csr (*A_mkl_handle, &indexing,
                (MKL_INT *) &nrows, (MKL_INT *) &ncols, (MKL_INT **) &Ap,
                (MKL_INT **) &(Ap1), (MKL_INT **) &Aj, (MKL_Complex8 **) &Ax)) ;
        }
        else
        {
            OK (mkl_sparse_c_export_csc (*A_mkl_handle, &indexing,
                (MKL_INT *) &nrows, (MKL_INT *) &ncols, (MKL_INT **) &Ap,
                (MKL_INT **) &(Ap1), (MKL_INT **) &Aj, (MKL_Complex8 **) &Ax)) ;
        }
    }
    else if (MATCH (type_name, "double complex"))
    {
        // export a double complex sparse MKL CSR or CSC matrix
        typesize = sizeof (MKL_Complex16) ;
        if (mkl_is_csr)
        {
            OK (mkl_sparse_z_export_csr (*A_mkl_handle, &indexing,
                (MKL_INT *) &nrows, (MKL_INT *) &ncols, (MKL_INT **) &Ap,
                (MKL_INT **) &(Ap1), (MKL_INT **) &Aj, (MKL_Complex16 **) &Ax));
        }
        else
        {
            OK (mkl_sparse_z_export_csc (*A_mkl_handle, &indexing,
                (MKL_INT *) &nrows, (MKL_INT *) &ncols, (MKL_INT **) &Ap,
                (MKL_INT **) &(Ap1), (MKL_INT **) &Aj, (MKL_Complex16 **) &Ax));
        }
    }
    else
    {
        // unsupported type
        CHECK (false) ;
    }

    // only contigous matrices are supported
    CHECK (Ap1 == Ap+1) ;

    // ensure the sizes are correct
    if (A_mkl_transpose)
    {
        // if the MKL is in CSC then the GrB matri is in CSR,
        // and visa-versa
        CHECK (nrows == ncols2) ;
        CHECK (ncols == nrows2) ;
    }
    else
    {
        // MKL and GrB are in the same format
        CHECK (nrows == nrows2) ;
        CHECK (ncols == ncols2) ;
    }

    //--------------------------------------------------------------------------
    // pack the contents into the GrB_Matrix A
    //--------------------------------------------------------------------------

    GrB_Index nvec = grb_is_csr ? nrows : ncols ;
    GrB_Index nvals = Ap [nvec] ;
    GrB_Index Aj_size = (nvals) * sizeof (GrB_Index) ;
    GrB_Index Ax_size = (nvals) * typesize ;

    if (grb_is_csr)
    {
        // pack the contents into a sparse CSR GrB_Matrix
        GrB_Index Ap_size = (nvec+1) * sizeof (GrB_Index) ;
        OK (GxB_Matrix_pack_CSR (A, &Ap, &Aj, &Ax, Ap_size, Aj_size, Ax_size,
            /* MKL matrices are never iso: */ false,
            /* the MKL matrix might be jumbled: */ jumbled,
            NULL)) ;
    }
    else
    {
        // pack the contents into a sparse CSC GrB_Matrix
        GrB_Index Ap_size = (nvec+1) * sizeof (GrB_Index) ;
        OK (GxB_Matrix_pack_CSC (A, &Ap, &Aj, &Ax, Ap_size, Aj_size, Ax_size,
            /* MKL matrices are never iso: */ false,
            /* the MKL matrix might be jumbled: */ jumbled,
            NULL)) ;
    }

    //--------------------------------------------------------------------------
    // break opacity
    //--------------------------------------------------------------------------

    if (mkl_owns_the_matrix)
    {
        // declare contents of the CSR/CSC GrB_Matrix A as shallow 
        A->p_shallow = true ;   // break opacity
        A->i_shallow = true ;   // break opacity
        A->x_shallow = true ;   // break opacity
    }
    else
    {
        OK (mkl_sparse_destroy (*A_mkl_handle)) ;
    }

    //--------------------------------------------------------------------------
    // A still exists and is populated.  Just to check:
    //--------------------------------------------------------------------------

    GrB_Index nvals2 ;
    OK (GrB_Matrix_nvals (&nvals2, A)) ;
    CHECK (nvals == nvals2) ;
}


//------------------------------------------------------------------------------
// convert_matrix_to_type: A = (atype) D*S
//------------------------------------------------------------------------------

// The sparsity format is unchanged.

void convert_matrix_to_type
(
    GrB_Matrix *A,
    GrB_Type atype,
    GrB_Matrix D,
    GrB_Matrix S
)
{
    double t = omp_get_wtime ( ) ;

    GrB_Type stype ;
    OK (GxB_Matrix_type (&stype, S)) ;

    char type_name [GxB_MAX_NAME_LEN] ;
    OK (GxB_Type_name (type_name, stype)) ;
    printf ("\nconvert matrix from %s", type_name) ;
    OK (GxB_Type_name (type_name, atype)) ;
    printf (" to %s\n", type_name) ;

    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, S)) ;
    OK (GrB_Matrix_ncols (&ncols, S)) ;
    GxB_Format_Value format ;
    int sparsity ;
    OK (GxB_get (S, GxB_FORMAT, &format)) ;
    OK (GxB_get (S, GxB_SPARSITY_STATUS, &sparsity)) ;

    OK (GrB_Matrix_new (A, atype, nrows, ncols)) ;
    OK (GxB_set (*A, GxB_FORMAT, format)) ;
    OK (GxB_set (*A, GxB_SPARSITY_CONTROL, sparsity)) ;

    // A = (atype) D*S
    GxB_set (GxB_BURBLE, true) ; 
    OK (GrB_mxm (*A, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, D, S, NULL)) ;
    GxB_set (GxB_BURBLE, false) ; 

    // GxB_print (*A,2) ;
    // GxB_print (S,2) ;

    t = omp_get_wtime ( ) - t ;
    printf ("conversion time: %g sec\n", t) ;
}

//------------------------------------------------------------------------------
// print_results
//------------------------------------------------------------------------------

#if 0
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
#endif

void print_results (timing_t mkl_results, timing_t grb_results)
{
    char *fmt ;

    if (mkl_results.twarmup > 0 || grb_results.twarmup > 0)
    {
        // with warmup
        fmt = " MKL: 1st %10.4f 2nd %10.4f GrB: %10.4f " ;
        fprintf (stderr, fmt,
        mkl_results.twarmup, mkl_results.tmedian, grb_results.tmedian) ;
         printf (        fmt,
        mkl_results.twarmup, mkl_results.tmedian, grb_results.tmedian) ;

        print_speedup ("1st:", mkl_results.twarmup, grb_results.tmedian) ;
        print_speedup ("2nd:", mkl_results.tmedian, grb_results.tmedian) ;

        if (!just_practicing)
        {
            if (grb_results.tmedian > 1.2 * mkl_results.tmedian &&
                mkl_results.twarmup >       mkl_results.tmedian)
            {
                fmt = "breakeven: %7.1f" ;
                double k = (mkl_results.twarmup - mkl_results.tmedian) /
                           (grb_results.tmedian - mkl_results.tmedian) ;
                if (k >= 1)
                {
                    fprintf (stderr, fmt, k) ;
                     printf (        fmt, k) ;
                }
            }
        }
    }
    else
    {
        // no warmup
        fmt = " MKL: %10.4f GrB: %10.4f " ;
        fprintf (stderr, fmt, mkl_results.tmedian, grb_results.tmedian) ;
         printf (        fmt, mkl_results.tmedian, grb_results.tmedian) ;
        print_speedup ("", mkl_results.tmedian, grb_results.tmedian) ;
    }
    fprintf (stderr, "\n") ;
     printf (        "\n") ;
}

void print_speedup (char *what, double mkl_time, double grb_time)
{
    if (just_practicing) return ;
    char *fmt ;
    double rel_time = mkl_time / grb_time ;
    if (rel_time >= 2)
    {
        // GrB is 2x faster, or more, than MKL
        fmt = "[%s SPEEDUP:  %7.1f        ] " ;
    }
    else if (rel_time > 1.2)
    {
        // GrB is 1.2 to 2x faster than MKL
        fmt = "[%s speedup:  %7.1f        ] " ;
    }
    else if (rel_time < 0.5)
    {
        // MKL is 2x faster, or more, than GrB
        rel_time = grb_time / mkl_time ;
        fmt = "[%s SLOWDOWN:         %7.1f] " ;
    }
    else if (rel_time < 1/1.2)
    {
        // MKL is 1.2 to 2x faster than GrB
        rel_time = grb_time / mkl_time ;
        fmt = "[%s slowdown:         %7.1f] " ;
    }
    else
    {
        // MKL and GrB are within 20% of each other; call it the same
        fmt = "[%s                  =       ] " ;
    }
    fprintf (stderr, fmt, what, rel_time) ;
     printf (        fmt, what, rel_time) ;
}

