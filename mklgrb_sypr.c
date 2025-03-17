// benchmark MKL sparse and SuiteSparse:GraphBLAS: sypr

// usage:
//      mklgrb_sypr filename.lagraph

// The input matrix must be symmetric; this condition is not checked

#ifndef MKL_ILP64
#define MKL_ILP64
#endif

#include "mklgrb.h"

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // startup LAGraph and load the sparse matrix S in CSR format
    //--------------------------------------------------------------------------

    startup_LAGraph ( ) ;
    CHECK (argc >= 2) ;
    char *filename = argv [1] ;
    GrB_Matrix S = NULL, D = NULL, Right = NULL, Left = NULL, T = NULL,
        P = NULL, M = NULL ;
    GrB_Vector Seed = NULL, Perm = NULL, Ramp = NULL ;
    load_matrix ("sypr", filename, &S, &D) ;

    // make sure S is symmetric: S = S+S'
    OK (GrB_eWiseAdd (S, NULL, NULL, GrB_PLUS_FP64, S, S, GrB_DESC_T1)) ;
    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, S)) ;

    // S = S + D, to ensure it is non-iso
    OK (GrB_eWiseAdd (S, NULL, NULL, GrB_PLUS_FP64, S, D, NULL)) ;
    fprintf (stderr, "After S = S+S'+D: nvals = %g million\n",
        1e-6 * (double) nvals) ;
     printf (        "After S = S+S'+D: nvals = %g million\n",
        1e-6 * (double) nvals) ;

    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, S)) ;
    OK (GrB_Matrix_ncols (&ncols, S)) ;
    GrB_Index n = nrows ;

    //--------------------------------------------------------------------------
    // get the # of threads for each trial
    //--------------------------------------------------------------------------

    int nthreads_default = omp_get_max_threads ( ) ;
    fprintf (stderr, "Default # threads: %d\n", nthreads_default) ;
    printf ("Default # threads: %d\n", nthreads_default) ;
    int nt [4], nthreads_trials ;
    char msg [LAGRAPH_MSG_LEN ] ;

#if 0
    nthreads_trials = 3 ;
    nt [0] = 1 ;
    nt [1] = nthreads_default / 2 ;
    nt [2] = nthreads_default ;
#else
    // just run with default # of threads
    nthreads_trials = 1 ;
    nt [0] = nthreads_default ;
#endif

    //--------------------------------------------------------------------------
    // run some benchmarks with the matrix in CSR format
    //--------------------------------------------------------------------------

    // GxB_print (D, 2) ;
    // GxB_print (S, 2) ;

    GrB_Index nvals2 = 1000 ;
    GrB_Index thin = 8 ;

    // create a ncols-by-thin matrix Right with ~nvals2 entries
    double d = nvals2 / ((double) ncols * thin) ;
    OK (LAGraph_Random_Matrix (&Right, GrB_FP64, ncols, thin, d, 0, msg)) ;
    OK (GxB_set (Right, GxB_FORMAT, GxB_BY_ROW)) ;
    OK (GxB_set (Right, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    // GxB_print (Right, 2) ;

    // create a thin-by-nrows matrix Left with ~nvals2 entries
    d = nvals2 / ((double) nrows * thin) ;
    OK (LAGraph_Random_Matrix (&Left, GrB_FP64, thin, nrows, d, 1, msg)) ;
    OK (GxB_set (Left, GxB_FORMAT, GxB_BY_ROW)) ;
    OK (GxB_set (Left, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    // GxB_print (Left, 2) ;

    just_practicing = true ;
    benchmark_sypr ("R'SRwarm", Right, S, true, NULL, false) ;
    just_practicing = false ;
    fprintf (stderr, "\n") ;
     printf (        "\n") ;

    // sparse sypr: compress to tiny 8-by-8 result matrix C
    benchmark_sypr ("C=R'*S*R", Right, S, true, NULL, false) ;
    benchmark_sypr ("C=R'*S*R", Right, S, true, NULL, true) ;
    benchmark_sypr ("C=L*S*L'", Left, S, false, NULL, false) ;
    benchmark_sypr ("C=L*S*L'", Left, S, false, NULL, true) ;
    GrB_free (&Right) ;
    GrB_free (&Left) ;

    // sparse sypr: diagonal scaling
    benchmark_sypr ("C=D'*S*D", D, S, true, NULL, false) ;
    benchmark_sypr ("C=D'*S*D", D, S, true, NULL, true) ;
    benchmark_sypr ("C=D*S*D'", D, S, false, NULL, false) ;
    benchmark_sypr ("C=D*S*D'", D, S, false, NULL, true) ;

    // create a random permutation matrix P
    OK (GrB_Vector_new (&Seed, GrB_UINT64, n)) ;
    OK (GrB_assign (Seed, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    OK (LAGraph_Random_Seed (Seed, 0, msg)) ;
    OK (GrB_Vector_new (&Perm, GrB_UINT64, n)) ;
    OK (GxB_Vector_sort (NULL, Perm, GrB_LT_UINT64, Seed, NULL)) ;
    OK (GrB_free (&Seed)) ;
    // OK (GxB_print (Perm, 2)) ;
    uint64_t *Pj, *Pp ;
    size_t Pj_size, Pp_size ;
    OK (GxB_Vector_unpack_Full (Perm, (void **) &Pj, &Pj_size, NULL, NULL)) ;
    GrB_free (&Perm) ;

    // Ramp = 0:n
    OK (GrB_Vector_new (&Ramp, GrB_INT64, n+1)) ;
    OK (GrB_assign (Ramp, NULL, NULL, 0, GrB_ALL, n+1, NULL)) ;
    OK (GrB_apply (Ramp, NULL, NULL, GrB_ROWINDEX_INT64, Ramp, 0, NULL)) ;
    // OK (GxB_print (Ramp, 2)) ;
    OK (GxB_Vector_unpack_Full (Ramp, (void **) &Pp, &Pp_size, NULL, NULL)) ;
    GrB_free (&Ramp) ;

    OK (GrB_Matrix_new (&P, GrB_FP64, n, n)) ;
    double *Px = malloc (sizeof (double))  ;
    Px [0] = 1.0 ;
    uint64_t Pj0 = Pj [0] ;
    OK (GxB_Matrix_pack_CSR (P, &Pp, &Pj, (void **) &Px, Pp_size, Pj_size,
        sizeof (double), true, false, NULL)) ;
    // OK (GxB_print (P, 2)) ;
    // make sure P is not iso
    OK (GrB_Matrix_setElement (P, 0, 0, Pj0)) ;
    OK (GrB_Matrix_setElement (P, 1, 0, Pj0)) ;
    // OK (GxB_print (P, 2)) ;

    // sparse sypr: permutation
    benchmark_sypr ("C=P'*S*P", P, S, true,  (GrB_Index *) P->i, false) ;
    benchmark_sypr ("C=P'*S*P", P, S, true,  (GrB_Index *) P->i, true) ;
    benchmark_sypr ("C=P*S*P'", P, S, false, (GrB_Index *) P->i, false) ;
    benchmark_sypr ("C=P*S*P'", P, S, false, (GrB_Index *) P->i, true) ;
    GrB_free (&P) ;

    // create a 2-to-1 coarsening matrix M of size n-by-n/2
    GrB_Index m = n / 2 ;
    OK (GrB_Matrix_new (&M, GrB_FP64, n, m)) ;
    for (int64_t k = 0 ; k < m ; k++)
    {
        // M (2*k  , k) = 1.2
        // M (2*k+1, k) = 0.5
        OK (GrB_Matrix_setElement (M, 1.2, 2*k, k)) ;
        OK (GrB_Matrix_setElement (M, 0.5, 2*k+1, k)) ;
    }
    OK (GrB_wait (M, GrB_MATERIALIZE)) ;

    // sparse sypr: coarsen
    benchmark_sypr ("C=M'*S*M", M, S, true, NULL, false) ;
    benchmark_sypr ("C=M'*S*M", M, S, true, NULL, true) ;
    GrB_free (&M) ;

    //--------------------------------------------------------------------------
    // free everything and finalize LAGraph
    //--------------------------------------------------------------------------

    set_nthreads (nthreads_default) ;
    GrB_free (&S) ;
    GrB_free (&D) ;
    finish_LAGraph ( ) ;
}

