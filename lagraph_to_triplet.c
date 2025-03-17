
// usage:
//      lagraph_to_triplet filename.lagraph filename.triplet

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
    char msg [LAGRAPH_MSG_LEN] ;
    CHECK (argc >= 3) ;
    char *filename = argv [1] ;
    GrB_Matrix *Set = NULL, A = NULL, D = NULL ;
    GrB_Index nmatrices = 0 ;
    GrB_Index nvals = 0 ;
    char *collection ;
    OK (LAGraph_SLoadSet (filename, &Set, &nmatrices, &collection, msg)) ;
    A = Set [0] ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    GrB_Type atype ;
    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;

    //--------------------------------------------------------------------------
    // extract the tuples and write to stdout
    //--------------------------------------------------------------------------

    FILE *f = fopen (argv [2], "w") ;

    // get the typename
    char *typename ;
    OK (LAGraph_TypeName (&typename, atype, msg)) ;

    GrB_Index *I = (GrB_Index *) LAGraph_Malloc (nvals, sizeof (GrB_Index)) ;
    GrB_Index *J = (GrB_Index *) LAGraph_Malloc (nvals, sizeof (GrB_Index)) ;
    CHECK (J != NULL) ;
    CHECK (I != NULL) ;

    if (MATCH (typename, "bool"))
    {
        bool *X = (bool *) LAGraph_Malloc (nvals, sizeof (bool)) ;
        OK (GrB_Matrix_extractTuples_BOOL (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %d\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }

    else if (MATCH (typename, "int8"))
    {
        int8_t *X = (int8_t *) LAGraph_Malloc (nvals, sizeof (int8_t)) ;
        OK (GrB_Matrix_extractTuples_INT8 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %d\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }
    else if (MATCH (typename, "int16"))
    {
        int16_t *X = (int16_t *) LAGraph_Malloc (nvals, sizeof (int16_t)) ;
        OK (GrB_Matrix_extractTuples_INT16 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %d\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }
    else if (MATCH (typename, "int32"))
    {
        int32_t *X = (int32_t *) LAGraph_Malloc (nvals, sizeof (int32_t)) ;
        OK (GrB_Matrix_extractTuples_INT32 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %d\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }
    else if (MATCH (typename, "int64"))
    {
        int64_t *X = (int64_t *) LAGraph_Malloc (nvals, sizeof (int64_t)) ;
        OK (GrB_Matrix_extractTuples_INT64 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %ld\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }

    else if (MATCH (typename, "uint8"))
    {
        uint8_t *X = (uint8_t *) LAGraph_Malloc (nvals, sizeof (uint8_t)) ;
        OK (GrB_Matrix_extractTuples_UINT8 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %u\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }
    else if (MATCH (typename, "uint16"))
    {
        uint16_t *X = (uint16_t *) LAGraph_Malloc (nvals, sizeof (uint16_t)) ;
        OK (GrB_Matrix_extractTuples_UINT16 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %u\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }
    else if (MATCH (typename, "uint32"))
    {
        uint32_t *X = (uint32_t *) LAGraph_Malloc (nvals, sizeof (uint32_t)) ;
        OK (GrB_Matrix_extractTuples_UINT32 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %u\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }
    else if (MATCH (typename, "uint64"))
    {
        uint64_t *X = (uint64_t *) LAGraph_Malloc (nvals, sizeof (uint64_t)) ;
        OK (GrB_Matrix_extractTuples_UINT64 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %lu\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }

    else if (MATCH (typename, "float"))
    {
        float *X = (float *) LAGraph_Malloc (nvals, sizeof (float)) ;
        OK (GrB_Matrix_extractTuples_FP32 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %16.10e\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }
    else if (MATCH (typename, "double"))
    {
        double *X = (double *) LAGraph_Malloc (nvals, sizeof (double)) ;
        OK (GrB_Matrix_extractTuples_FP64 (I, J, X, &nvals, A)) ;
        for (int64_t k = 0 ; k < nvals ; k++)
        {
            fprintf (f, "%lu %lu %32.16e\n", I [k], J [k], X [k]) ;
        }
        LAGraph_Free ((void **) &X) ;
    }
    else
    {
        CHECK (0) ;
    }

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    fclose (f) ;
    OK (GrB_free (&A)) ;
    LAGraph_Free ((void **) &I) ;
    LAGraph_Free ((void **) &J) ;
    finish_LAGraph ( ) ;
}

