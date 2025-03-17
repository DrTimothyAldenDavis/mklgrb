// mtx_to_triplet: convert an ascii Matrix Market file to triplet,
// for old GraphBLAS/Demo/Source/read_matrix.
//
// usage:
//
// mtx_to_triplet infile.mtx outfile.triplet

#include <stdio.h>
#include <LAGraph.h>
#include <LAGraphX.h>
#include <GraphBLAS.h>

#define MATCH(s,t) (strcmp (s,t) == 0)
#define OK(method)                                                      \
{                                                                       \
    int res = method ;                                                  \
    if (res != 0)                                                       \
    {                                                                   \
        fprintf (stderr, "Fail: %d, line %d\n", res, __LINE__) ;        \
        abort ( ) ;                                                     \
    }                                                                   \
}

#define CHECK(x) OK((x) ? 0 : 1)

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // get the filenames
    //--------------------------------------------------------------------------

    #define LEN 1024
    char infile [LEN], outfile [LEN+16], temp [LEN] ;
    if (argc == 2)
    {
        // usage: mtxconvert infile.mtx, creates outfile.triplet
        strncpy (infile, argv [1], LEN) ;
        infile [LEN-1] = '\0' ;
        strncpy (temp, argv [1], LEN) ;
        temp [LEN-1] = '\0' ;
        for (int k = strlen (temp) - 1 ; k >= 0 ; k--)
        {
            if (temp [k] == '.')
            {
                temp [k] = '\0' ;
                break ;
            }
        }
        snprintf (outfile, LEN+16, "%s.triplet", temp) ;
    }
    else if (argc == 3)
    {
        // usage: mtxconvert infile.mtx outfile.triplet
        strncpy (infile, argv [1], LEN) ;
        infile [LEN-1] = '\0' ;
        strncpy (outfile, argv [2], LEN) ;
        outfile [LEN-1] = '\0' ;
    }
    else
    {
        fprintf (stderr, "usage: mtxconvert infile.mtx outfile.triplet") ;
        abort ( ) ;
    }
    printf ("infile: [%s] outfile [%s]\n", infile, outfile) ;

    //--------------------------------------------------------------------------
    // startup LAGraph and GraphBLAS and check versions
    //--------------------------------------------------------------------------

    char msg [LAGRAPH_MSG_LEN] ;
    OK (LAGraph_Init (msg)) ;
    char *library ;
    int ver [3], v ;
    OK (GxB_get (GxB_LIBRARY_NAME, &library)) ;
    OK (GxB_get (GxB_LIBRARY_VERSION, ver)) ;
    printf ("GrB include: %s v%d.%d.%d\n", GxB_IMPLEMENTATION_NAME,
        GxB_IMPLEMENTATION_MAJOR, GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB) ;
    printf ("GrB runtime: %s v%d.%d.%d\n", library, ver [0], ver [1], ver [2]) ;
    int v_include = GxB_VERSION (GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR, GxB_IMPLEMENTATION_SUB) ;
    int v_runtime = GxB_VERSION (ver [0], ver [1], ver [2]) ;
    if (!MATCH ("SuiteSparse:GraphBLAS", library) ||
        !MATCH ("SuiteSparse:GraphBLAS", GxB_IMPLEMENTATION_NAME) ||
        (v_include != v_runtime) || v_runtime < GxB_VERSION (6,0,3))
    {
        fprintf (stderr, "SuiteSparse:GraphBLAS v6.0.3 or later required\n") ;
        abort ( ) ;
    }

    //--------------------------------------------------------------------------
    // read the *.mtx file
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, B = NULL ;
    GrB_Type atype ;
    GrB_Index nvals ;
    FILE *f = fopen (infile, "r") ;
    CHECK (f != NULL) ;
    double t = omp_get_wtime ( ) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    t = omp_get_wtime ( ) - t ;
    printf ("read *.mtx time:    %g sec, nvals %g\n", t, (double) nvals) ;
    fclose (f) ;
    // GxB_print (A, 2) ;

    //--------------------------------------------------------------------------
    // create the *.triplet file
    //--------------------------------------------------------------------------

    f = fopen (outfile, "w") ;

    // get the typename
    char typename [LAGRAPH_MAX_NAME_LEN] ;
    OK (LAGraph_Matrix_TypeName (typename, A, msg)) ;

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
    OK (LAGraph_Finalize (msg)) ;
}

