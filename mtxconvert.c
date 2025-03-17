// mtxconvert: convert an ascii Matrix Market file to binary *.lagraph
//
// usage:
//
// mtxconvert infile.mtx outfile.lagraph

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
        // usage: mtxconvert infile.mtx, creates outfile.lagraph
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
        snprintf (outfile, LEN+16, "%s.lagraph", temp) ;
    }
    else if (argc == 3)
    {
        // usage: mtxconvert infile.mtx outfile.lagraph
        strncpy (infile, argv [1], LEN) ;
        infile [LEN-1] = '\0' ;
        strncpy (outfile, argv [2], LEN) ;
        outfile [LEN-1] = '\0' ;
    }
    else
    {
        fprintf (stderr, "usage: mtxconvert infile.mtx outfile.lagraph") ;
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
    // create the *.lagraph file
    //--------------------------------------------------------------------------

    GrB_Descriptor desc = NULL ;
    OK (GrB_Descriptor_new (&desc)) ;
    OK (GxB_set (desc, GxB_COMPRESSION, GxB_COMPRESSION_LZ4HC + 9)) ;

    void *blob = NULL ;
    size_t blob_size ;
    t = omp_get_wtime ( ) ;
    OK (GxB_Matrix_serialize (&blob, (GrB_Index *) &blob_size, A, desc)) ;
    t = omp_get_wtime ( ) - t ;
    printf ("serialize time:     %g sec\n", t) ;

    t = omp_get_wtime ( ) ;
    f = fopen (outfile, "w") ;
    CHECK (f != NULL) ;

    // remove the .mtx from the infile
    int len = strlen (infile) ;
    infile [len-4] = '\0' ;

    // get the typename
    char typename [LAGRAPH_MAX_NAME_LEN] ;
    OK (LAGraph_Matrix_TypeName (typename, A, msg)) ;

    // write the header for a single matrix
    OK (LAGraph_SWrite_HeaderStart (f, infile, msg)) ;
    OK (LAGraph_SWrite_HeaderItem (f, LAGraph_matrix_kind, "A",
        typename, 0, blob_size, msg)) ;
    OK (LAGraph_SWrite_HeaderEnd (f, msg)) ;

    // write the binary blob to the file then free the blob
    OK (LAGraph_SWrite_Item (f, blob, blob_size, msg)) ;
    LAGraph_Free (&blob) ;
    fclose (f) ;
    t = omp_get_wtime ( ) - t ;
    printf ("write lagraph time: %g sec\n", t) ;

    //--------------------------------------------------------------------------
    // read the outfile back in, just to check the result
    //--------------------------------------------------------------------------

    // if (nvals < 1e9)
    {
        t = omp_get_wtime ( ) ;
        f = fopen (outfile, "r") ;
        CHECK (f != NULL) ;

        char *collection ;
        LAGraph_Contents *Contents ;
        GrB_Index ncontents ;
        OK (LAGraph_SRead (f, &collection, &Contents, &ncontents, msg)) ;
        CHECK (collection != NULL) ;
        CHECK (ncontents == 1) ;
        fclose (f) ;

        // convert the contents to a matrix B
        void *blob2 = Contents [0].blob ;
        size_t blob_size2 = Contents [0].blob_size ;
        CHECK (blob_size == blob_size2) ;
        OK (GrB_Matrix_deserialize (&B, atype, blob2, blob_size2)) ;
        t = omp_get_wtime ( ) - t ;
        printf ("read lagraph time:  %g sec\n", t) ;

        // ensure the matrices A and B are the same
        // GxB_print (B,2) ;
        bool ok ;
        OK (LAGraph_Matrix_IsEqual (&ok, A, B, msg)) ;
        CHECK (ok) ;
        OK (GrB_free (&B)) ;
    }

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    OK (GrB_free (&A)) ;
    OK (LAGraph_Finalize (msg)) ;
}

