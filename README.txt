mklgrb: Benchmark for MKL and GrB.  SuiteSparse:GraphBLAS v6.1.3 is required,
and see the tagged version of LAGraph, dated Dec 31, 2021)

Last updated: Jan 4, 2021.  Tim Davis, davis@tamu.edu.

See Summary/Jan4_short_summary.txt for a very short summary of the latest
benchmark results.

To run this benchmark (please ask me for help if you need more details):

(1) Ensure that LAGraph and GraphBLAS are in ../LAGraph and ../GraphBLAS,
relative to this directory.  Compile GraphBLAS and then LAGraph.
Run the LAGraph tests to make sure everything works.

(2) download the following matrices from sparse.tamu.edu in
Matrix Market form, uncompress them, and ensure that following
files exist:

    ../MM/Freescale/Freescale2/Freescale2.mtx
    ../MM/GAP/GAP-kron/GAP-kron.mtx
    ../MM/GAP/GAP-road/GAP-road.mtx
    ../MM/GAP/GAP-twitter/GAP-twitter.mtx
    ../MM/GAP/GAP-urand/GAP-urand.mtx
    ../MM/GAP/GAP-web/GAP-web.mtx
    ../MM/LAW/indochina-2004/indochina-2004.mtx
    ../MM/ND/nd12k/nd12k.mtx
    ../MM/ND/nd24k/nd24k.mtx
    ../MM/ND/nd3k/nd3k.mtx
    ../MM/ND/nd6k/nd6k.mtx
    ../MM/SNAP/com-Orkut/com-Orkut.mtx
    ../MM/SNAP/roadNet-CA/roadNet-CA.mtx
    ../MM/SNAP/soc-Pokec/soc-Pokec.mtx

    The simplest way to do this is to download ssget as ../ssget, and use its
    Java interface to get the matrices.  Then put a symbolic link of ../MM to
    point to ../ssget/MM.  The Mac and Dell laptops are not able to run the
    GAP matrices so those are not yet in the dobench script.

(3) compile the benchmark and helper commands, via:

        make

(4) run the doconvert script to create the binary *.lagraph files from
    the *.mtx files.  This only needs to be done once:

        ./doconvert

    It takes a while to read in the *.mtx files; please be patient.
    Reading the compressed, binary *.lagraph is up to 1000x faster,
    which is why this step is essential to speed up repeated benchmarks
    with the same set of matrix files.

(5) run the benchmark script:

        for a system with little RAM (16GB or so):
        ./dobench > dobench1_output.txt

        for a system with a lot of RAM (256GB or so):
        ./dobench2 > dobench2_output.txt

    Detailed output of the mklgrb program goes the stdout. 
    A short summary is printed on stderr.

    To understand the output:  each line starts with an optional description,
    then the # of threads (say 4 in this example below).  Then the method is
    displayed, then type type (d:double, s:single, c:single complex, z:double
    complex).  Next, the run time of MKL in seconds then GrB in seconds.

    The mkl_sparse_*_mv and mkl_sparse_d_mm methods are benchmarked:

         y+=S_*x            S is sparse in CSR form ("S_"), x and y are dense
         y=S_*x                 vectors.  a and b are the alpha & beta scalars.
         y=a*S_*x               y+=S*x mean b=1 and a=1, y=S*x means b=0, a=1, etc.
         y=b*y+a*S_*x 

         y+=S|*x            ditto but with S sparse in CSC form ("S|").
         y=S|*x       
         y=a*S|*x     
         y=b*y+a*S|*x 

         C+=S_*B (| n)      S is sparse in CSR form, C and B are dense
                                matrices in column layout (written as "| n)
                                with n columns

         C+=S_*B (_ n)      S is sparse in CSR form, C and B are dense
                                matrices in row layout with n columns ("(_ n)")

         C+=S|*B (| n)      S is sparse in column form, C and B are dense
                                matrices in column layout with n columns

         C+=S|*B (_ n)      S is CSC, C and B dense in row layout, n columns

    The mkl_sparse_*_sp2m method are benchmarked.  If the matrix is small enough:

        C = A*A
        C = A*A'
        C = A'*A
        C = A'*A'

    For larger matrices, two extra matrices are created: Left and Right.
    These matrices are very sparse (1000 entries) and have 8 rows/columns
    respectively:

        C=Left*A    
        C=A'*Left'  
        C=Left*A'   
        C=A*Left'   
        C=A*Right   
        C=Right'*A' 
        C=A'*Right  
        C=Right'*A  

    The mkl_sparse_*_add method is benchmarked:  B and A are square.
    A is the original input matrix. B is randomly generated with about 10%
    the # of nonzeros as A.

        C=B+A       
        C=2*B+A     
        C=B'+A      
        C=2*B'+A    
        C=A+B       
        C=2*A+B     
        C=A+A       
        C=2*A+A     
        C=A'+B      
        C=2*A'+B    
        C=A'+A     
        C=2*A'+A    

    The mkl_sparse_sypr method is benchmarked.  R and L are the same n-by-8
    and 8-by-n Right and Left matrices.  D is diagonal. P is permutation matrix.
    M is a "coarsening" matrix, of size n-by-(n/2), with kth column entries
    in M(2k,k) and (2k+1,k).

        C=R'*S*R 
        C=L*S*L' 
        C=D'*S*D 
        C=D*S*D' 
        C=P'*S*P 
        C=P*S*P' 
        GrB_extract C=S(p,p) instead of C=P*S*P':
        C=M'*S*M 

    Next, the data type is printed: (d) double, (s) single, (z) double complex,
    and (c) single complex.

    Then the run times are compared as the max(MKL,GrB)/min(MKL,GrB) ratio,
    for both the 1st run and "2nd" (subsequent) runs.

        If MKL is 2x or faster:  "SLOWDOWN"
        If MKL is 1.2x to 2x or faster:  "slowdown"
        If the max time / min time is 1.2 or less: "="
        If GrB is 2x or faster:  "SPEEDUP"
        If GrB is 1.2x to 2x or faster:  "speedup"

    The relative run time ratio is printed one of in 3 ways.  If GrB is faster,
    the ratio is printed to the left, and if MKL is faster, the ratio is
    on the right.  For example:

         8:C+=S|*B (| 2)(d) MKL:     0.0384 GrB:     0.0686 slowdown:          1.8
         8:C+=S|*B (| 4)(d) MKL:     0.0425 GrB:     0.0419                  =
         8:C+=S|*B (| 5)(d) MKL:     0.0800 GrB:     0.0594 speedup:     1.3

    8 threads were used, and S is CSC.  For C and B with 2 columns (in column
    layout), MKL was faster (0.0384 seconds) and GrB/MKL time is 1.8.
    For n = 4, the times are about the same ("=").
    For n = 5, GrB is a bit faster (MKL/GrB time is 1.3).

    Both the warmup time ("1st") and subsequent times ("2nd") are compared,
    since MKL does extra work in the first iteration, which is used to
    speed up its 2nd runs.  GrB doesn't do this.

    To go back and parse and summarize the detailed output file, use the awk
    scripts:

        res.awk         all results summarized
        res8.awk        just results with 8 and 40 threads
        fast.awk        just where SPEEDUP is printed (GrB 2x or faster than MKL)
        slow.awk        just where SLOWDOWN is printed (MKL 2x or faster than GrB)
        mm16.awk        just results for C+=S*B where C and B have 16 columns
        mm4.awk         just results for C+=S*B where C and B have 4 columns
        mv.awk          just results for mkl_sparse_*_mv
        mm.awk          just results for mkl_sparse_*_mm
        sp2m.awk        just results for mkl_sparse_sp2m

    For example:

        ./dobench > output.txt
        awk -f res.awk < output.txt

========
Progress
========

My current raw results are in the Keep/ folder, and summarized in the
Summary/ folder.

mkl_sparse_*_mv, mkl_sparse_*_mm, mkl_sparse_sp2m, and mkl_sparse_add
and are benchmarked for all 4 data types.  mkl_sparse_sypr is only
benchmarked for double.

In one case, mkl_sparse_d_mm doesn't support the method directly: when the
sparse matrix is CSC and C, B are in column layout.  (I'm using 0-based
indexing only).  So to implement this, I import the sparse matrix A in CSR
format (no explicit transpose, no work at all, with nrows and ncols swapped),
and I then use the op = SPARSE_OPERATION_TRANSPOSE.  MKL does fine in terms of
its performance, relative to its other uses.  This is for the method
"C+=S|*F (| n)", as printed.

My results are on these platforms:

    Mac: a MacBook Pro (13-inch, 2020) with Intel Core i7-1068NG7.
        (4 hardware cores, 8 threads, IceLake, AVX2 and AVX512F)
        MacOS 12.1 Monterey.
        memory: 32 GB 3733 MHz LPDDR4X in two 16GB banks.
        with MKL version: 2021.0.4, sizeof(MKL_INT): 8.
        Variants:
            (1) GraphBLAS compiled with gcc-11.2.0 (HomeBrew), mklgrb linked
                against mkl_intel_thread and libiomp.
            (2) GraphBLAS compiled with icc 2021.5.0, mklgrb linked
                against mkl_intel_thread and libiomp.
        I can compile a program gcc-11.2.0 and link with libomp from HomeBrew
        (which uses LLVM) but then there is no mkl_gnu_thread available on the
        Mac, so I can't use this option for mklgrb with the Intel MKL sparse
        BLAS.  Also, icx is not available on the Mac.

    DGX Station: hypersparse.engr.tamu.edu (Intel Xeon CPU E5-2698 v4 @ 2.20GHz)
        (20 hardware cores, 40 threads, with AVX2 but not AVX512F).
        memory: 256GB
        Ubuntu 20.04.  with MKL version 2020 but I'm unable to link against it
        (not in /opt/intel, but installed in /usr/lib with an apt package,
        and setvars.sh is not available).
        Hopefully will be fixed when MKL 2022 is installed in /opt/intel.
        Variants:
            (1) GraphBLAS compiled with gcc-11.2.0 and linked against libgomp,
                and mklgrb linked with mkl_gnu_thread, and libgomp.
            (2) TODO GraphBLAS compiled with icx and linked against libiomp,
                and mklgrb linked with mkl_intel_thread, and libiomp.

    Dell Linux laptop: Intel Core i7-8565U @ 1.8GHz
        (4 hardware cores, 8 threds, with AVX2 but not AVX512F)
        memory: 16GB.  Ubuntu 18.04. with MKL 2022
        Variants:
            (1) GraphBLAS compiled with gcc-11.2.0 and linked against libgomp,
                and mklgrb linked with mkl_gnu_thread, and libgomp.
            (2) GraphBLAS compiled with gcc-11.2.0 and linked against libiomp,
                and mklgrb linked with mkl_intel_thread, and libiomp.
                Unable to make this work; GraphBLAS still brings in libgomp,
                not libiomp5, likely from the cmake for libgraphblas.so.
            (3) GraphBLAS compiled with icx 2022 and linked against libiomp,
                and mklgrb linked with mkl_intel_thread, and libiomp.
            (4) GraphBLAS compiled with icc 2021.5.0 and linked against libiomp,
                and mklgrb linked with mkl_intel_thread, and libiomp (TODO)

