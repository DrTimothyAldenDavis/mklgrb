# on slash (Dell laptop)

JOBS ?= 8

INC = \
    -I../LAGraph/include -I../GraphBLAS/Include \
    -I../GraphBLAS/Source -I../GraphBLAS/cpu_features/include \
    -I../GraphBLAS/Source/Template -I../GraphBLAS/rmm_wrap

# trying to use gcc and libiomp (but getting libomp and libgomp...!)
CC = gcc
CFLAGS = -fopenmp -DMKL_ILP64 -m64 -I/usr/include/mkl $(INC)
LDFLAGS = \
   -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 \
   -L../LAGraph/build/lib   -Wl,-rpath,../LAGraph/build/lib \
   -L../GraphBLAS/build -Wl,-rpath,../GraphBLAS/build \
   -l:libgraphblas.so.6 -llagraph -llagraphx \
   -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# clang and libomp
CC = clang
CFLAGS = -fopenmp -DMKL_ILP64 -I${MKLROOT}/include $(INC)
LDFLAGS = -L../LAGraph/build/lib   -Wl,-rpath,../LAGraph/build/lib \
    -L../GraphBLAS/build -Wl,-rpath,../GraphBLAS/build \
    -l:libgraphblas.so.6 -llagraph -llagraphx \
    -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_gnu_thread \
    -lmkl_core -liomp5 -lpthread -lm -ldl

# gcc and gomp
CC = gcc
CFLAGS = -g -fopenmp -DMKL_ILP64 -m64  -I/usr/include/mkl $(INC)
LDFLAGS = \
   -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
   -L../LAGraph/build/lib   -Wl,-rpath,../LAGraph/build/lib \
   -L../GraphBLAS/build -Wl,-rpath,../GraphBLAS/build \
   -l:libgraphblas.so.6 -llagraph -llagraphx \
   -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread \
   -lmkl_core -lgomp -lpthread -lm -ldl

# icx and libiomp
CC = icx
CFLAGS = -fiopenmp -DMKL_ILP64 -I${MKLROOT}/include $(INC) 
LDFLAGS = -L../LAGraph/build/lib   -Wl,-rpath,../LAGraph/build/lib \
    -L../GraphBLAS/build -Wl,-rpath,../GraphBLAS/build \
    -l:libgraphblas.so.6 -llagraph -llagraphx \
    -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread \
    -lmkl_core -liomp5 -lpthread -lm -ldl

default:
	$(CC) $(CFLAGS) mklgrb.c -c -o mklgrb.o
	$(CC) $(CFLAGS) mklgrb_util.c -c -o mklgrb_util.o
	$(CC) $(CFLAGS) benchmark_iter.c -c -o benchmark_iter.o
	$(CC) $(CFLAGS) benchmark_mv.c -c -o benchmark_mv.o
	$(CC) $(CFLAGS) benchmark_mm.c -c -o benchmark_mm.o
	$(CC) $(CFLAGS) benchmark_sp2m.c -c -o benchmark_sp2m.o
	$(CC) $(CFLAGS) benchmark_spmmd.c -c -o benchmark_spmmd.o
	$(CC) $(CFLAGS) benchmark_add.c  -c -o benchmark_add.o
	$(CC) $(CFLAGS) benchmark_sypr.c -c -o benchmark_sypr.o
	$(CC) $(CFLAGS) benchmark_transpose.c -c -o benchmark_transpose.o
	$(CC) $(CFLAGS) benchmark_syrk.c -c -o benchmark_syrk.o
	$(CC) $(CFLAGS) test_iter.c -c -o test_iter.o
	$(CC) $(CFLAGS) mklgrb_sypr.c -c -o mklgrb_sypr.o
	$(CC) $(CFLAGS) mklgrb_transpose.c -c -o mklgrb_transpose.o
	$(CC) $(CFLAGS) mklgrb_sp2m.c -c -o mklgrb_sp2m.o
	$(CC) $(CFLAGS) mklgrb_square.c -c -o mklgrb_square.o
	$(CC) $(CFLAGS) mklgrb_add.c  -c -o mklgrb_add.o
	$(CC) $(CFLAGS) mklgrb_spmmd.c  -c -o mklgrb_spmmd.o
	$(CC) mklgrb.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb
	$(CC) test_iter.o mklgrb_util.o bench*.o $(LDFLAGS) -o test_iter
	$(CC) mklgrb_sypr.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_sypr
	$(CC) mklgrb_transpose.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_transpose
	$(CC) mklgrb_sp2m.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_sp2m
	$(CC) mklgrb_square.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_square
	$(CC) mklgrb_add.o  mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_add
	$(CC) mklgrb_spmmd.o  mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_spmmd
	$(CC) $(CFLAGS) mtxconvert.c -c -o mtxconvert.o
	$(CC) mtxconvert.o $(LDFLAGS) -o mtxconvert

triplet:
	$(CC) $(CFLAGS) mtx_to_triplet.c -c -o mtx_to_triplet.o
	$(CC) mtx_to_triplet.o $(LDFLAGS) -o mtx_to_triplet
	$(CC) $(CFLAGS) lagraph_to_triplet.c -c -o lagraph_to_triplet.o
	$(CC) lagraph_to_triplet.o mklgrb_util.o $(LDFLAGS) -o lagraph_to_triplet

# on the Mac:  report libraries used, and flags for cc and ld
macset:
	otool -L ./mklgrb
	echo "mklroot: " ${MKLROOT}
	echo "cflags:  " ${CFLAGS}
	echo "ldflags: " ${LDFLAGS}

# on the Linux:  report libraries used, and flags for cc and ld
linset:
	ldd ./mklgrb
	echo "mklroot: " ${MKLROOT}
	echo "cflags:  " ${CFLAGS}
	echo "ldflags: " ${LDFLAGS}

# (broken) build with cmake
broken:
	( cd build ; cmake $(CMAKE_OPTIONS) .. ; $(MAKE) --jobs=$(JOBS) )
	otool -L ./build/mklgrb
	otool -L ./build/mtxconvert

# (broken) build with cmake and run 1 small matrix
also_broken: default
	./build/mklgrb ../MM/ND/nd3k/nd3k.lagraph

clean: distclean

purge: distclean

# remove all files not in the distribution
distclean:
	rm -rf *.o mklgrb mtxconvert mklgrb_sypr mklgrb_add mklgrb_sp2m
	rm -rf mklgrb_square mklgrb_transpose mklgrb_spmmd test_iter
	rm -rf mtx_to_triplet lagraph_to_triplet

