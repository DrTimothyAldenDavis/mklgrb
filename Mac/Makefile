
# on the Mac
JOBS ?= 8

INC =  \
   -I../LAGraph/include -I../GraphBLAS/Include \
   -I../GraphBLAS/Source -I../GraphBLAS/cpu_features/include \
   -I../GraphBLAS/Source/Template -I../GraphBLAS/rmm_wrap

# gcc-11 with libgomp: fails (no mkl_gnu_thread on the Mac)

# icc with libiomp
# Intel MKL is fine but GraphBLAS is a bit slower than with gcc
CC = icc
CFLAGS = -qopenmp -DMKL_ILP64 -m64 -I"${MKLROOT}/include" -std=c11 $(INC)
LDFLAGS = \
    -L/usr/local/lib -Wl,-rpath,/usr/local/lib \
    -lgraphblas -llagraph -llagraphx \
    -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
    -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 \
    -lpthread -lm -ldl

# gcc-11 with libiomp (GraphBLAS is faster with gcc-11 than with icc)
# This is the fastest for GraphBLAS, and does fine for MKL too
IOMP = /opt/intel/oneapi/compiler/2022.0.0/mac/compiler/lib
CC = gcc-11
CFLAGS = -fopenmp -DMKL_ILP64 -m64  -I"${MKLROOT}/include" -std=c11 \
    -I/usr/local/include $(INC)
LDFLAGS = \
    -L/usr/local/lib -Wl,-rpath,/usr/local/lib \
    -lgraphblas -llagraph -llagraphx \
    -L${IOMP} -Wl,-rpath,${IOMP} \
    -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
    -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 \
    -lpthread -lm -ldl

default:
	$(CC) $(CFLAGS) mklgrb.c -c -o mklgrb.o
	$(CC) $(CFLAGS) mklgrb_util.c -c -o mklgrb_util.o
	$(CC) $(CFLAGS) benchmark_mv.c -c -o benchmark_mv.o
	$(CC) $(CFLAGS) benchmark_mm.c -c -o benchmark_mm.o
	$(CC) $(CFLAGS) benchmark_sp2m.c -c -o benchmark_sp2m.o
	$(CC) $(CFLAGS) benchmark_spmmd.c -c -o benchmark_spmmd.o
	$(CC) $(CFLAGS) benchmark_add.c  -c -o benchmark_add.o
	$(CC) $(CFLAGS) benchmark_sypr.c -c -o benchmark_sypr.o
	$(CC) $(CFLAGS) benchmark_transpose.c -c -o benchmark_transpose.o
	$(CC) $(CFLAGS) benchmark_syrk.c -c -o benchmark_syrk.o
	$(CC) $(CFLAGS) mklgrb_sypr.c -c -o mklgrb_sypr.o
	$(CC) $(CFLAGS) mklgrb_transpose.c -c -o mklgrb_transpose.o
	$(CC) $(CFLAGS) mklgrb_sp2m.c -c -o mklgrb_sp2m.o
	$(CC) $(CFLAGS) mklgrb_square.c -c -o mklgrb_square.o
	$(CC) $(CFLAGS) mklgrb_add.c  -c -o mklgrb_add.o
	$(CC) $(CFLAGS) mklgrb_spmmd.c  -c -o mklgrb_spmmd.o
	$(CC) mklgrb.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb
	$(CC) mklgrb_sypr.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_sypr
	$(CC) mklgrb_transpose.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_transpose
	$(CC) mklgrb_sp2m.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_sp2m
	$(CC) mklgrb_square.o mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_square
	$(CC) mklgrb_add.o  mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_add
	$(CC) mklgrb_spmmd.o  mklgrb_util.o bench*.o $(LDFLAGS) -o mklgrb_spmmd
	$(CC) $(CFLAGS) mtxconvert.c -c -o mtxconvert.o
	$(CC) mtxconvert.o $(LDFLAGS) -o mtxconvert

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
	rm -rf *.o mklgrb mtxconvert mklgrb_sypr mklgrb_add mklgrb_sp2m mklgrb_square mklgrb_transpose mklgrb_spmmd

