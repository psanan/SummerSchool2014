# you need to specify the correct arch if you're going to specify it explicitly! This is for the hardware on todi
ARCH = 35
NVCC = nvcc
CXXFLAGS = -mtune=native -ffast-math -fopenmp
NVFLAGS = -Icuda_launch_config -g -O3 -rdc=true -lineinfo -arch=sm_$(ARCH) -use_fast_math $(addprefix -Xcompiler=,$(CXXFLAGS)) 
#NVFLAGS+=-Xptxas=-v 

HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   main.o

.SUFFIXES: .cu

all: main

stats.o: stats.cu stats.h
	$(NVCC) $(NVFLAGS) $(IFLAGS) -c $< -o $@

data.o: data.cu data.h
	$(NVCC) $(NVFLAGS) $(IFLAGS) -c $< -o $@

main.o: main.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) $(IFLAGS) -c $< -o $@

main: $(OBJ)
	$(NVCC) $(NVFLAGS) $^ -o $@

clean:
	rm -f main *.o *.i output.bov output.bin

run : main
	aprun ./main 128 128 100 0.01

test: main
	aprun ./main 128 128 100 0.01 > result && diff -u result result.ref

