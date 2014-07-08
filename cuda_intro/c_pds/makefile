#CXX= cc
CXX=nvcc
#CXXFLAGS=-O3
#CXXFLAGS=-O3 -h acc
CXXFLAGS=  -O3 -arch=sm_35 -Xcompiler -fopenmp #-openmp is just to use their timers!

SOURCES = stats.c data.c operators.c linalg.c
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o

.SUFFIXES: .c

all: main

stats.o: stats.c stats.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c stats.c -o $@

data.o: data.c data.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c data.c -o $@

operators.o: operators.c operators.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c operators.c -o $@

linalg.o: linalg.c linalg.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c $<

main: $(OBJ) main.c $(HEADERS)
	$(CXX) $(CXXFLAGS) *.o main.c -o $@

clean:
	rm -f main
	rm -f *.o
	rm -f *.i

