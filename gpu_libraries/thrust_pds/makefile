# module load PrgEng-cray
# module load craype-accel-nvidia35

all : sort_host sort_device

.SUFFIXES:.cu

% : %.cu
	nvcc $< -o $@

clean :
	rm -f sort_host sort_device
