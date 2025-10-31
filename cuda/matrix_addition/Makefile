.PHONY: build sweep clean

build:
	nvcc -o matadd matrix_addition.cu

sweep: build
	./param_sweep.sh

clean:
	rm ./matadd metrics.csv
