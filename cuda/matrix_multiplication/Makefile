.PHONY: build clean sweep

build:
	nvcc -o matmul matmul.cu

clean:
	rm matmul metrics.csv

sweep: build
	./param_sweep.sh
