#/bin/bash

for i in {2..10}
do
	#echo "Number: $((2**i))"
	
	gcc -O1 -fopenmp test_mmm_inter_omp.c -lrt -o test_mmm_inter_omp -DDELTA="$((2**i))"
	OMP_NUM_THREADS=4 ./test_mmm_inter_omp
	
done
