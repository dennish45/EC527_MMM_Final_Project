#/bin/bash

# use this command to compile GPU programs (sub in GPU program name for the one you want, and change ARR_SIZE and NUM_BLOCKS to whatever):
# nvcc mmm_shared.cu -o mmm_shared.out -arch compute_35 -code sm_35 -lcublas -DARR_SIZE=4160 -DNUM_BLOCKS=130 -Xptxas -O3
# This is written for the ECE-HPC-02 machine ONLY. Run on other machines at your own risk.

module load cuda

prgList=(
	"mmm_vect_omp"
	#"mmm_kij_omp"
	#"mmm_vect"
	#"mmm_kij"
	#"mmm_bkij"
	#"mmm_bbkij"
	#"mmm_ijk_omp"
	#"mmm_ijk"
)

for prgName in "${prgList[@]}"
do

	prg="cpu-opt/$prgName"
	echo -e "\n$prg"
	echo -e "optimization level, 128+64, 256+64, 512+64, 1024+64, 2048+64, 4096+64, 8192+64, 16384+64"
	#for opt in {0..3}
	#do
		
		# print optimization level
		echo -n "O3, "
		
		for i in {7..14}
		do
			
			# compile and run
			gcc -O3 -lm -fopenmp -mavx2 -mfma -march=native -funroll-loops $prg.c -lrt -o $prg.out -DARRSIZE="$((2**i + 64))"
			printval=`OMP_NUM_THREADS=20 ./$prg.out`
			echo -en "$printval"
			
			# if last elem, no comma
			if [ $i != 14 ]
			then
				echo -n ", "
			fi
			
		done
		
		echo
	#done
done
	
