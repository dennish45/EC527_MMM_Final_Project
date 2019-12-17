#/bin/bash

# use this command to compile GPU programs (sub in GPU program name for the one you want, and change ARR_SIZE and NUM_BLOCKS to whatever):
# nvcc mmm_shared.cu -o mmm_shared.out -arch compute_35 -code sm_35 -DARR_SIZE=4160 -DNUM_BLOCKS=130 -Xptxas -O3

prgList=(
	#"mmm_ijk"
	#"mmm_ijk_omp"
	#"mmm_bbkij"
	#"mmm_bkij"
	#"mmm_kij"
	#"mmm_vect"
	#"mmm_kij_omp"
	"mmm_vect_omp"
)

for prgName in "${prgList[@]}"
do

	prg="cpu-opt/$prgName"
	echo -e "\n$prg"
	echo -e "optimization level, 128+64, 256+64, 512+64, 1024+64, 2048+64, 4096+64"
	for opt in {0..3}
	do
		
		# print optimization level
		echo -n "O$opt, "
		
		for i in {7..12}
		do
			
			# compile and run
			gcc -O"$((opt))" -lm -fopenmp -mavx2 -mfma -march=native -funroll-loops $prg.c -lrt -o $prg.out -DARRSIZE="$((2**i + 64))"
			printval=`OMP_NUM_THREADS=16 ./$prg.out`
			echo -en "$printval"
			
			# if last elem, no comma
			if [ $i != 11 ]
			then
				echo -n ", "
			fi
			
		done
		
		echo
	done
done
	
