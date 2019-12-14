#/bin/bash

prgList=(
	"mmm_ijk"
	"mmm_ijk_omp"
	"mmm_bbkij"
	"mmm_bkij"
	"mmm_kij"
	"mmm_vect"
	"mmm_kij_omp"
	"mmm_vect_omp"
)

for prgName in "${prgList[@]}"
do

	prg="cpu-opt/$prgName"
	echo -e "\n$prg"
	echo -e "optimization level, 64, 128, 256, 512, 1024, 2048"
	for opt in {0..3}
	do
		
		# print optimization level
		echo -n "O$opt, "
		
		for i in {6..11}
		do
			
			# compile and run
			gcc -O"$((opt))" -lm -fopenmp -mavx2 -mfma -march=native -funroll-loops $prg.c -lrt -o $prg.out -DARRSIZE="$((2**i))"
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
	
