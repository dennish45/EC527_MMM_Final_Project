#/bin/bash

prgList=(
	"mmm_ijk"
	"mmm_ijk_omp"
	"mmm_bbkij"
	"mmm_bkij"
	"mmm_kij"
	"mmm_kij_omp"
)

for prg in "${prgList[@]}"
do
	echo -e "\n$prg"
	echo -e "optimization level, 8, 16, 32, 64, 128, 256, 512, 1024"
	for opt in {0..3}
	do
		
		# print optimization level
		echo -n "O$opt, "
		
		for i in {3..10}
		do
			
			# compile and run
			gcc -O"$((opt))" -mavx -fopenmp $prg.c -lrt -o $prg.out -DARRSIZE="$((2**i))"
			printval=`OMP_NUM_THREADS=16 ./$prg.out`
			echo -en "$printval"
			
			# if last elem, no comma
			if [ $i != 10 ]
			then
				echo -n ", "
			fi
			
		done
		
		echo
	done
done
	
