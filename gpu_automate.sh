#/bin/bash

# use this command to compile GPU programs (sub in GPU program name for the one you want, and change ARR_SIZE and NUM_BLOCKS to whatever):
# nvcc mmm_shared.cu -o mmm_shared.out -arch compute_35 -code sm_35 -lcublas -DARR_SIZE=4160 -DNUM_BLOCKS=130 -Xptxas -O3

prgList=(
	#"mmm_ijk"
	#"mmm_ijk_omp"
	#"mmm_bbkij"
	#"mmm_bkij"
	#"mmm_kij"
	#"mmm_vect"
	#"mmm_kij_omp"
	#"mmm_global"
	#"mmm_shared"
	"mmm_cublas"
)

for prgName in "${prgList[@]}"
do

	prg="gpu-opt/$prgName"
	echo -e "\n$prg"
	echo -e "array size, time"
	#    						 192,    320    , 576  ,  1088,   2112‬,  4160

	# 192 = 6,8,12,16,24,32,48,64,96,192
	# 320 = 10, 16, 20, 32, 40, 64, 80, 160, 320

	# 576 = 18, 24, 32, 36, 48, 64, 72, 96, 144, 192, 288, 576.

	# 1088 = 32 // 34,64,68,136,272,544,1088

	# 4160 = 130,160,208,260,320,416,520,832,1040,2080,4160

	# 2112 = 66, 88, 96,132, 176, 192, 264, 352, 528, 704, 1056, and 2112

	#for i in {7..12}
	for i in {7..17}
	do
		
		# print optimization level
		echo -n "$((2**i + 64)), "
		
		#for block_power in {0..5}
		#do
			
			array_size="$((2**i + 64))"
			#num_blocks="$((array_size/(2**block_power)))"

			# compile and run
			nvcc $prg.cu -o $prg.out -arch compute_60 -code sm_60 -lcublas -DARR_SIZE=$array_size -Xptxas -O3
			#gcc -O"$((opt))" -lm -fopenmp -mavx2 -mfma -march=native -funroll-loops $prg.c -lrt -o $prg.out -DARRSIZE="$((2**i + 64))"
			printval=`OMP_NUM_THREADS=16 ./$prg.out`
			echo -en "$printval"
			
			# if last elem, no comma
			if [ $i != 17 ]
			then
				echo -n ", "
			fi
		#done
		
		echo
	done
done
	
