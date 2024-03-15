#valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all ./main
#cuda-gdb
#compute-sanitizer --tool memcheck
rm *.o
rm tmpt/*
clear
printf "[\033[93m***\033[0m] \033[103mCompilation ...\033[0m \n"

# g c
# G cuda
#echo "!!! -G -g !!!";A="-Idef -diag-suppress 2464 -G -g -O0 -lm -lcublas_static -lcublasLt_static -lculibos -Xcompiler -fopenmp -Xcompiler -O0"
#echo "!!! -g !!!";A="-Idef -diag-suppress 2464 -g -O0 -lm -lcublas_static -lcublasLt_static -lculibos -Xcompiler -fopenmp -Xcompiler -O3"
A="-Idef -diag-suppress 2464 -O3 -lm -lcublas_static -lcublasLt_static -lculibos -Xcompiler -fopenmp -Xcompiler -O3"

# les 3 lignes au dessus : debbug cuda, debbug, optimiser  

#	/etc
nvcc -c impl/etc/etc.cu     ${A} &
nvcc -c	impl/etc/marchee.cu ${A} &
nvcc -c	impl/etc/nature0__directe.cu    ${A} &
nvcc -c	impl/etc/nature1__macd.cu       ${A} &
nvcc -c	impl/etc/nature2__chiffre.cu    ${A} &
nvcc -c	impl/etc/nature3__awesome.cu    ${A} &
nvcc -c	impl/etc/nature4__pourcent_r.cu ${A} &
nvcc -c	impl/etc/nature5__rsi.cu        ${A} &
nvcc -c	impl/etc/outils_natures.cu ${A} &
nvcc -c	impl/etc/exemple_outil.cu  ${A} &
#
#	/insts
nvcc -c impl/insts/dot1d.cu                   ${A} &
nvcc -c impl/insts/dot1d/dot1d_naive.cu       ${A} &
nvcc -c impl/insts/dot1d/dot1d_shared.cu      ${A} &
nvcc -c impl/insts/dot1d/dot1d_shared_2_16.cu ${A} &
#
nvcc -c impl/insts/filtres.cu                ${A} &
nvcc -c impl/insts/filtres/filtres_naive.cu  ${A} &
nvcc -c impl/insts/filtres/filtres_shared.cu ${A} &
#
nvcc -c impl/insts/filtres_tolerant.cu                         ${A} &
nvcc -c impl/insts/filtre_tolerant/filtres_tolerant_naive.cu  ${A} &
nvcc -c impl/insts/filtre_tolerant/filtres_tolerant_shared.cu ${A} &
#
nvcc -c impl/insts/dot1d_blk.cu                       ${A} &
nvcc -c impl/insts/dot1d_blk/dot1d_blk_naive.cu       ${A} &
nvcc -c impl/insts/dot1d_blk/dot1d_blk_shared_2_16.cu ${A} &
#
nvcc -c impl/insts/dot1d_mul.cu                       ${A} &
nvcc -c impl/insts/dot1d_mul/dot1d_mul_shared_2_16.cu ${A} &
#
#	/scores
nvcc -c impl/scores/cuda_S.cu    ${A} &
nvcc -c impl/scores/cpu_S.cu     ${A} &
nvcc -c impl/scores/cuda_pred.cu ${A} &
nvcc -c impl/scores/cpu_pred.cu  ${A} &
#
#	/mdl
nvcc -c impl/mdl/mdl.cu             ${A} &
nvcc -c impl/mdl/mdl_io.cu          ${A} &
nvcc -c impl/mdl/mdl_f.cu           ${A} &
nvcc -c impl/mdl/mdl_df.cu          ${A} &
nvcc -c impl/mdl/mdl_plume.cu       ${A} &
nvcc -c impl/mdl/mdl_utilisation.cu ${A} &
nvcc -c impl/mdl/mdl_calc_alpha.cu  ${A} &
nvcc -c impl/mdl/mdl_perturber.cu   ${A} &
#
#	/opti
nvcc -c impl/opti/opti_simple.cu  ${A} &
nvcc -c impl/opti/opti_rmsprop.cu ${A} &
nvcc -c impl/opti/opti_adam.cu    ${A} &
#
nvcc -c impl/opti/opti_opti.cu         ${A} &
nvcc -c impl/opti/opti_masque.cu       ${A} &
nvcc -c impl/opti/opti_mini_paquets.cu ${A} &
#
#	/main
nvcc -c impl/main/verif_mdl.cu          ${A} &
nvcc -c impl/main/structure_generale.cu ${A} &
#
#	Attente de terminaison des differents fils de compilation
#
wait

#	Compilation du programme principale
nvcc -c impl/main.cu ${A}
nvcc *.o -o main ${A}; rm main.o;
#	Compilation prog3
nvcc -c impl/prog3__plume_filtre.cu ${A}
nvcc *.o -o prog3__plume_filtre ${A}; rm prog3__plume_filtre.o
#	Compilation prog4
nvcc -c impl/prog4__simple_mdl_pour_python.cu ${A}
nvcc *.o -o prog4__simple_mdl_pour_python ${A}; rm prog4__simple_mdl_pour_python.o

#	Verification d'erreure
if [ $? -eq 1 ]
then
	printf "\n[\033[91m***\033[0m] \033[101mErreure. Pas d'execution.\033[0m\n"
	rm *.o
	exit
fi
rm *.o

#	Executer
printf "[\033[92m***\033[0m] \033[102m========= Execution du programme =========\033[0m\n"

#valgrind --leak-check=yes --track-origins=yes ./prog
time ./main
if [ $? -ne 0 ]
then
	printf "[\033[91m***\033[0m] \033[101mErreur durant l'execution.\033[0m\n"
	#valgrind --leak-check=yes --track-origins=yes ./prog
	#sudo systemd-run --scope -p MemoryMax=100M gdb ./prog
	exit
else
	printf "[\033[92m***\033[0m] \033[102mAucune erreure durant l'execution.\033[0m\n"
fi