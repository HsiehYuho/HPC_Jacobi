for P in 1 2 4 16 64; do
	for N in 16; do
 		qsub -v p=$P,n=$N ./pbs_script.pbs
	done
done
