#!/bin/bash

name=tree_test;

main(){
	for N in 5000 10000 20000 40000
	do
	for T in 0.2 0.4 0.8
	do
	./tree -n $N -t $T -m >> ${name}.txt;		# without Quadrupole
   	./tree -n $N -t $T -m -q >>${name}.txt; 	# with Quadrupole
   	echo "N = $N, T = $T";
	done 
	done
	echo "Test Complete."
}

main

