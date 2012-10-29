#!/bin/sh

for i in $(seq --format="%02g" 01 10)
do
	for j in $(seq --format="%02g" 01 30)
	do
		./TabuSolver -i 5000 -d 2 -c 2 -m 8 -t 100 -f "problema_${i}.txt" -O "out/${i}_${j}.txt"
	done
done
