#!/bin/bash
cd $1;
for a in * ;
	do echo "$a";
	cd "$a";
	for i in *.au;
		do sox "$i" "${i%.au}.wav";
		rm "$i";
	done;
	cd ..;
done;
