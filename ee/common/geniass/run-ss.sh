#1: input folder, 2: output folder
#! /bin/sh
for entry in "$1"/*; do
	filename=${entry##*/}
	filename="$2/"$filename
    ./geniass $entry $filename
done