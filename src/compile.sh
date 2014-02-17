#! /bin/sh

arg=$1
name=${arg%.*}.out
g++ -O2 $1 -std=c++11 -o $name

exit

