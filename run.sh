#!/bin/bash

#Required installation of screen: sudo apt-get install screen

if [ "$1" != "" ]; then
    echo "Running main1.py $1 times..."
else
    echo "Syntax: $0 <no of times>"
	exit
fi

for (( i=1; i <= $1; ++i ))
do
  screen  -dm -S forgit bash -c 'python3 main1.py; exec bash'
done
