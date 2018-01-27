#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running main1.py $1 times..."
else
    echo "Syntax: $0 <no of times>"
	exit
fi

for run in {1..$1}
do
  screen -S "Forgit" python3 main1.py
done