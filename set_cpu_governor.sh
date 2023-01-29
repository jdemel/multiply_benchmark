#!/bin/bash

## Script requires "cpufrequtils" on Debianoids (e.g. Ubuntu)

next="$1"

if [ -z "$1" ]
then
    next="performance"
fi

echo "Set governor to '$next'"

ncpus=$(nproc)
echo "Found ${ncpus} CPUs"

for ((i=0; i<$ncpus; i++))
do
    IFS=' ' read -r -a data <<< "$(cpufreq-info -c $i -p)"
    curr="${data[2]}"

    echo "CPU$i set governor to '$next' (was '$curr')"
    cpufreq-set -c $i -r -g $next
done

echo "CPU governor options: $(cpufreq-info -g)"
