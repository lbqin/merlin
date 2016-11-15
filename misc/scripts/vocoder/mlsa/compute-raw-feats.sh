#!/bin/sh
# Generates marytts lf0 from a list of wav file.

srate=16000
job=1
. ./parse_options.sh

outdir=$2

for i in `awk -v lst="$1" 'BEGIN{if (lst ~ /^scp/) sub("[^:]+:[[:space:]]*","", lst); while (getline < lst) print $1 "___" $2}'`; do
    name=${i%%___*}
    wfilename=${i##*___}
    featname=`basename $wfilename .wav`.raw
    sox $wfilename $outdir/$featname
done
