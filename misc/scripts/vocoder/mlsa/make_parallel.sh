#!/bin/bash 

# Copyright 2016  Huanshi LTD (Author: Liushouda)
# Apache 2.0

# Begin configuration section.
nj=8
cmd=./run.pl
compress=false
sample_frequency=16000
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. ./parse_options.sh || exit 1;

data=$1
logdir=$2
outdir=$3
compute_cmd=$4


# make $outdir an absolute pathname.
outdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $outdir ${PWD}`


# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $outdir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/wav.scp

required="$scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_mgc.sh: no such file $f"
    exit 1;
  fi
done

split_scps=""
for n in $(seq $nj); do
  split_scps="$split_scps $logdir/wav_${name}.$n.scp"
done

./split_scp.pl $scp $split_scps || exit 1;

in_feats="$logdir/wav_${name}.JOB.scp"

$cmd JOB=1:$nj $logdir/make_$name.JOB.log $compute_cmd --srate $sample_frequency --job JOB $in_feats $outdir || exit 1;

if [ -f $logdir/.error.$name ]; then
  echo "Error producing mgc features for $name:"
  tail $logdir/make_$name.*.log
  exit 1;
fi
