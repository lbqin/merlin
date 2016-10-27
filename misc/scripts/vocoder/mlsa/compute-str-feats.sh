#!/bin/sh
# Generates marytts str from a list of wav file.

export KALDI_ROOT=/home/sooda/speech/kaldi
export tooldir=$KALDI_ROOT/tools/SPTK/bin

help_message="Usage: ./compute-str-feats.sh [options] scp:<in.scp> <wspecifier>\n\tcf. top of file for list of options."

STRFILTERNAME=mix_excitation_5filters_199taps_48Kz.txt
tmpdir=/tmp

AWK=gawk
PERL=/usr/bin/perl
BC=/usr/bin/bc
TCLSH=/usr/bin/tclsh
WC=/usr/bin/wc

# SPTK commands
X2X=$tooldir/x2x
FRAME=$tooldir/frame
WINDOW=$tooldir/window
MGCEP=$tooldir/mcep
LPC2LSP=$tooldir/lpc2lsp
STEP=$tooldir/step
MERGE=$tooldir/merge
VSTAT=$tooldir/vstat
NRAND=$tooldir/nrand
SOPR=$tooldir/sopr
VOPR=$tooldir/vopr
NAN=$tooldir/nan
MINMAX=$tooldir/minmax

SAMPFREQ=44100   # Sampling frequency (48kHz)
FRAMELEN=1103   # Frame length in point (1200 = 48000 * 0.025)
FRAMESHIFT=221 # Frame shift in point (240 = 48000 * 0.005)
WINDOWTYPE=1 # Window type -> 0: Blackman 1: Hamming 2: Hanning
NORMALIZE=1  # Normalization -> 0: none  1: by power  2: by magnitude
FFTLEN=2048     # FFT length in point
FREQWARP=0.53   # frequency warping factor
GAMMA=0      # pole/zero weight for mel-generalized cepstral (MGC) analysis
MGCORDER=34   # order of MGC analysis
STRORDER=5     # order of STR analysis, number of filter banks for mixed excitation
MAGORDER=10    # order of Fourier magnitudes for pulse excitation generation
LNGAIN=1     # use logarithmic gain rather than linear gain
LOWERF0=80    # lower limit for f0 extraction (Hz)
UPPERF0=420    # upper limit for f0 extraction (Hz)
NOISEMASK=50  # standard deviation of white noise to mask noises in f0 extraction
MLEN=$(($MGCORDER+1))
job=1
srate=16000

#echo "$0 $@"  # Print the command line for logging

. ./parse_options.sh

outdir=$2
SAMPFREQ=$srate
fshift=5
FRAMESHIFT=$(( $srate * $fshift / 1000 ))

if [ "$srate" == "16000" ]; then
  FREQWARP=0.42
  FFTLEN=512
  FRAMELEN=400
  STRFILTERNAME=mix_excitation_5filters_99taps_16Kz.txt
elif [ "$srate" == "44100" ]; then
  FREQWARP=0.53
  FFTLEN=2048
  FRAMELEN=1103
  STRFILTERNAME=mix_excitation_5filters_199taps_48Kz.txt
elif [ "$srate" == "48000" ]; then
  FREQWARP=0.55
  FFTLEN=2048
  FRAMELEN=1200
  STRFILTERNAME=mix_excitation_5filters_199taps_48Kz.txt
fi

for i in `awk -v lst="$1" 'BEGIN{if (lst ~ /^scp/) sub("[^:]+:[[:space:]]*","", lst); while (getline < lst) print $1 "___" $2}'`; do
    name=${i%%___*}
    wfilename=${i##*___}
    featname=`basename $wfilename .wav`.str
    raw=$tmpdir/str$job.raw
    sox $wfilename $raw
    count=`echo "0.005 * $SAMPFREQ" | $BC -l`;
    $STEP -l `printf "%.0f" $count` | $X2X +fs > $tmpdir/str$job.head;
    count=`echo "0.025 * $SAMPFREQ" | $BC -l`;
    $STEP -l `printf "%.0f" $count` | $X2X +fs > $tmpdir/str$job.tail;
    cat $tmpdir/str$job.head $raw $tmpdir/str$job.tail > $tmpdir/str$job;
    $TCLSH get_str.tcl -l -H $UPPERF0 -L $LOWERF0 -p $FRAMESHIFT -r $SAMPFREQ -f $STRFILTERNAME -n $STRORDER $tmpdir/str$job > $outdir/$featname
done
