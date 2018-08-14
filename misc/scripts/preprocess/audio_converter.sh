#!/usr/bin/env bash

# top merlin directory
merlin_dir="/home/research/data/speech/merlin"

# tools directory
tools="${merlin_dir}/tools/bin"
sptk="${tools}/SPTK-3.9"

voice_dir="/home/research/data1/tts/tuozi_48k"

# input cmp directory
wav_dir="${voice_dir}/wav0"

# Output synthesis wav directory
out_dir="${voice_dir}/wav"

# initializations
powernormalise=1
maxamplitude=1
removenoise=1
trimsilence=1
tarsamplerate=48000

### audio converter
$tools/audioConverter ${wav_dir} ${out_dir} ${powernormalise} ${maxamplitude} ${removenoise} ${trimsilence} ${tarsamplerate}

