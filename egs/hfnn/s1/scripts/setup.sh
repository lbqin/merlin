#!/bin/bash

if test "$#" -ne 1; then
    echo "Usage: ./scripts/setup.sh <voice_directory_name>"
    exit 1
fi

current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $current_working_dir)))
experiments_dir=${current_working_dir}/experiments

voice_name=$1
voice_dir=${experiments_dir}/${voice_name}

acoustic_dir=${voice_dir}/acoustic_model
duration_dir=${voice_dir}/duration_model
synthesis_dir=${voice_dir}/test_synthesis

mkdir -p ${experiments_dir}
mkdir -p ${voice_dir}
mkdir -p ${acoustic_dir}
mkdir -p ${duration_dir}
mkdir -p ${synthesis_dir}

if [ "$voice_name" == "hfnn" ]
then
    data_dir=hfnn_data
else
    echo "The data for voice name ($voice_name) is not available...please use hfnn!!"
    exit 1
fi

if [[ ! -d ${acoustic_dir}/data ]]; then
    # extract the feature
    cp -r ${data_dir}/acoustic_data ${acoustic_dir}/data
    cp -r ${data_dir}/data/* ${acoustic_dir}/data
    cp -r ${data_dir}/data ${duration_dir}/data
    cp -r ${data_dir}/test_data/* ${synthesis_dir}/
    # prepare the alignment file
fi
echo "data is ready!"

global_config_file=conf/global_settings.cfg

### default settings ###
echo "MerlinDir=${merlin_dir}" >  $global_config_file
echo "WorkDir=${current_working_dir}" >>  $global_config_file
echo "Voice=${voice_name}" >> $global_config_file
echo "Labels=state_align" >> $global_config_file
echo "Vocoder=WORLD" >> $global_config_file
echo "SamplingFreq=48000" >> $global_config_file

if [ "$voice_name" == "hfnn" ]
then
    echo "FileIDList=basename.scp" >> $global_config_file
    echo "Train=1800" >> $global_config_file 
    echo "Valid=230" >> $global_config_file 
    echo "Test=15" >> $global_config_file 
else
    echo "The data for voice name ($voice_name) is not available...please use hfnn !!"
    exit 1
fi

echo "Merlin default voice settings configured in $global_config_file"
echo "setup done...!"

