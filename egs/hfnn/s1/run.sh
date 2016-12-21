#!/bin/bash

if test "$#" -ne 0; then
    echo "Usage: ./run.sh"
    exit 1
fi

### Step 1: setup directories and the training data files ###
echo "Step 1: setting up experiments directory and the training data files..."
voice_name=hfnn
current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $current_working_dir)))
experiments_dir=${current_working_dir}/experiments
voice_dir=${experiments_dir}/${voice_name}
acoustic_dir=${voice_dir}/acoustic_model
duration_dir=${voice_dir}/duration_model
synthesis_dir=${voice_dir}/test_synthesis

mkdir -p ${experiments_dir}
mkdir -p ${voice_dir}
mkdir -p ${acoustic_dir}
mkdir -p ${duration_dir}
mkdir -p ${synthesis_dir}

data_dir=${current_working_dir}/${voice_name}_data

if [[ ! -d ${acoustic_dir}/data ]]; then
    ln -s ${data_dir}/data ${acoustic_dir}/data
    ln -s ${data_dir}/data ${duration_dir}/data
    cp -r ${data_dir}/test_data/* ${synthesis_dir}/
fi
echo "data is ready!"

global_config_file=conf/global_settings.cfg
./scripts/prepare_config_files.sh $global_config_file 0
./scripts/prepare_config_files.sh $global_config_file 1

if [ ! -f  $global_config_file ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $global_config_file
fi

### Step 2: train duration model ###
echo "Step 2: training duration model..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin_cppmary.py conf/duration_${Voice}.conf

### Step 3: train acoustic model ###
echo "Step 3: training acoustic model..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin_cppmary.py conf/acoustic_${Voice}.conf


### Step 4: synthesize speech   ###
echo "Step 4: synthesizing speech..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin_cppmary.py conf/test_dur_synth_${Voice}.conf
./scripts/submit.sh ${MerlinDir}/src/run_merlin_cppmary.py conf/test_synth_${Voice}.conf

### Step 5: delete intermediate synth files ###
echo "Step 5: deleting intermediate synthesis files..."
./scripts/remove_intermediate_files.sh conf/global_settings.cfg

echo "synthesized audio files are in: experiments/${Voice}/test_synthesis/wav"
echo "All successfull!! Your demo voice is ready :)"

