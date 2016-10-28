corpus_dir=/home/sooda/data/tts/xll_48k/
audio_dir=$corpus_dir/wav 
cppmary_base=/home/sooda/speech/cppmary/
cppmary_bin=$cppmary_base/build/
sample_rate=48000
FRAMESHIFT=0.005
nj=8
featdir=/tmp/acoustic_data
mgcdir=$featdir/mgc
strdir=$featdir/bap
lf0dir=$featdir/lf0

rm -rf $mgcdir $strdir $lf0dir
mkdir -p $mgcdir $strdir $lf0dir

makeid="xargs -i basename {} .wav"

find $audio_dir -name "*.wav" | sort | $makeid | awk -v audiodir=$audio_dir '{line=$1" "audiodir"/"$1".wav"; print line}' > $corpus_dir/wav.scp

./make_parallel.sh --sample-frequency $sample_rate --nj $nj $corpus_dir /tmp $lf0dir "./compute-lf0-feats.sh" || exit 1
./make_parallel.sh --sample-frequency $sample_rate --nj $nj $corpus_dir /tmp $strdir "./compute-str-feats.sh" || exit 1
./make_parallel.sh --sample-frequency $sample_rate --nj $nj $corpus_dir /tmp $mgcdir "./compute-mgc-feats.sh" || exit 1

