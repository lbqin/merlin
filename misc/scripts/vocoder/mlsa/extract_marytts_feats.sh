corpus_dir=/home/sooda/data/tts/labixx120_48k/
audio_dir=$corpus_dir/wav 
cppmary_base=/home/sooda/speech/cppmary/
cppmary_bin=$cppmary_base/build/
sample_rate=48000
FRAMESHIFT=0.005
nj=8
featdir=/tmp/acoustic
mgcdir=$featdir/mgc
strdir=$featdir/str
lf0dir=$featdir/lf0
rawdir=$featdir/raw

rm -rf $mgcdir $strdir $lf0dir $rawdir
mkdir -p $mgcdir $strdir $lf0dir $rawdir

makeid="xargs -i basename {} .wav"

find $audio_dir -name "*.wav" | sort | $makeid | awk -v audiodir=$audio_dir '{line=$1" "audiodir"/"$1".wav"; print line}' > $corpus_dir/wav.scp

#if [ "$sample_rate" == "16000" ]; then
#    make_pitch.sh --pitch-config conf/pitch.conf data/$step exp/make_pitch/$step $featdir || exit 1
#elif [ "$sample_rate" == "44100" ]; then
#    make_pitch.sh --pitch-config conf/pitch-44k.conf data/$step exp/make_pitch/$step $featdir || exit 1
#elif [ "$sample_rate" == "48000" ]; then
#    make_pitch.sh --pitch-config conf/pitch-48k.conf data/$step exp/make_pitch/$step $featdir || exit 1
#fi

#make_str.sh --sample-frequency $sample_rate . exp/make_str/$step $featdir || exit 1

#./make_mgc.sh --sample-frequency $sample_rate $corpus_dir /tmp $mgcdir || exit 1
#./make_parallel.sh --sample-frequency $sample_rate $corpus_dir /tmp $rawdir "./compute-raw-feats.sh" || exit 1
#./make_parallel.sh --sample-frequency $sample_rate $corpus_dir /tmp $lf0dir "./compute-lf0-feats.sh" || exit 1
#./make_parallel.sh --sample-frequency $sample_rate $corpus_dir /tmp $mgcdir "./compute-mgc-feats.sh" || exit 1
./make_parallel.sh --sample-frequency $sample_rate --nj $nj $corpus_dir /tmp $strdir "./compute-str-feats.sh" || exit 1

