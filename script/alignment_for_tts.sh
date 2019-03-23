#!/bin/bash
cd /home/gyzhang/kaldi/egs
proj_name=nameclean4
front_name=namestory
char_level=true 
copy_to_front=true
KALDI_ROOT=/home/gyzhang/kaldi
if [[ -d $proj_name ]]; then
	rm -rf $proj_name
fi
mkdir $proj_name;
cd $proj_name
ln -s $KALDI_ROOT/egs/wsj/s5/steps .
ln -s $KALDI_ROOT/egs/wsj/s5/utils .
ln -s $KALDI_ROOT/src . 

cp $KALDI_ROOT/egs/wsj/s5/path.sh .

# change the path.sh kaldi root manually 

mkdir exp
mkdir conf 
mkdir data

mkdir data/train
mkdir data/lang
mkdir data/local

mkdir data/local/lang

# create files for data/train manually
# move text to data/train
train=data/train
lang=data/lang
cp /home/gyzhang/projects/cFrontEnd/exp/$front_name/kaldi/no_tone_kaldi_text $train/text
cp /home/gyzhang/projects/cFrontEnd/exp/$front_name/kaldi/wav.scp $train/wav.scp
cp /home/gyzhang/projects/cFrontEnd/exp/$front_name/kaldi/utt2spk $train/utt2spk

cut -d ' ' -f 2- $train/text | sed 's/ /\n/g' | sort -u > $train/words.txt
cp $train/words.txt $lang/words.txt 

# phone model create lexicon directly
# if the lexicon are char level it should use char level lexicon
utils/fix_data_dir.sh $train
if [ "$char_level"=true ]; then
	cp /home/gyzhang/projects/cFrontEnd/data/dicts/lexicon.txt data/local/lang/lexicon.txt;
	awk '{print $2 "\n" $3 "\n" $4}' data/local/lang/lexicon.txt| sed 's/ /\n/g' | sort -u > data/local/lang/nonsilence_phones.txt;
	sed -i '/^\s*$/d' data/local/lang/nonsilence_phones.txt
	echo -e 'sil'\\n'<oov>' > data/local/lang/silence_phones.txt;
	echo 'sil' > data/local/lang/optional_silence.txt;
else
	cat $lang/words.txt | awk '{print $1 "\t" $1}' > data/local/lang/lexicon.txt;
	awk '{print $1}' data/local/lang/lexicon.txt| sed 's/ /\n/g' | sort -u > data/local/lang/nonsilence_phones.txt;
	echo -e 'sil'\\n'<oov>' > data/local/lang/silence_phones.txt;
	echo 'sil' > data/local/lang/optional_silence.txt;
fi
# at the top of lexicon, add <oov> <oov>

sed -i '1i <oov> \t <oov>' data/local/lang/lexicon.txt

# move files wav.scp and utt2spk manually


sed -i s#'export KALDI_ROOT.*'#'export KALDI_ROOT='${KALDI_ROOT}# path.sh
utils/prepare_lang.sh data/local/lang '<oov>' data/local/ data/lang

# parallelization setting
echo -e "train_cmd=\"run.pl\"\ndecode_cmd=\"run.pl --mem 2G\"" > cmd.sh

. ./cmd.sh

# create mfcc.conf
echo -e "--use-energy=false\n--sample-frequency=16000\n--frame-shift=5" > conf/mfcc.conf
#extract mfcc features
mfccdir=mfcc
steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 $train exp/make_mfcc/$train $mfccdir
steps/compute_cmvn_stats.sh $train exp/make_mfcc/train $mfccdir

# monophone training and alignment
steps/train_mono.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" data/train data/lang exp/mono
steps/align_si.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali 

# steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 data/train data/lang exp/mono_ali exp/tri1
#Align delta-based triphones
# steps/align_si.sh --nj 1 --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_ali
#train delta delta -delta triphones
# steps/train_deltas.sh --cmd "$train_cmd" 2500 15000 data/train data/lang exp/tri1_ali exp/tri2a

# steps/align_si.sh --nj 1 --cmd "$train_cmd" --use-graphs true data/train data/lang exp/tri2a exp/tri2a_ali 

# obtain ctm output
. ./path.sh
ali-to-phones --ctm-output --frame-shift=0.005 exp/mono_ali/final.mdl ark:"gunzip -c exp/mono_ali/ali.1.gz|" -> phone_ctm.ctm
ali-to-pdf exp/mono/final.mdl ark:"gunzip -c exp/mono_ali/ali.1.gz|" ark,t:pdf.ali
feat-to-len scp:data/train/feats.scp ark,t:feats.lengths
if [ "$copy_to_front"=true ]; then
	cp phone_ctm.ctm /home/gyzhang/projects/cFrontEnd/exp/$front_name/kaldi/
	cp pdf.ali /home/gyzhang/projects/cFrontEnd/exp/$front_name/kaldi/
	cp feats.lengths /home/gyzhang/projects/cFrontEnd/exp/$front_name/kaldi/
	cp data/lang/phones.txt /home/gyzhang/projects/cFrontEnd/exp/$front_name/kaldi/
fi