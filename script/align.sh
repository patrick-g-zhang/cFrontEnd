#!/usr/bin/env bash
# create mfcc feature
# silence will put at the end of phone dict and their number should be expressed
export LC_ALL=C
num_iters=10    # Number of iterations of training
max_iter_inc=30 # Last iter to increase #Gauss on.
totgauss=2500 # Target #Gaussians.
data=/home/gyzhang/kingtts/data
num_nonsil_states=3
num_sil_states=5
sil_num=4
careful=false
num_iters=10
power=0.25 # exponent to determine number of gaussians from occurrence counts
ndisambig=1
scp=$data/wav.scp
text=$data/text
utt2spk=$data/utt2spk
# notice the shift and windows there we will talk this later
compute-mfcc-feats --verbose=2 --config=$data/conf/mfcc.conf scp,p:$scp ark:- | copy-feats ark:- ark,scp:$data/mfcc/raw_mfcc.ark,$data/feats.scp 
# notice --spk2utt add train.spk2utt
utt2spk_to_spk2utt.pl utt2spk >spk2utt
compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp ark,scp:$data/mfcc/cmvn.ark,$data/cmvn.scp

#train_mono
#calc 39-dim reduce cmvn
apply-cmvn --utt2spk=ark:$utt2spk scp:$data/cmvn.scp scp:$data/feats.scp ark:- | add-deltas ark:- ark:$data/mfcc/cmvn_39.ark

feats="ark,s,cs:$data/mfcc/cmvn_39.ark"

( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) >$data/phones/disambig.txt


#create topo file
paste $data/phones/phones.txt $data/phones/phones.txt > $data/phones/lexicon.txt
perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $data/phones/lexicon.txt > $data/phones/lexiconp.txt
grep -v -F -f $data/phones/silence_phones.txt $data/phones/phones.txt > $data/phones/nonsilence_phones.txt 

cat $data/phones/silence_phones.txt $data/phones/nonsilence_phones.txt | awk '{for(n=1;n<=NF;n++) print $n; }' > $data/phones/phones

paste -d' ' $data/phones/phones $data/phones/phones > $data/phones/phone_map.txt

 echo "<eps>" | cat - $data/phones/{silence,nonsilence,disambig}.txt | awk '{n=NR-1; print $1, n;}' > $data/phones/phones.txt


for f in silence nonsilence disambig; do
	/home/gyzhang/kaldi/egs/timit/s5/utils/sym2int.pl $data/phones/phones.txt <$data/phones/$f.txt >$data/phones/$f.int
 	/home/gyzhang/kaldi/egs/timit/s5/utils/sym2int.pl $data/phones/phones.txt <$data/phones/$f.txt | awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// > $data/phones/$f.csl
done

cat $data/phones/lexiconp.txt | awk '{print $1}' | sort | uniq  | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' > $data/phones/words.txt


silphonelist=`cat $data/phones/silence.csl`
nonsilphonelist=`cat $data/phones/nonsilence.csl`

/home/gyzhang/kaldi/egs/timit/s5/utils/gen_topo.pl $num_nonsil_states $num_sil_states $nonsilphonelist $silphonelist >$data/topo
cat phones/phones.txt | cut -d' ' -f2 > $data/phones/sets.int
gmm-init-mono topo 39 $data/0.mdl $data/tree

# generate fst file

/home/gyzhang/kaldi/egs/timit/s5/utils/make_lexicon_fst.pl --pron-probs $data/phones/lexiconp.txt | fstcompile --isymbols=$data/phones/phones.txt --osymbols=$data/phones/words.txt --keep_isymbols=false --keep_osymbols=false | fstrmepsilon | fstarcsort > L.fst

numgauss=`gmm-info --print-args=false $data/0.mdl | grep gaussians | awk '{print $NF}'`
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss

mkdir -p $data/exp/mono
sym2int.pl -f 2- $data/phones/words.txt <$data/text >text.tra
compile-train-graphs --read-disambig-syms=$data/phones/disambig.int $data/tree $data/0.mdl $data/L.fst ark:$data/text.tra "ark:|gzip -c >$data/exp/mono/fsts.gz" 

align-equal-compiled "ark:gunzip -c $data/exp/mono/fsts.gz|" "$feats" ark,t:-  | gmm-acc-stats-ali --binary=true $data/0.mdl "$feats" ark:- $data/0.acc
gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power $data/0.mdl $data/0.acc $data/1.mdl
beam=10
x=1
while [ $x -lt $num_iters ]; do
     gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*4] --careful=$careful $x.mdl "ark:gunzip -c $data/exp/mono/fsts.gz|" "$feats" "ark,t:|gzip -c >$data/ali.gz" 
      gmm-acc-stats-ali  $data/$x.mdl "$feats" "ark:gunzip -c $data/ali.gz|" $data/$x.acc
      gmm-est --write-occs=$data/$[$x+1].occs --mix-up=$numgauss --power=$power $data/$x.mdl $data/$x.acc $data/$[$x+1].mdl
  x=$[$x+1]
done
