import pdb
from linguistic_dict import Linguistic_DICT
ld = Linguistic_DICT()
from pypinyin import pinyin,Style, style
import glob
import re
import os
import logging
import shutil
import numpy as np
import pickle
logging.basicConfig(level=logging.INFO)
class Samesen:
    def __init__(self):
        self.wav_dir="/home/gyzhang/corpus/allwav"
        self.lab_dir = "/home/gyzhang/corpus/all_labs"
        self.ipa_dict = ld.get_lexicon_dict(lexicon_path='../data/dicts/ipa_m.dict')
        self.utt2spk = '../exp/sameSen300/kaldi/utt2spk'
        self.no_tone_kaldi_text = '../exp/sameSen300/kaldi/text'
        self.wav_scp = '../exp/sameSen300/kaldi/wav.scp'
        self.mon_wav = "/home/gyzhang/speech_database/sameSen300/mon_wav"
        self.lexicon_char = "../data/dicts/lexicon_char"

    def normalizef0(self,):
        relative_tone_ratio_dict = dict()
        for spk_name in ['lc','wz','zq','zz']:
            spk_lab = os.path.join(self.mon_wav, spk_name)
            relative_tone = np.zeros((5, 5, 20))
            relative_tone_f0_mean = np.zeros((5, 5, 2))
            relative_tone_ratio = np.zeros((5, 5, 1))
            tone_sum = np.zeros((5, 10))
            tone_count = np.zeros((5, 1))
            for wav_path in glob.glob(spk_lab + '/*.wav'):
            # we apply tone normalization for first speaker firstly
                wav_basename = os.path.basename(wav_path)
                e_utt_id = re.split('\.', wav_basename)[0]
                logging.info("process utterance id {0}".format(e_utt_id))
                tone_lab_path = os.path.join(spk_lab, e_utt_id+'.tonelab')
                timenormf0_path = os.path.join(spk_lab, e_utt_id+'.actutimenormf0')
                f0_mean_path = os.path.join(spk_lab,e_utt_id+'.means')
                assert os.path.exists(tone_lab_path) and os.path.exists(timenormf0_path) and os.path.exists(f0_mean_path)
                with open(tone_lab_path,'r') as tone_lab_fid:
                    tone_lab_lines = tone_lab_fid.readlines()
                with open(timenormf0_path,'r') as norm_f0_fid:
                    norm_f0_lines = norm_f0_fid.readlines()
                with open(f0_mean_path,'r') as mean_f0_fid:
                    mean_f0_lines = mean_f0_fid.readlines()
                tone_lab_line = tone_lab_lines[0]
                syl_list = re.split('\s+',tone_lab_line.strip())
                # syl_f0 dict list
                syl_f0_list = []

                # create (syl, mean f0) map
                syl_mean_f0_list = []
                for num, syl_mean_f0 in enumerate(mean_f0_lines):
                    if num == 0:
                        continue
                    line_list = re.split('\s+', syl_mean_f0.strip())
                    mean_f0_value = line_list[4]
                    syl = line_list[0]
                    if mean_f0_value != "--undefined--":
                        mean_f0_value_float = float(mean_f0_value)
                    else:
                        print("A undefine mean f0 value was found")
                        mean_f0_value_float = 0
                    syl_mean_f0 = (syl, mean_f0_value_float)
                    syl_mean_f0_list.append(syl_mean_f0)
                for num, norm_f0 in enumerate(norm_f0_lines):
                    # first line is not our stuffs
                    # because we use ten points, we can use a list to store. num(1, 10) (11, 20)
                    if num == 0:
                        continue
                    if num%10 == 1:
                        f0_value_list = []
                        syl_f0 = tuple()
                    syl_notone, _, f0_value = re.split('\s+', norm_f0.strip())
                    f0_value_list.append(f0_value)
                    if num%10 == 0:
                        # syl_f0[syl_notone] = f0_value_list
                        syl_f0 = (syl_notone, f0_value_list)
                        syl_f0_list.append(syl_f0)
                syl_tone_list = []
                for syl in syl_list:
                    assert syl[-1].isdigit()
                    tone = syl[-1]
                    syl_notone = syl[0:-1]
                    # syl_tone_map[syl_notone] = tone
                    syl_tone_map = (syl_notone, tone)
                    syl_tone_list.append(syl_tone_map)
                assert len(syl_f0_list) == len(syl_tone_list)
                # R_matrix = np.zeros((4,4))
                max_len = len(syl_f0_list)
                for num,(syl_mean_f0, syl_tone_map) in enumerate(zip(syl_mean_f0_list, syl_tone_list)):
                    if num == max_len - 1:
                        continue
                # determine whether the same string the initial
                    assert syl_mean_f0[0] == syl_tone_map[0]
                    f0_mean = syl_mean_f0[1]
                    next_f0_mean = syl_mean_f0_list[num+1][1]
                    tone=syl_tone_map[1]
                    next_tone = syl_tone_list[num + 1][1]
                    # bi_f0_list = np.concatenate((f0_list_float, next_f0_list_float), axis=0)
                    # relative_tone[int(tone)-1,int(next_tone)-1,:] += bi_f0_list
                    relative_tone_f0_mean[int(tone)-1, int(next_tone)-1,0] += f0_mean
                    relative_tone_f0_mean[int(tone)-1, int(next_tone)-1,1] += next_f0_mean
            for index_1 in range(5):
                for index_2 in range(5):
                    bi_f0 = relative_tone_f0_mean[index_1, index_2, :]
                    pre_f0 = bi_f0[0]
                    next_f0 = bi_f0[1]
                    relative_tone_ratio[index_1, index_2, 0] = pre_f0/next_f0
            relative_tone_ratio_dict[spk_name] = relative_tone_ratio
        f = open("relative_tone_ratio.dict","wb")
        pickle.dump(relative_tone_ratio_dict, f)
        f.close()
            # np.save("./relative_tone_ratio-"+spk_name+".npy", relative_tone_ratio)
    #             for num,(syl_f0, syl_tone_map) in enumerate(zip(syl_f0_list, syl_tone_list)):
    #                 if num == max_len-1:
    #                     continue
    #                 # determine whether the same string the initial
    #                 assert syl_f0[0] == syl_tone_map[0]
    #                 f0_list = syl_f0[1]
    #                 next_f0_list = syl_f0_list[num+1][1]
    #                 tone=syl_tone_map[1]
    #                 next_tone = syl_tone_list[num + 1][1]
    #                 # tone_count[int(tone)-1] = tone_count[int(tone)-1] + 1
    #                 f0_list_float = np.array([float(ele) for ele in f0_list])
    #                 next_f0_list_float = np.array([float(ele) for ele in next_f0_list])
    #                 # tone_sum[int(tone)-1] = tone_sum[int(tone)-1] + f0_list_float
    #                 bi_f0_list = np.concatenate((f0_list_float, next_f0_list_float), axis=0)
    #                 relative_tone[int(tone)-1,int(next_tone)-1,:] += bi_f0_list
    # for index_1 in range(5):
    #     for index_2 in range(5):
    #         bi_f0 = relative_tone[index_1, index_2, :]
    #         pre_f0 = bi_f0[3:8]
    #         next_f0 = bi_f0[13:28]
    #         sum_pre_f0 = np.sum(pre_f0)
    #         sum_next_f0 = np.sum(next_f0)
    #         relative_tone_ratio[index_1, index_2, 0] = sum_pre_f0/sum_next_f0
    # print(relative_tone_ratio)
    # np.save("./tone_sum.npy", tone_sum)
    # np.save("./tone_count.npy",tone_count)

    def double_check_lab(self,):
        """
        make sure tone lab and lab are same
        :return:
        """
        count=0
        for spk in ['wz']:
            spk_lab = os.path.join(self.mon_wav, spk)
            for wav_path in glob.glob(spk_lab+'/'+ spk +'-Angry*.wav'):
                wav_basename = os.path.basename(wav_path)
                utt_id = re.split('\.', wav_basename)[0]
                spk_name, emotion, index =re.split('-',utt_id)
                for emotion in ['Angry','Fear','Happy','Neutral','Sad','Surprise']:
                    for spk_name in ['lc','wz','zq','zz']:
                        spk_mon_wav = os.path.join(self.mon_wav, spk_name)
                        e_utt_id = spk_name + '-' + emotion +'-'+ index
                        tone_lab_path = os.path.join(spk_mon_wav, e_utt_id+'.tonelab')
                        lab_path = os.path.join(spk_mon_wav, e_utt_id + '.lab')
                        # logging.info("process file {0}".format(utt_id))
                        if os.path.exists(lab_path) and os.path.exists(wav_path) and os.path.exists(tone_lab_path):
                            fid = open(lab_path,'r')
                            lab_txt = fid.readlines()[0]
                            fid.close()
                            tone_fid = open(tone_lab_path,'r')
                            tone_lab_txt = tone_fid.readlines()[0]
                            tone_fid.close()
                            lab_txt_list = re.split('\s+',lab_txt.strip())
                            tone_lab_list = re.split('\s+',tone_lab_txt.strip())
                            if len(lab_txt_list) != len(tone_lab_list):
                                print(tone_lab_path)
                            for lab, tone_label in zip(lab_txt_list, tone_lab_list):
                                if not lab == tone_label[0:-1]:
                                    count=count+1
                                    print(tone_lab_path)
                                    # pdb.set_trace()
        print(count)

    def check_labs(self,):
        '''
            Because there are some errors of this corpus, so I write this program
            to check transcriptions for alignment task
        '''
        for spk in ['wz']:
            spk_lab = os.path.join(self.mon_wav, spk)
            for wav_path in glob.glob(spk_lab+'/'+ spk +'-Angry*.wav'):
                wav_basename = os.path.basename(wav_path)
                utt_id = re.split('\.', wav_basename)[0]
                spk_name, emotion, index =re.split('-',utt_id)
                emotion_lab_list = []
                emotion_path_list = []
                for emotion in ['Angry','Fear','Happy','Neutral','Sad','Surprise']:
                    for spk_name in ['lc','wz','zq','zz']:
                        spk_mon_wav = os.path.join(self.mon_wav, spk_name)
                        e_utt_id = spk_name + '-' + emotion +'-'+ index
                        lab_path = os.path.join(spk_mon_wav, e_utt_id+'.tonelab')
                        wav_path = os.path.join(spk_mon_wav,e_utt_id+'.wav')
                        if os.path.exists(lab_path) and os.path.exists(wav_path):
                            fid = open(lab_path,'r')
                            lab_txt = fid.readlines()[0]
                            fid.close()
                            lab_txt_list = re.split('\s+',lab_txt.strip())
                            emotion_lab_list.append(lab_txt_list)
                            emotion_path_list.append(lab_path)
                        else:
                            print(wav_path)
                            emotion_lab_list.append("")
                            emotion_path_list.append(lab_path)
                        # pdb.set_trace()
                len_array=np.array([len(emotion_lab) for emotion_lab in emotion_lab_list])
                if not np.max(len_array) == np.min(len_array):
                    for lab_path,lab_txt in zip(emotion_path_list,emotion_lab_list):
                        # print(lab_path)
                        # print(lab_txt)
                    # pdb.set_trace()
                        max_ele_index = np.argmax(len_array)
                    for lab_path,lab_txt in zip(emotion_path_list,emotion_lab_list):
                        if len(lab_txt) != len_array[max_ele_index]:
                            with open(lab_path,'w') as fid:
                                fid.write(' '.join(emotion_lab_list[max_ele_index])+'\n')
                                # print(lab_path)

    def create_lexicon_syl(self, ):
        """
            create lexicon with syllabel level and replace text and mon with syllabel level
        """
        lexicon_syl=[]
        lcid = open(self.lexicon_char,'w')
        logger = logging.getLogger(__name__)
        for lab_file in glob.glob(self.lab_dir+ '/*.lab'):
            lab_basename = os.path.basename(lab_file)
            utt_id = re.split('\.', lab_basename)[0]
            spk_name =re.split('-',utt_id)[0]
            mon_lab_path = os.path.join(self.mon_wav, spk_name, utt_id+'.lab')
            with open(lab_file,'r') as fid:
                lab_lines = fid.readlines()
            syl_list = []
            for lab_line in lab_lines:
                try:
                    _, _ , syl = re.split('\s+', lab_line.strip())
                except ValueError:
                    pdb.set_trace()
                if syl == "SIL":
                    continue
                # pdb.set_trace()
                assert syl[-1].isdigit
                syl = syl[0:-1]
                if syl not in lexicon_syl:
                    lexicon_syl.append(syl)
                syl_list.append(syl)
            with open(mon_lab_path, 'w') as mfid:
                mfid.write(' '.join(syl_list)+'\n')
            logger.info("process lab file:{0}".format(lab_file))
        for syl in lexicon_syl:
            syl_phone_list = []
            initial = style.convert(syl, Style.INITIALS, True)
            final = style.convert(syl, Style.FINALS, True)
            if initial != "":
                ipa_initial = self.ipa_dict[initial]
                syl_phone_list.extend(ipa_initial)
            ipa_final = self.ipa_dict[final]
            syl_phone_list.extend(ipa_final)
            lcid.write("{0} {1}\n".format(syl, ' '.join(syl_phone_list)))
        lcid.close()


    def process_corpus(self, write_kaldi=False, write_lab=False,write_tone=True):
        """
        Target:
        1. there are some errors in labs which we need to fix and remove the original silence 
        2. remove to a mon wav but keep tone 
        Args:
            write_kaldi: whether write kaldi file
            write_lab: whether write lab
            write_tone: whther write syllabus with tone
        """
        logger = logging.getLogger(__name__)
        for lab_file in glob.glob(self.lab_dir+ '/*.lab'):
            lab_basename = os.path.basename(lab_file)
            utt_id = re.split('\.', lab_basename)[0]
            spk_name =re.split('-',utt_id)[0]
            wav_path=os.path.join(self.wav_dir,utt_id+'.wav')
            mon_wav_path = os.path.join(self.mon_wav, spk_name, utt_id+'.wav')
            mon_lab_path = os.path.join(self.mon_wav, spk_name, utt_id+'.lab')
            mon_lab_tone_path = os.path.join(self.mon_wav, spk_name, utt_id+'.tone1_lab')
            with open(lab_file,'r') as fid:
                lab_lines = fid.readlines()
            phone_list = []
            syllable_list = []
            for lab_line in lab_lines:
                try:
                    _, _ , syl = re.split('\s+', lab_line.strip())
                except ValueError:
                    pdb.set_trace()
                if syl == "SIL":
                    continue
                syllable_list.append(syl)
                # pdb.set_trace()
                assert syl[-1].isdigit
                syl = syl[0:-1]
                initial = style.convert(syl, Style.INITIALS, True)
                final = style.convert(syl, Style.FINALS, True)
                if initial != "":
                    ipa_initial = self.ipa_dict[initial]
                    phone_list.extend(ipa_initial)
                ipa_final = self.ipa_dict[final]
                phone_list.extend(ipa_final)

            # if write kaldi file
            if write_kaldi:
                with open(self.utt2spk,'w') as utt2spk_fid:
                    utt2spk_fid.write('{0} {1}\n'.format(utt_id,spk_name))
                with open(self.wav_scp,'w') as wav_scp_fid:
                    wav_scp_fid.write('{0} {1}\n'.format(utt_id, wav_path))
            logger.info("process lab file:{0}".format(lab_file))
            if write_lab:
                shutil.copy(wav_path,mon_wav_path)
                with open(mon_lab_path, 'w') as mfid:
                    mfid.write(' '.join(phone_list))
            if write_tone:
                with open(mon_lab_tone_path, 'w') as rfid:
                    rfid.write(' '.join(syllable_list)+'\n')
           
if __name__ == '__main__':
    ss = Samesen()
    # ss.process_corpus()
    # ss.create_lexicon_syl()
    # ss.check_labs()
    ss.normalizef0()
    # ss.double_check_lab()