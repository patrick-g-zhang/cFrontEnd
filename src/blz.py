from pypinyin import pinyin,Style, style
from linguistic_dict import Linguistic_DICT
ld = Linguistic_DICT()
import logging
import urllib, sys
import ssl
import time
import re
import os
import glob
from utils import text2list
from aip import AipNlp
from shutil import copyfile
from FrontEnd import FrontEnd
from praatio import tgio
from collections import OrderedDict
import pdb
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import coloredlogs
coloredlogs.install(level='DEBUG')
logger = logging.getLogger(__name__)

class blzFrontEnd(FrontEnd):

    def __init__(self):
        ## path of input chinese character file
        self.text_file = None
        APP_ID = '15716974'
        API_KEY = 'QZ4ee5tvyLrKCZ5FZib1eDFN'
        SECRET_KEY = 'Zhe7VieQlGeSvGoPbeHdfLLeDF78KOYO'
        self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
        self.text_file_path = "/home/gyzhang/speech_database/text/blz19/text.txt"
        # where is alignment file
        self.alignment_file_dir = "/home/gyzhang/Documents/aligned_blz_multi_2/"
        self.label_phone_align = "/home/gyzhang/projects/cFrontEnd/exp/blz/train/label_phone_align"
        self.merlin_files="../exp/blz/train"

    def pre_process(self, raw_text):
        index = raw_text[0:7]
        text_line = raw_text.strip()[7:]
        return index, text_line


    def align_phone_map(self, ali_phone_list):
        """
            because alignment file contains 'sil' and 'sp' two symbols which we must need to remove
            we return a file which contains a list contains non-silence index
            [1,2,3,5]
        :param ali_phone_list:
        :return:
        """
        non_silence_index_list = []
        for num, phone in enumerate(ali_phone_list):
            if phone != "sil" and phone != "sp":
                non_silence_index_list.append(num)
        return non_silence_index_list

    def gen_file_id_list(self,):
        """
            generate file id list
            because we know some files cannot be decoded, so the size of fild id list is determined by our alignment files
        :return:
        """
        logger = logging.getLogger("generate file id")
        logger.info("generate file id")
        fild_id_list = os.path.join(self.merlin_files, "file_id_list.scp")
        fid = open(fild_id_list,'w')
        for file_path in glob.glob(self.alignment_file_dir+'*.TextGrid'):
            file_index = re.split('\.',os.path.basename(file_path))[0]
            fid.write(file_index+'\n')

    def create_phone_labels(self, ):
        with open(self.text_file_path, 'r') as fid:
            textlines = fid.readlines()
        for num, text_line in enumerate(textlines):
            sent_index, sent_content = self.pre_process(text_line)
            # corresponding alignment file
            sent_index = sent_index.strip()
            # test code
            #if sent_index != "100168":
           #      continue
           # pdb.set_trace()
           # if alignment file does not exist
            alignment_file_path = os.path.join(self.alignment_file_dir,sent_index+'.TextGrid')
            if not os.path.exists(alignment_file_path):
                logging.warning("alignment file {} does not exist".format(alignment_file_path))
                continue
            # corresponding label file
            label_phone_path = os.path.join(self.label_phone_align, sent_index + '.lab')
            label_phone_fid = open(label_phone_path, 'w')
            ## extract word and pos tagging

            # extract alignment file for whole paragraph
            para_sil_phone_list = []
            para_start_time_list = []
            para_end_time_list = []
            tg = tgio.openTextgrid(alignment_file_path)
            wordTier = tg.tierDict['phones']
            for start, stop, label in wordTier.entryList:
                start_time = str(int(start * 10000000))
                end_time = str(int(stop * 10000000))
                phone = label
                para_sil_phone_list.append(phone)
                para_start_time_list.append(start_time)
                para_end_time_list.append(end_time)
            # replace sp to sil 
            replace_sil_phone_list = []
            for my_sil_phone in para_sil_phone_list:
                if my_sil_phone == "sp":
                    replace_sil_phone_list.append("sil")
                else:
                    replace_sil_phone_list.append(my_sil_phone)
            para_sil_phone_list = replace_sil_phone_list
           # pdb.set_trace()
            non_silence_index_list = self.align_phone_map(para_sil_phone_list)
            sil_nonsil_map = OrderedDict()
            if num%5==4:
                time.sleep(1)

            new_para_sil_phone_list = []
            for phone in para_sil_phone_list:
                if phone != "sil" and phone != "sp":
                    new_para_sil_phone_list.append(phone)
            word_list, pos_list = self.get_word_pos_list(sent_content,"baidu")

            # 分句子，按照道理应该在切词的前面 但是由于比较方便就放在后面了
            # all_sent [(sent1's wordlist, pos list)，（sent2's wordlist, sent3's wordlist）]
            all_sent = self.sentence_spliting(word_list,pos_list)
            sent_num = len(all_sent)
            prev_sil_phone_index = 0 # index in align phone
            cur_sil_phone_index = 0
            all_verbal_phone = []
            for num, word_pos in enumerate(all_sent):
                # for each sentence and num is position of sentence in paragraph
                sent_word_list, sent_pos_list = word_pos
                # remove inter-sentence's punc and add prosodic information here
                new_word_list, new_pos_list,phrase_map = self.remove_punc(sent_word_list,sent_pos_list)
                logger.debug("sentence content {}".format(''.join(new_word_list)))
                phone_list, tone_list, syl_map, word_map,non_tone_line_phones = self.get_word_phone_list(new_word_list,using_tool=True)
                all_verbal_phone.extend(phone_list)
                non_sil_len = len(phone_list)
                cur_sil_phone_index += non_sil_len

                try:
                    sil_ali_phone_index = non_silence_index_list[cur_sil_phone_index-1]
                except IndexError:
                    pdb.set_trace()
                sil_phone_list = para_sil_phone_list[prev_sil_phone_index:sil_ali_phone_index+1]
               # pdb.set_trace()
                start_time_list = para_start_time_list[prev_sil_phone_index:sil_ali_phone_index+1]
                end_time_list = para_end_time_list[prev_sil_phone_index:sil_ali_phone_index+1]
                prev_sil_phone_index = sil_ali_phone_index+1

                non_sil_index = 0
                for sil_index, sil_phone in enumerate(sil_phone_list):
                    if sil_phone != 'sil' and sil_phone != 'sp':
                        sil_nonsil_map[sil_index] = non_sil_index
                        non_sil_index = non_sil_index + 1

                all_syl_len = len(tone_list)
                all_phrase_len = phrase_map[len(phrase_map)-1]+1
                iter_len = len(sil_phone_list)
                all_word_len = len(new_word_list)
                # because there is a phone named 'x' we can only use 'xx' as 'x'
                sil_phone_list.insert(0, 'xx')
                sil_phone_list.insert(0, 'xx')
                sil_phone_list.insert(len(sil_phone_list), 'xx')
                sil_phone_list.insert(len(sil_phone_list), 'xx')

                syl_phrase_map = self.find_syl_phrase_map(word_map,phrase_map)
                for index in range(iter_len):
                    # phone level
                    ll_phone = sil_phone_list[index]
                    l_phone = sil_phone_list[index + 1]
                    c_phone = sil_phone_list[index + 2]
                    r_phone = sil_phone_list[index + 3]
                    rr_phone = sil_phone_list[index + 4]
                    fw_ph_syl = 'x'
                    bw_ph_syl = 'x'

                    # syl level
                    prev_tone = 'x'
                    next_tone = 'x'
                    cur_tone = 'x'
                    # # of phone in pre next current phone
                    # # phones in previous syllabel
                    pre_syl_num = 'x'
                    next_syl_num = 'x'
                    cur_syl_num = 'x'
                    fw_syl_word = 'x'
                    bw_syl_word = 'x'
                    fw_syl_phrase = 'x'
                    bw_syl_phrase = 'x'

                    # word level
                    pre_word_num = 'x'
                    next_word_num = 'x'
                    cur_word_num = 'x'
                    prev_pos = 'xxx'
                    next_pos = 'xxx'
                    cur_pos = 'x'
                    fw_word_phrase = 'x'
                    bw_word_phrase = 'x'

                    # phrase level
                    pre_phrase_syl_num = 'x'
                    next_phrase_syl_num = 'x'
                    pre_phrase_word_num = 'x'
                    next_phrase_word_num = 'x'
                    cur_phrase_syl_num = 'x'
                    cur_phrase_word_num = 'x'
                    fw_phrase_utt = 'x'
                    bw_phrase_utt = 'x'
                    # pdb.set_trace()
                    if c_phone != 'sil' and c_phone != 'sp':
                        # current is not silence
                        non_sil_index = sil_nonsil_map[index]
                        fw_ph_syl, bw_ph_syl, syl_index, cur_syl_num = self.pos_phone_in_syl(non_sil_index, syl_map)
                        fw_syl_word, bw_syl_word, word_index, cur_word_num = self.pos_syl_in_word(syl_index, word_map)
                        fw_syl_phrase, bw_syl_phrase,phrase_index, cur_phrase_syl_num =self.pos_syl_in_phrase(syl_index,syl_phrase_map)
                        fw_word_phrase,bw_word_phrase,phrase_index,cur_phrase_word_num = self.pos_word_in_phrase(word_index,phrase_map)
                        fw_phrase_utt = phrase_index + 1
                        bw_phrase_utt = all_phrase_len - phrase_index
                        cur_tone = tone_list[syl_index]
                        # position of phone in the syllabel forward and backward
                        if syl_index != 0:
                            # if current syl index is not zero
                            prev_tone = tone_list[syl_index - 1]
                            pre_syl_num = self.get_phone_num_of_syl(syl_index - 1, syl_map)
                            if word_index != 0:
                                pre_word_num = self.get_syl_num_of_word(word_index - 1, word_map)
                                prev_pos = new_pos_list[word_index - 1]
                                if phrase_index != 0:
                                    pre_phrase_syl_num = self.get_syl_num_of_phrase(phrase_index-1,syl_phrase_map)
                                    pre_phrase_word_num = self.get_word_num_of_phrase(phrase_index-1, phrase_map)
                        if syl_index != all_syl_len - 1:
                            next_tone = tone_list[syl_index + 1]
                            next_syl_num = self.get_phone_num_of_syl(syl_index + 1, syl_map)
                            if word_index != all_word_len - 1:
                                next_word_num = self.get_syl_num_of_word(word_index + 1, word_map)
                                next_pos = new_pos_list[word_index + 1]
                                if phrase_index != all_phrase_len - 1:
                                    next_phrase_syl_num = self.get_syl_num_of_phrase(phrase_index+1, syl_phrase_map)
                                    next_phrase_word_num = self.get_syl_num_of_phrase(phrase_index+1, phrase_map)
                        cur_pos = pos_list[word_index]
                    else:
                        if index == 0:
                            # at the beginning of sent
                            _, _, next_syl_index, next_syl_num = self.pos_phone_in_syl(0, syl_map)
                            next_tone = tone_list[next_syl_index]
                            _, _, next_word_index, next_word_num = self.pos_syl_in_word(next_syl_index, word_map)
                            next_pos = pos_list[next_word_index]
                        elif index == iter_len - 1:
                            # at the end of senetence
                            _, _, pre_syl_index, pre_syl_num = self.pos_phone_in_syl(non_sil_len - 1, syl_map)
                            prev_tone = tone_list[pre_syl_index]
                            _, _, pre_word_index, pre_word_num = self.pos_syl_in_word(pre_syl_index, word_map)
                            prev_pos = pos_list[pre_word_index]
                        else:
                            non_sil_prev_index = sil_nonsil_map[index - 1]
                            non_sil_next_index = sil_nonsil_map[index + 1]
                            _, _, pre_syl_index, pre_syl_num = self.pos_phone_in_syl(non_sil_prev_index, syl_map)
                            _, _, next_syl_index, next_syl_num = self.pos_phone_in_syl(non_sil_next_index, syl_map)
                            _, _, pre_word_index, pre_word_num = self.pos_syl_in_word(pre_syl_index, word_map)
                            prev_pos = new_pos_list[pre_word_index]
                            _, _, next_word_index, next_word_num = self.pos_syl_in_word(next_syl_index, word_map)
                            next_pos = new_pos_list[next_word_index]
                    logger.info("write to sentence lab file:{0}".format(label_phone_path))
                    output_lab = "{38} {39} {0}^{1}-{2}+{3}={4}@{5}_{6}/A:{7}_{8}/B:{9}_{10}!{11}_{12}#{13}|{14}/C:{15}+{16}/D:{17}_{18}/E:{19}&{20}^{21}_{22}/F:{23}_{24}/G:{25}_{26}/H:{27}={28}^{29}|{30}/I:{31}={32}/J:{33}+{34}-{35}/K:{36}={37}\n".format(
                                ll_phone, l_phone, c_phone, r_phone, rr_phone, str(fw_ph_syl), str(bw_ph_syl),
                                str(prev_tone),str(pre_syl_num),  # A
                                str(cur_tone), str(cur_syl_num), str(fw_syl_word), str(bw_syl_word),str(fw_syl_phrase),str(bw_syl_phrase), # B
                                str(next_tone),str(next_syl_num), # C
                                prev_pos,str(pre_word_num), #D
                                cur_pos,  str(fw_word_phrase),str(bw_word_phrase), str(cur_word_num),#E
                                next_pos,str(next_word_num), #F
                                str(pre_phrase_syl_num),str(pre_phrase_word_num),
                                str(fw_phrase_utt),str(bw_phrase_utt),str(cur_phrase_syl_num),str(cur_phrase_word_num),
                                str(next_phrase_syl_num),str(next_phrase_word_num),
                                str(all_syl_len),str(all_word_len),str(all_phrase_len),
                                str(num+1),str(sent_num-num),
                                start_time_list[index], end_time_list[index])
                    logger.info(output_lab)
                    label_phone_fid.write(output_lab)
            label_phone_fid.close()

m_blz = blzFrontEnd()
m_blz.create_phone_labels()



class blz:
    def __init__(self):
        # baidu tokenizer
        APP_ID= '15716974'
        API_KEY='QZ4ee5tvyLrKCZ5FZib1eDFN'
        SECRET_KEY='Zhe7VieQlGeSvGoPbeHdfLLeDF78KOYO'
        self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

        #
        # self.text_file_path = "/home/gyzhang/speech_database/blizzard_release_2019_v1/text/text.txt"
        # self.ipa_dict = ld.get_lexicon_dict(lexicon_path='../data/dicts/ipa_m.dict')
        # self.english_dict = ld.get_lexicon_dict(lexicon_path='/home/gyzhang/speech_database/blizzard_release_2019_v1/text/english.dict')
        # self.chinese_syl_dict = ld.get_lexicon_dict(lexicon_path='/home/gyzhang/speech_database/blizzard_release_2019_v1/text/lexicon_char.txt')
        # self.chinese_syl_dict_new = '/home/gyzhang/speech_database/blizzard_release_2019_v1/text/lexicon_chinese_char.txt'
        # self.chinese_syl_dict_new = dict()
        # self.write_syl_phone_dict_flag=False
        # self.wav_dir = "/home/gyzhang/speech_database/blizzard_release_2019_v1/wav_16k"
        self.wav_scp = "../exp/blz/kaldi/wav.scp"
        self.utt2spk = "../exp/blz/kaldi/utt2spk"
        self.text = "../exp/blz/kaldi/text"
        self.write_kaldi=True

    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)


    def contains_letters(self, inputString):
        regexp = re.compile(r"[a-z]|[A-Z]")
        return regexp.search(inputString)

    def write_dict_text(self, chinese_syl_dict_new, chinese_syl_dict):
        with open(chinese_syl_dict_new, 'w') as lcid:
            for syl, syl_phone_list in chinese_syl_dict.items():
                lcid.write("{0} {1}\n".format(syl, ' '.join(syl_phone_list)))

    def re_name_wav(self,):
        for wav_path in glob.glob(self.wav_dir + '/*.wav'):
            wav_basename = os.path.basename(wav_path)
            old_utt_id = re.split('\.', wav_basename)[0]
            new_utt_id = re.split('\-',old_utt_id)[0]
            new_file_name = os.path.join(self.wav_dir, new_utt_id+'.wav')
            os.rename(wav_path,new_file_name)

    def create_wav_scp(self,):
        utt2spk_fid = open(self.utt2spk, 'w')
        wav_scp_fid = open(self.wav_scp, 'w')
        for wav_path in glob.glob(self.wav_dir + '/*.wav'):
            wav_basename = os.path.basename(wav_path)
            utt_id = re.split('\.', wav_basename)[0]
            utt2spk_fid.write('{0} {1}\n'.format(utt_id, "spk1"))
            wav_scp_fid.write('{0} {1}\n'.format(utt_id, wav_path))

    def multi_spks(self,):
        for wav_path in glob.glob(self.wav_dir + '/*.wav'):
            wav_basename = os.path.basename(wav_path)
            utt_id = re.split('\.', wav_basename)[0]
            try:
                os.remove(os.path.join(self.wav_dir, utt_id+' .lab'))
            except FileNotFoundError:
                print("not exists")

    def split_folder(self, target_dir, num_fold):
        count = 0
        for wav_path in glob.glob(self.wav_dir+'/*.wav'):
            wav_basename = os.path.basename(wav_path)
            utt_id = re.split('\.', wav_basename)[0]
            lab_path = os.path.join(self.wav_dir, utt_id+'.lab')
            fold_order = count%num_fold
            target_fold = os.path.join(target_dir,str(fold_order))
            target_fold_wav = os.path.join(target_fold, wav_basename)
            target_fold_lab = os.path.join(target_fold, utt_id+'.lab')
            if not os.path.exists(target_fold):
                os.mkdir(target_fold)
            copyfile(wav_path, target_fold_wav)
            copyfile(lab_path,target_fold_lab)
            count+=1

    def test(self, ): 
        self.text_file_path = "/home/gyzhang/speech_database/text/blz19/text.txt"
        self.text_punc = "/home/gyzhang/speech_database/text/blz19/punc.txt"
        k_fid = open(self.text, 'w')
        punc_list = text2list(self.text_punc)
        unpunc_list = []
        with open(self.text_file_path, 'r') as fid:
            text_lines = fid.readlines()
        for num, line in enumerate(text_lines):
            index = line[0:7]
            text_line = line.strip()[7:]
            if num%5==4:
                time.sleep(1)
            return_list = self.client.lexer(text_line)['items']
            for item in return_list:
                if item['pos'] == 'w':
                    if item['item'] not in punc_list:
                        unpunc_list.append(item['item'])
                        #print(item['item'])
                        #punc_list.append(item['item'])
                        print(line)
        print(unpunc_list)

    def create_lab_file(self,):
        k_fid = open(self.text, 'w')
        with open(self.text_file_path, 'r') as fid:
            text_lines = fid.readlines()
        for num, line in enumerate(text_lines):
            index = line[0:7]
            lab_file_path = os.path.join(self.wav_dir, index.strip()+'.lab')
            text_line = line.strip()[7:]
            """ 调用词法分析 """
            if num%5==4:
                time.sleep(1)
            return_list = self.client.lexer(text_line)['items']
            pinyin_line = []
            lexicon_syl = []
            # contain phone sequence of a line
            phone_line_list = []
            flag=False
    #     insert_flag=False
            for item in return_list:
                # determine current token is punc or not
                if item['pos'] != 'w':
                    # determine current token is digit or not
                    if self.hasNumbers(item['item']):
                        print("There is a error")
                        exit(0)
                    # determine current token is english
                    if self.contains_letters(item['item']):
                        # print(item['item'],item['pos'], item['basic_words'], index)
                        m_list = re.findall(r'([a-zA-Z]+)', item['item'])
                        for m in m_list:
                            try:
                                # english phone sequence which is prepared before.
                                a = self.english_dict[m]
                                # print(m, english_dict[m])
                            except KeyError:
                                print("wrong key error")
                    for character in pinyin(item['item'],style=Style.NORMAL):
                        if character[0] in self.english_dict and flag:
                            flag = False
                            if character[0] == "you":
                                character = ["E_you"]
                            pinyin_line.extend(character)
                            continue
                        pinyin_line.extend(character)

            if self.write_kaldi:
                k_fid.write('{0} {1}\n'.format(index,' '.join(pinyin_line)))
            else:
                with open(lab_file_path, 'w') as lab_fid:
                    print(pinyin_line)
                    lab_fid.write('{1}'.format(' '.join(pinyin_line)))
