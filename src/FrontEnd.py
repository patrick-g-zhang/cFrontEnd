import pdb
import os
import glob
import logging
import re
import jieba
# jieba.initialize()  # (optional)
# jieba.load_userdict('/home/gyzhang/projects/cFrontEnd/data/dicts/dict_name.dict')
# from jieba import posseg
import time
# import pdb
import json, uuid, http.client, urllib.parse
import pycantonese as pc
from utils import *
import jyutping
from collections import OrderedDict
from linguistic_dict import Linguistic_DICT
from jieba import posseg
import tensorflow as tf
from aip import AipNlp
from pypinyin import pinyin,Style, style
posseg.initialize(dictionary='../data/dicts/dict_name.dict')
ld = Linguistic_DICT()
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration parameters for front end

class FrontEnd(object):
    def __init__(self, ):
        # project and dictionary file paths
        self.project_path = '/home/gyzhang/projects/cFrontEnd'
        self.wav_folder = '/home/gyzhang/speech_database/cuprosody/Wave'
        self.name = "cuprosody"
        self.text_file = os.path.join(self.project_path, "exp", self.name, 'train/cn_text.txt')
        # this is test file out of this domain
        self.test_text_file = os.path.join(self.project_path, "exp", self.name, "train/cn_text_test.txt")
        self.mld = Linguistic_DICT()
        self.word_dict = self.mld.get_phone_dict(dict_file=os.path.join(self.project_path, 'data/dicts/word2jyut.lex'))
        self.lex_dict = self.mld.get_lexicon_dict(os.path.join(self.project_path, "data/dicts/lexicon.txt"))



        # files for kaldi project
        self.wav_file = os.path.join(self.project_path, "exp", self.name, "kaldi/wav.scp")
        self.utt2spk = os.path.join(self.project_path, "exp", self.name, "kaldi/utt2spk")
        self.kaldi_text_no_tone = os.path.join(self.project_path, "exp", self.name, "kaldi/no_tone_kaldi_text")

        # files for training
        self.written_text_file = os.path.join(self.project_path, "exp", self.name, 'train/clean_text_gen')
        self.file_id_path = os.path.join(self.project_path, "exp", self.name, "train/file_id_list.scp")
        self.label_phone_align = os.path.join(self.project_path, "exp", self.name, "train/label_phone_align")
        self.test_label_phone_align = os.path.join(self.project_path, "exp", self.name, "train/test_label_phone_align")
        # when other files are in the directory
        # this is alignment files which used to recognize non-silence utterance
        # you must update to the latest alignment file directory
        self.aud_format_dir = os.path.join(self.project_path, "exp", self.name, "train/aud_ali_0828855")

        logger = logging.getLogger('create front end for cantonese')

    def pos_word_in_phrase(self, word_index, phrase_map):
        phrase_index = phrase_map[word_index]
        phrase_index_list = []
        for key, value in phrase_map.items():
            if value == phrase_index:
                phrase_index_list.append(key)
        phrase_len = len(phrase_index_list)
        fw_word_pos = phrase_index_list.index(word_index) + 1
        bw_word_pos = phrase_len - fw_word_pos + 1
        return fw_word_pos, bw_word_pos, phrase_index, phrase_len

    def pos_syl_in_phrase(self, char_index, phrase_map):
        phrase_index = phrase_map[char_index]
        phrase_index_list = []
        for key, value in phrase_map.items():
            if value == phrase_index:
                phrase_index_list.append(key)
        phrase_len = len(phrase_index_list)
        fw_syl_pos = phrase_index_list.index(char_index) + 1
        bw_syl_pos = phrase_len - fw_syl_pos + 1
        return fw_syl_pos, bw_syl_pos, phrase_index, phrase_len


    def find_syl_phrase_map(self, word_map,phrase_map):
        # syl in phrase map
        syl_phrase_map = OrderedDict()
        for key,value in word_map.items():
            # key is syl and value in word
            syl_phrase_map[key] = phrase_map[value]
        return syl_phrase_map


    def pos_phone_in_syl(self, phone_index, syl_map):
        syl_index = syl_map[phone_index]
        syl_index_list = []
        for key, value in syl_map.items():
            if value == syl_index:
                syl_index_list.append(key)
        syl_len = len(syl_index_list)
        fw_pos = syl_index_list.index(phone_index) + 1
        bw_pos = syl_len - fw_pos + 1
        return fw_pos, bw_pos, syl_index, syl_len

    def get_phone_num_of_syl(self, syl_index, syl_map):
        syl_index_list = []
        for key, value in syl_map.items():
            if value == syl_index:
                syl_index_list.append(key)
        syl_len = len(syl_index_list)
        return syl_len

    def pos_syl_in_word(self, char_index, word_map):
        word_index = word_map[char_index]
        word_index_list = []
        for key, value in word_map.items():
            if value == word_index:
                word_index_list.append(key)
        word_len = len(word_index_list)
        fw_syl_pos = word_index_list.index(char_index) + 1
        bw_syl_pos = word_len - fw_syl_pos + 1
        return fw_syl_pos, bw_syl_pos, word_index, word_len

    def get_syl_num_of_word(self, word_index, word_map):
        word_index_list = []
        for key, value in word_map.items():
            if value == word_index:
                word_index_list.append(key)
        word_len = len(word_index_list)
        return word_len

    def get_syl_num_of_phrase(self, phrase_index, phrase_map):
        phrase_index_list = []
        for key, value in phrase_map.items():
            if value == phrase_index:
                phrase_index_list.append(key)
        word_len = len(phrase_index_list)
        return word_len

    def get_word_num_of_phrase(self, phrase_index , phrase_map):
        word_index_list = []
        for key, value in phrase_map.items():
            if value == phrase_index:
                word_index_list.append(key)
        word_len = len(word_index_list)
        return word_len

    def create_wav_file(self, ):
        """
        Create wav.scp and utt2spk files for kaldi project

        Arguments:
        wav_folder -- folder with wave files with wave files like '10001.wav'
        wav_file -- our scp file

        """

        f_out = open(self.wav_file, 'w')
        u_utt2spk = open(self.utt2spk, 'w')
        for file in glob.glob(self.wav_folder + '/*.wav'):
            base = os.path.basename(file).split('.')[0]
            # write to scp file
            f_out.write(base + '\t' + file + '\n')
            u_utt2spk.write(base + '\t' + 'tts' + '\n')

    def sentence_spliting(self, word_list, pos_list):
        """
        because of history, first we toknize our paragraph the we split sentence
        Here the criterion for sentence is ！… ：；  。
        :param word_list:
        :param pos_list:
        :param raw_text:
        :return:
        [(word_list,pos_list),(word_list,pos_list)]
        """
        all_sent = []
        sent_word = []
        sent_pos = []
        for word, pos in zip(word_list, pos_list):
            if pos == "w":
                if word in ["·" ,"《" ,"》","“","”","「","」" ]:
                    continue
                if word in ["！","… ","；" ,"。","？" ]:
                    all_sent.append((sent_word,sent_pos))
                    sent_word=[]
                    sent_pos=[]
                    continue
            sent_word.append(word)
            sent_pos.append(pos)
        return all_sent

    def remove_punc(self, word_list, pos_list):
        """
        remove punc of each word
        in the further it will return position of punc as intonation boundary
        add here using prosidc map: [w1:p1,w2,p1,w3,p2]
        :param word_list:
        :param pos_list:
        :return:
        """
        new_word_list = []
        new_pos_list = []
        prosodic_map = OrderedDict()
        word_index = 0
        phrase_index = 0
        for word, pos in zip(word_list,pos_list):
            if word == "":
                continue
            prosodic_map[word_index] = phrase_index
            if pos == "w":
                # here we just use punctuation as prosodic boundary
                phrase_index += 1
                continue
            word_index += 1
            new_word_list.append(word)
            new_pos_list.append(pos)
        logging.info(''.join(new_word_list))
        return new_word_list,new_pos_list, prosodic_map

    def get_word_pos_list(self, raw_text, tokenizer):
        """
            get word and pos lists from raw text

        Parameters:
            raw_text -- raw input as a sentence in our corpus
            tokenizer -- which tokenizer you choose to use baidu or jieba
        Returns:
            word_list -- list of words in a sentence
            pos_list -- list of pos in a sentence
        """
        raw_text = raw_text.strip()
        word_list = []
        pos_list = []
        # pdb.set_trace()
        if tokenizer == "jieba":
            seg_list = jieba.posseg.cut(raw_text, HMM=False)  # 默认是精确模式
            for word, flag in seg_list:
                # remove the punctuation
                if word in ['「', '」', '.', '－', '', '  ', '。', '—', '？', '：', '、', '…', '；', '，', ',', '！']:
                    continue
                word_list.append(word)
                pos_list.append(flag)
        elif tokenizer == "baidu":
            return_list = self.client.lexer(raw_text)['items']

            for item in return_list:
                # we should contain blank and it will be recognized as punc
                if item['pos'] == 'w' and item['item'] == ' ':
                    print("output blank punc")
                # since baidu contains name entity recognition function, we merge ne and pos to pos list.
                if item['pos'] == '':
                    pos_list.append(item['ne'])
                else:
                    pos_list.append(item['pos'])
                word_list.append(item['item'])
        return word_list, pos_list

    def get_word_phone_list(self,word_list,using_tool):
        """
            get phone list and phone array of word_list

        Parameters:
            word_dict -- dictionary of word
            word_list -- list of words in a sentence ["我"，"是个"] without non-verbal information
            using_tool -- whether use tool instead of dictionary to fetch phone sequence
            lang -- cantonse or mandarin
        TO DO: add more functions for language support
        :return
            phone list : ph e m e j
            tone list : 1 2 3
            syl_map: [p1:s1,p2,s1,p3,s1,p4,p4]
            word_map
            non_tone_line_phones
        """
        self.english_dict = ld.get_lexicon_dict(lexicon_path='/Users/patrickzhang/Documents/text/english.dict')
        self.chinese_dict = ld.get_lexicon_dict(lexicon_path="/Users/patrickzhang/Documents/text/lexicon_chinese_char.txt")
        flag = False
        phone_list = []
        tone_list = []
        syl_map = OrderedDict()
        word_map = OrderedDict()
        # phone index the index of phone in one sentence
        phone_index = 0
        # char index the index of char in one sentence
        char_index = 0

        word_index = 0
        non_tone_line_phones = []
        for word in word_list:
            word = word.strip()
            # get the phone or word
            if not using_tool:
                try:
                    word_phone = word_dict[word]
                except Exception as e:
                    temp_word_phone = jyutping.get(word)
                    temp_word_phone_renew = []
                    # if polyphone appear, just pick first one
                    for char_phone in temp_word_phone:
                        if isinstance(char_phone, list):
                            temp_word_phone_renew.append(char_phone[0])
                        else:
                            temp_word_phone_renew.append(char_phone)
                    word_phone = ''.join(temp_word_phone_renew)
                    # word_phone [('j', 'a', 't', '1'),
                    #  ('g', 'a', 'u', '2'),
                    #  ('s', 'e', 'i', '3'),
                    #  ('g', 'a', 'u', '2'),
                    #  ('n', 'i', 'n', '4')]
                    if word_phone == 'hng1':
                        word_phone_list = [('h', 'ng', '1')]
                    elif word_phone == 'ung2':
                        word_phone_list = [('u', 'ng', '2')]
                    else:
                        try:
                            word_phone_list = pc.parse_jyutping(word_phone)
                        except Exception as e:
                            pdb.set_trace()
            else:
                word_phone_list = []
                if hasNumbers(word):
                    logging.warnings("There is a error")
                    exit(0)
                if contains_letters(word):
                    # pdb.set_trace()
                    # 字里现在有字母
                    m_list = re.findall(r'([a-zA-Z]+)', word)
                    logging.warning("there is a letter {0}".format(word))
                    for m in m_list:
                        try:
                            # Here we need a english dictionary to get phone sequence of
                            english_seq = self.english_dict[m]
                            # add tone of english
                            english_seq.append('5')
                            word_phone_list.append(english_seq)
                            flag = True
                        except KeyError:
                            logging.warning("wrong key error")
                else:
                    for character in pinyin(word,style=Style.TONE3):
                        if character[0][:-1] in self.english_dict and flag:
                            flag = False
                            if character[0] == "you":
                                character = ["E_you"]
                            continue
                        if not character[0][-1].isdigit():
                            # 轻声作为第五声
                            character[0]+='5'
                        # assert character[0][-1].isdigit()
                        char_phone_sequence = []
                        char_phone_sequence = self.chinese_dict[character[0][:-1]].copy()
                        char_phone_sequence.append(character[0][-1])
                        word_phone_list.append(char_phone_sequence)

            for phone_t in word_phone_list:
                char_phone = phone_t
                char_phone = [e_phone for e_phone in char_phone if e_phone != '']
                assert char_phone[-1].isdigit()
                char_phone_list = char_phone[:-1]
                for my_phone in char_phone_list:
                    syl_map[phone_index] = char_index
                    phone_index = phone_index + 1
                phone_list.extend(char_phone_list)
                tone_list.append(char_phone[-1])
                word_map[char_index] = word_index
                char_index = char_index + 1
                non_tone_line_phones.append(''.join(char_phone[:-1]))
            word_index = word_index + 1
        #     logging.debug("phone_list:" + ' '.join(phone_list))
        return phone_list, tone_list, syl_map, word_map, non_tone_line_phones

    def pre_process(self, raw_text):
        """
        input raw text
        and output line
        """
        # remove the space or other symbols
        word_lists = re.split(r'\s+', raw_text.strip())
        if len(word_lists) < 2:
            print(word_lists)
        #  exit(1)
        sent_index = word_lists[0]
        word_lists = ''.join(word_lists[1:])
        # word_lists = re.split(r'。', word_lists)
        # sent_content = ''.join(word_lists)
        return sent_index, word_lists

    def create_file_for_kaldi(self, ):
        def intersection(lst1, lst2):
            lst3 = [value for value in lst1 if value in lst2]
            return lst3

        all_pos_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'ng', 'nr', 'ns', 'nt',
                        'o', 'p', 'q', 'r', 'rg', 'rr', 'rz', 's', 't', 'u', 'v', 'y', 'z']
        with open(self.text_file, 'r') as fid:
            textlines = fid.readlines()
        kfid = open(self.kaldi_text_no_tone, 'w')
        for text_line in textlines:
            logging.info("process sentence: " + text_line)
            sent_index, sent_content = self.pre_process(text_line)
            word_list, pos_list = self.get_word_pos_list(sent_content)
            intersect_list = intersection(pos_list, all_pos_list)
            if len(intersect_list) != len(pos_list):
                indices = pos_list.index('x')
                print(word_list[indices])
                pdb.set_trace()
            # phone_list, tone_list, syl_map, word_map = self.get_word_phone_list(word_list)
            phone_list, tone_list, syl_map, word_map, non_tone_line_phones = self.get_word_phone_list(self.word_dict, word_list)
            # logging.info(word_list)
            # logging.info(pos_list)
            # logging.info(phone_list)
            non_tone_line = sent_index + ' ' + ' '.join(non_tone_line_phones) + '\n'
            kfid.write(non_tone_line)
        kfid.close()
        logging.info('all sentences are processed')

    def create_phone_labels(self, training_data):
        # with open(self.text_file, 'r') as fid:
        # textlines = fid.readlines()
        if training_data:
            text_file = self.text_file
            label_phone_dir = self.label_phone_align
        else:
            text_file = self.test_text_file
            label_phone_dir = self.test_label_phone_align
        with open(text_file, 'r') as fid:
            textlines = fid.readlines()
        file_id_canto = open(self.file_id_path, 'w')
        for text_line in textlines:
            sent_index, sent_content = self.pre_process(text_line)
            word_list, pos_list = self.get_word_pos_list(sent_content)
            phone_list, tone_list, syl_map, word_map, _ = self.get_word_phone_list(self.word_dict, word_list)
            file_id_canto.write(sent_index + '\n')
            sil_phone_list = []
            start_time_list = []
            end_time_list = []
            if training_data:
                # extract alignment file
                ali_file = os.path.join(self.aud_format_dir, sent_index + '.txt')
                try:
                    with open(ali_file, 'r') as fid:
                        ali_file_lines = fid.readlines()
                except Exception as e:
                    print(ali_file)
                    continue
                for line in ali_file_lines:
                    line = line.strip()
                    line_list = re.split('\s+', line)
                    start_time = str(int(round(float(line_list[0]), 3) * 10000000))
                    end_time = str(int(round(float(line_list[1]), 3) * 10000000))
                    phone = line_list[2]
                    phone = re.split('_', phone)[0]
                    sil_phone_list.append(phone)
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
            else:
                sil_phone_list = phone_list[:]
                sil_phone_list.insert(0, 'sil')
                sil_phone_list.insert(len(sil_phone_list), 'sil')
            sil_nonsil_map = OrderedDict()
            non_sil_index = 0
            for sil_index, sil_phone in enumerate(sil_phone_list):
                if sil_phone != 'sil':
                    sil_nonsil_map[sil_index] = non_sil_index
                    non_sil_index = non_sil_index + 1
            label_phone_path = os.path.join(label_phone_dir, sent_index + '.lab')
            label_phone_fid = open(label_phone_path, 'w')
            non_sil_len = len(phone_list)
            all_syl_len = len(tone_list)
            iter_len = len(sil_phone_list)
            all_word_len = len(word_list)
            sil_phone_list.insert(0, 'x')
            sil_phone_list.insert(0, 'x')
            sil_phone_list.insert(len(sil_phone_list), 'x')
            sil_phone_list.insert(len(sil_phone_list), 'x')
            for index in range(iter_len):
                ll_phone = sil_phone_list[index]
                l_phone = sil_phone_list[index + 1]
                c_phone = sil_phone_list[index + 2]
                r_phone = sil_phone_list[index + 3]
                rr_phone = sil_phone_list[index + 4]
                fw_ph_syl = 'x'
                bw_ph_syl = 'x'
                prev_tone = '0'
                next_tone = '0'
                cur_tone = 'x'
                pre_syl_num = '0'
                next_syl_num = '0'
                cur_syl_num = 'x'
                fw_syl_word = 'x'
                bw_syl_word = 'x'
                pre_word_num = '0'
                next_word_num = '0'
                cur_word_num = 'x'
                prev_pos = '0'
                next_pos = '0'
                cur_pos = 'x'
                fw_word_utt = 'x'
                bw_word_utt = 'x'
                # pdb.set_trace()
                if c_phone != 'sil':
                    non_sil_index = sil_nonsil_map[index]
                    fw_ph_syl, bw_ph_syl, syl_index, cur_syl_num = self.pos_phone_in_syl(non_sil_index, syl_map)
                    fw_syl_word, bw_syl_word, word_index, cur_word_num = self.pos_syl_in_word(syl_index, word_map)
                    fw_word_utt = word_index + 1
                    bw_word_utt = len(word_list) - word_index
                    cur_tone = tone_list[syl_index]
                    # position of phone in the syllabel forward and backward
                    if syl_index != 0:
                        prev_tone = tone_list[syl_index - 1]
                        pre_syl_num = self.get_phone_num_of_syl(syl_index - 1, syl_map)
                        if word_index != 0:
                            pre_word_num = self.get_syl_num_of_word(word_index - 1, word_map)
                            prev_pos = pos_list[word_index - 1]
                    if syl_index != all_syl_len - 1:
                        next_tone = tone_list[syl_index + 1]
                        next_syl_num = self.get_phone_num_of_syl(syl_index + 1, syl_map)
                        if word_index != all_word_len - 1:
                            next_word_num = self.get_syl_num_of_word(word_index + 1, word_map)
                            next_pos = pos_list[word_index + 1]
                    cur_pos = pos_list[word_index]
                else:
                    if index == 0:
                        # at the beginning of sent
                        _, _, next_syl_index, next_syl_num = self.pos_phone_in_syl(0, syl_map)
                        next_tone = tone_list[next_syl_index]
                        _, _, next_word_index, next_word_num = self.pos_syl_in_word(next_syl_index, word_map)
                        next_pos = pos_list[next_word_index]
                    elif index == iter_len - 1:
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
                        prev_pos = pos_list[pre_word_index]
                        _, _, next_word_index, next_word_num = self.pos_syl_in_word(next_syl_index, word_map)
                        next_pos = pos_list[next_word_index]
                if training_data:
                    logging.info("write to sentence lab file:{0}".format(label_phone_path))
                    label_phone_fid.write(
                        "{25} {26} {0}^{1}-{2}+{3}={4}@{5}_{6}/A:{7}_{8}/B:{9}_{10}!{11}_{12}/C:{13}+{14}/D:{15}-{16}/E:{17}&{18}^{19}_{20}/F:{21}_{22}/G:{23}-{24}\n".format(
                            ll_phone, l_phone, c_phone, r_phone, rr_phone,
                            str(fw_ph_syl), str(bw_ph_syl), str(prev_tone), str(pre_syl_num), str(cur_tone),
                            str(cur_syl_num), str(fw_syl_word), str(bw_syl_word),
                            str(next_tone), str(next_syl_num), prev_pos, pre_word_num, cur_pos, cur_word_num,
                            str(fw_word_utt), str(bw_word_utt), next_pos, next_word_num, str(all_syl_len),
                            str(len(word_list)), start_time_list[index], end_time_list[index]))
                else:
                    label_phone_fid.write(
                        "{0}^{1}-{2}+{3}={4}@{5}_{6}/A:{7}_{8}/B:{9}_{10}!{11}_{12}/C:{13}+{14}/D:{15}-{16}/E:{17}&{18}^{19}_{20}/F:{21}_{22}/G:{23}-{24}\n".format(
                            ll_phone, l_phone, c_phone, r_phone, rr_phone,
                            str(fw_ph_syl), str(bw_ph_syl), str(prev_tone), str(pre_syl_num), str(cur_tone),
                            str(cur_syl_num), str(fw_syl_word), str(bw_syl_word),
                            str(next_tone), str(next_syl_num), prev_pos, pre_word_num, cur_pos, cur_word_num,
                            str(fw_word_utt), str(bw_word_utt), next_pos, next_word_num, str(all_syl_len),
                            str(len(word_list))))
            label_phone_fid.close()
        file_id_canto.close()


if __name__ == '__main__':
    mf = CFrontEnd()
    # mf.create_wav_file()
    # mf.create_file_for_kaldi()
    mf.create_phone_labels(False)
