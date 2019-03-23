from pypinyin import pinyin,Style, style
from linguistic_dict import Linguistic_DICT
ld = Linguistic_DICT()
import urllib, sys
import ssl
import time
import re
import os
import glob
from aip import AipNlp
from shutil import copyfile


class blz:
    def __init__(self):
        # baidu tokenizer
        APP_ID= '15716974'
        API_KEY='QZ4ee5tvyLrKCZ5FZib1eDFN'
        SECRET_KEY='Zhe7VieQlGeSvGoPbeHdfLLeDF78KOYO'
        self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

        #
        self.text_file_path = "/home/gyzhang/speech_database/blizzard_release_2019_v1/text/text.txt"
        self.ipa_dict = ld.get_lexicon_dict(lexicon_path='../data/dicts/ipa_m.dict')
        self.english_dict = ld.get_lexicon_dict(lexicon_path='/home/gyzhang/speech_database/blizzard_release_2019_v1/text/english.dict')
        self.chinese_syl_dict = ld.get_lexicon_dict(lexicon_path='/home/gyzhang/speech_database/blizzard_release_2019_v1/text/lexicon_char.txt')
        self.chinese_syl_dict_new = '/home/gyzhang/speech_database/blizzard_release_2019_v1/text/lexicon_chinese_char.txt'
        self.chinese_syl_dict_new = dict()
        self.write_syl_phone_dict_flag=False
        self.wav_dir = "/home/gyzhang/speech_database/blizzard_release_2019_v1/wav_16k"
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


    def create_lab_file(self,):
        k_fid = open(self.text, 'w')
        with open(self.text_file_path, 'r') as fid:
            text_lines = fid.readlines()
        for num, line in enumerate(text_lines):
            index = line[0:7]
            lab_file_path = os.path.join(self.wav_dir, index.strip()+'.lab')
            text_line = line.strip()[7:]
            # print(index, text_line)
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
                                a = self.english_dict[m]
                                flag = True
                                # print(m, english_dict[m])
                            except KeyError:
                                print("wrong key error")
                    for character in pinyin(item['item'],style=Style.NORMAL):
                        if character[0] in self.english_dict and flag:
                            # print(character)
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

m_blz = blz()
m_blz.split_folder('/home/gyzhang/speech_database/blizzard_release_2019_v1/wav_multi', 16)

# with open(text_file_path, 'r') as fid:
#     text_lines = fid.readlines()
# for num, line in enumerate(text_lines):
#     index = line[0:7]
#     text_line = line.strip()[7:]
#     # print(index, text_line)
#     """ 调用词法分析 """
#     if num%5==4:
#         time.sleep(1)
#     return_list = client.lexer(text_line)['items']
#     pinyin_line = []
#     lexicon_syl = []
#     # contain phone sequence of a line
#     phone_line_list = []
#     flag=False
#     # derermine whether insert a symbol to our chinese syllable to phone dict
#     insert_flag=False
#     for item in return_list:
#         # determine current token is punc or not
#         if item['pos'] != 'w':
#             # determine current token is digit or not
#             if hasNumbers(item['item']):
#                 print(item['item'], item['pos'], index)
#             # determine current token is english
#             if contains_letters(item['item']):
#                 # print(item['item'],item['pos'], item['basic_words'], index)
#                 m_list = re.findall(r'([a-zA-Z]+)', item['item'])
#                 for m in m_list:
#                     try:
#                         a = english_dict[m]
#                         flag = True
#                         # print(m, english_dict[m])
#                     except KeyError:
#                         print("wrong key error")
#             for character in pinyin(item['item'],style=Style.NORMAL):
#                 if character not in lexicon_syl:
#                     lexicon_syl.append(character)
#                 if character[0] in english_dict and flag:
#                     # print(character)
#                     flag = False
#                     continue
#                 # if character[0]
#                 if character[0] not in chinese_syl_dict:
#                     print(character)
#                     insert_flag = True
#                 initial = style.convert(character[0], Style.INITIALS, True)
#                 final = style.convert(character[0], Style.FINALS, True)
#                 if initial != "":
#                     try:
#                         ipa_initial = ipa_dict[initial]
#                     except KeyError:
#                         print(item['item'],character)
#                     # syl_phone_list.extend(ipa_initial)
#                     # print(ipa_initial,ipa_final)
#                 if final != "":
#                     try:
#                         ipa_final = ipa_dict[final]
#                     except KeyError:
#                         print(item['item'],character)
#                     # print(ipa_final)
#                 if insert_flag:
#                     syl_phone = []
#                     syl_phone.extend(ipa_initial)
#                     syl_phone.extend(ipa_final)
#                     chinese_syl_dict[character[0]] = syl_phone
#                 pinyin_line.extend(character)
#
#
#     # print(lexicon_syl)
#     # break