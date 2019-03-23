import jieba
from jieba import posseg
from linguistic_dict import Linguistic_DICT
jieba.set_dictionary('./freq_merged.dict')
import pycantonese as pc
import re
import pdb
import logging
from collections import OrderedDict
import pdb
import glob
import os
def create_sil_file():
    ali_files = glob.glob('./cup_ali/*.txt')
    phone_with_sil = './phone_with_sil.txt'
    mfid = open(phone_with_sil, 'w')
    all_files_num = len(ali_files)
    for sent_index in range(all_files_num):
        ali_file = './cup_ali/' + str(sent_index + 1) + '.txt'
        with open(ali_file, 'r') as fid:
            ali_file_lines = fid.readlines()
        phone_list = []
        for line in ali_file_lines:
            line = line.strip()
            line_list = re.split('\s+', line)
            start_time = str(int(round(float(line_list[0]), 3) * 10000000))
            end_time = str(int(round(float(line_list[1]), 3) * 10000000))
            phone = line_list[2]
            phone = re.split('_', phone)[0]
            phone_list.append(phone)
            phone_line = str(sent_index) + ' ' + start_time + ' ' + end_time +' ' + ' '.join(phone_list) + '\n'
        mfid.write(phone_line)
    mfid.close()

def pre_process(raw_text):
    """
    input raw text
    and output line
    """
    # remove the space and
    word_lists = re.split(r'\s+', raw_text.strip())

    sent_index = word_lists[0]
    word_lists = word_lists[1:]
    sent_content = ''.join(word_lists)
    return sent_index, sent_content

def get_word_pos_list(raw_text):
    """
        get word and pos lists from raw text

    Parameters:
        raw_text -- raw input as a sentence in our corpus
    Returns:
        word_list -- list of words in a sentence
        pos_list -- list of pos in a sentence
    """
    raw_text = raw_text.strip()
    word_list = []
    pos_list = []
    seg_list = posseg.cut(raw_text, HMM=False)  # 默认是精确模式
#     seg_list = jieba.cut(raw_text, HMM=False)  # 默认是精确模式
    # will store the phone list of
    for word, flag in seg_list:
        word_list.append(word)
        pos_list.append(flag)
    return word_list, pos_list


def get_word_phone_list(word_dict, word_list):
    """
        get phone list and phone array of word_list

    Parameters:
        word_list -- list of words in a sentence ["我"，"是个"]
        create_oov -- oov dictionary needed to be created at first step.
    """

    phone_list = []
    tone_list = []
    syl_map = OrderedDict()
    word_map = OrderedDict()
    phone_index = 0
    char_index = 0
    word_index = 0
    lex_dict = mld.get_lexicon_dict('./lexicon.txt')
    for word in word_list:
        word = word.strip()
        try:
            word_phone = word_dict[word]
        except Exception as e:
            word_phone_list = []
            # need find one by one
            for char in list(word):
                try:
                    char_phone = word_dict[char]
                except Exception as e:
                    char_phone = mld.search_single_char(word_dict, char)
                    if char_phone is None:
                        print(char)
                char_phone_list = char_phone.strip().split(" ")
                word_phone_list.extend(char_phone_list)
            word_phone = ''.join(word_phone_list)
        jp = pc.parse_jyutping(word_phone)
        for phone_t in jp:
            char_phone = list(phone_t)
            char_phone = [e_phone for e_phone in char_phone if e_phone != '']
            assert char_phone[-1].isdigit()
            char_phone_list = lex_dict[''.join(char_phone[:-1])]
            for my_phone in char_phone_list:
                syl_map[phone_index] = char_index
                phone_index = phone_index + 1
            phone_list.extend(char_phone_list)
            tone_list.append(char_phone[-1])
            word_map[char_index] = word_index
            char_index = char_index + 1
            # non_tone_line_phones.append(''.join(char_phone[:-1]))
        word_index = word_index + 1
    #     logging.debug("phone_list:" + ' '.join(phone_list))
    return phone_list, tone_list, syl_map, word_map


def pos_phone_in_syl(phone_index, syl_map):
    syl_index = syl_map[phone_index]
    syl_index_list = []
    for key, value in syl_map.items():
        if value == syl_index:
            syl_index_list.append(key)
    syl_len = len(syl_index_list)
    fw_pos = syl_index_list.index(phone_index) + 1
    bw_pos = syl_len - fw_pos + 1
    return fw_pos, bw_pos, syl_index, syl_len

def get_phone_num_of_syl(syl_index, syl_map):
    syl_index_list = []
    for key, value in syl_map.items():
        if value == syl_index:
            syl_index_list.append(key)
    syl_len = len(syl_index_list)
    return syl_len

def pos_syl_in_word(char_index, word_map):
    word_index = word_map[char_index]
    word_index_list = []
    for key,value in word_map.items():
        if value == word_index:
            word_index_list.append(key)
    word_len = len(word_index_list)
    fw_syl_pos = word_index_list.index(char_index) + 1
    bw_syl_pos = word_len - fw_syl_pos + 1
    return  fw_syl_pos, bw_syl_pos, word_index, word_len

def get_syl_num_of_word(word_index, word_map):
    word_index_list = []
    for key, value in word_map.items():
        if value == word_index:
            word_index_list.append(key)
    word_len = len(word_index_list)
    return word_len
# created_file = True
# if created_file:
#     create_sil_file()
training_data = True
state = True
if training_data:
    file_id_path = './file_id_list_canto.scp'
    text_file = './raw_text.txt'
    label_phone_align = './label_state_align/'
else:
    file_id_path = './test_id_list.scp'
    text_file = './test_raw_text.txt'
    label_phone_align = './prompt-lab/'

file_id_canto = open(file_id_path,'w')
mld = Linguistic_DICT()
word_dict = mld.get_phone_dict(dict_file='./word2jyut.lex')
with open(text_file, 'r') as fid:
    textlines = fid.readlines()

# for training data , this will create time duration
ali_files = glob.glob('./cup_ali/*.txt')
all_files_num = len(ali_files)
if not os.path.exists(label_phone_align):
    os.mkdir(label_phone_align)
for text_line in textlines:
    sent_index, sent_content = pre_process(text_line)
    word_list, pos_list = get_word_pos_list(sent_content)
    phone_list, tone_list, syl_map, word_map = get_word_phone_list(word_dict, word_list)
    file_id_canto.write(sent_index+'\n')
    sil_phone_list = []
    start_time_list = []
    end_time_list  = []
    phone_state_list = []
    phone_state_list_sil_phone = dict()
    if training_data:
        # extract alignment file
        ali_file = './align_phone/cup_state_ali/' + sent_index + '.txt'
        with open(ali_file, 'r') as fid:
            ali_file_lines = fid.readlines()
        sil_phone_index = -1
        for line in ali_file_lines:
            line = line.strip()
            line_list = re.split('\s+', line)
            start_time = str(int(round(float(line_list[0]), 3) * 10000000))
            end_time = str(int(round(float(line_list[1]), 3) * 10000000))
            phone = line_list[2]
            phone = re.split('_', phone)[0]
            state_phone = line_list[3]
            start_time_list.append(start_time)
            end_time_list.append(end_time)
            if state_phone == "[2]":
                sil_phone_list.append(phone)
                sil_phone_index = sil_phone_index + 1
            if phone == "sil" and state_phone not in ["[2]", "[3]", "[4]"]:
                continue
            phone_state_list.append((phone, state_phone, sil_phone_index))


    else:
        sil_phone_list = phone_list[:]
        sil_phone_list.insert(0, 'sil')
        sil_phone_list.insert(len(sil_phone_list),'sil')
    sil_nonsil_map = OrderedDict()
    non_sil_index = 0
    for sil_index, sil_phone in enumerate(sil_phone_list):
        if sil_phone != 'sil':
            sil_nonsil_map[sil_index] = non_sil_index
            non_sil_index = non_sil_index + 1

    label_phone_path = label_phone_align+sent_index+'.lab'
    label_phone_fid = open(label_phone_path, 'w')
    non_sil_len = len(phone_list)
    all_syl_len = len(tone_list)
    iter_len = len(phone_state_list)
    phone_iter_len = len(sil_phone_list)
    all_word_len = len(word_list)
    sil_phone_list.insert(0, 'x')
    sil_phone_list.insert(0,'x')
    sil_phone_list.insert(len(sil_phone_list), 'x')
    sil_phone_list.insert(len(sil_phone_list), 'x')
    for index in range(iter_len):
        (phone, state_phone,phone_index) = phone_state_list[index]
        # phone_index = phone_state_list_sil_phone[index]
        ll_phone = sil_phone_list[phone_index]
        l_phone = sil_phone_list[phone_index+1]
        c_phone = sil_phone_list[phone_index+2]
        r_phone = sil_phone_list[phone_index+3]
        rr_phone = sil_phone_list[phone_index+4]
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
            non_sil_index = sil_nonsil_map[phone_index]
            fw_ph_syl, bw_ph_syl,syl_index, cur_syl_num = pos_phone_in_syl(non_sil_index, syl_map)
            fw_syl_word,bw_syl_word,word_index, cur_word_num = pos_syl_in_word(syl_index,word_map)
            fw_word_utt = word_index + 1
            bw_word_utt = len(word_list) - word_index
            cur_tone = tone_list[syl_index]
            # position of phone in the syllabel forward and backward
            if syl_index != 0:
                prev_tone = tone_list[syl_index-1]
                pre_syl_num = get_phone_num_of_syl(syl_index-1, syl_map)
                if word_index != 0:
                    pre_word_num = get_syl_num_of_word(word_index-1, word_map)
                    prev_pos = pos_list[word_index-1]
            if syl_index != all_syl_len-1:
                next_tone = tone_list[syl_index+1]
                next_syl_num = get_phone_num_of_syl(syl_index+1, syl_map)
                if word_index != all_word_len-1:
                    next_word_num = get_syl_num_of_word(word_index+1, word_map)
                    next_pos = pos_list[word_index+1]
        else:
            if phone_index == 0:
                # at the beginning of sent
                _, _, next_syl_index, next_syl_num = pos_phone_in_syl(0, syl_map)
                next_tone = tone_list[next_syl_index]
                _,_, next_word_index, next_word_num = pos_syl_in_word(next_syl_index, word_map)
                next_pos = pos_list[next_word_index]
            elif phone_index == phone_iter_len - 1:
                _,_, pre_syl_index, pre_syl_num = pos_phone_in_syl(non_sil_len-1,syl_map)
                prev_tone = tone_list[pre_syl_index]
                _,_, pre_word_index, pre_word_num = pos_syl_in_word(pre_syl_index, word_map)
                prev_pos = pos_list[pre_word_index]
            else:
                non_sil_prev_index = sil_nonsil_map[phone_index-1]
                non_sil_next_index = sil_nonsil_map[phone_index+1]
                _,_,pre_syl_index,pre_syl_num = pos_phone_in_syl(non_sil_prev_index, syl_map)
                _,_,next_syl_index, next_syl_num = pos_phone_in_syl(non_sil_next_index,syl_map)
                _,_, pre_word_index, pre_word_num = pos_syl_in_word(pre_syl_index, word_map)
                prev_pos = pos_list[pre_word_index]
                _, _, next_word_index, next_word_num = pos_syl_in_word(next_syl_index, word_map)
                next_pos = pos_list[next_word_index]
        if training_data:
            label_phone_fid.write("{25} {26} {0}^{1}-{2}+{3}={4}@{5}_{6}/A:{7}_{8}/B:{9}_{10}!{11}_{12}/C:{13}+{14}/D:{15}-{16}/E:{17}&{18}^{19}_{20}/F:{21}_{22}/G:{23}-{24}{27}\n".format(ll_phone,l_phone,c_phone,r_phone,rr_phone,
                                                            str(fw_ph_syl),str(bw_ph_syl), str(prev_tone),str(pre_syl_num),str(cur_tone),
                                                            str(cur_syl_num), str(fw_syl_word), str(bw_syl_word),
                                                            str(next_tone), str(next_syl_num), prev_pos, pre_word_num, cur_pos, cur_word_num,
                                                            str(fw_word_utt), str(bw_word_utt),next_pos,next_word_num,str(all_syl_len),str(len(word_list)), start_time_list[index], end_time_list[index],state_phone))
        else:
            label_phone_fid.write("{0}^{1}-{2}+{3}={4}@{5}_{6}/A:{7}_{8}/B:{9}_{10}!{11}_{12}/C:{13}+{14}/D:{15}-{16}/E:{17}&{18}^{19}_{20}/F:{21}_{22}/G:{23}-{24}\n".format(ll_phone,l_phone,c_phone,r_phone,rr_phone,
                                                            str(fw_ph_syl),str(bw_ph_syl), str(prev_tone),str(pre_syl_num),str(cur_tone),
                                                            str(cur_syl_num), str(fw_syl_word), str(bw_syl_word),
                                                            str(next_tone), str(next_syl_num), prev_pos, pre_word_num, cur_pos, cur_word_num,
                                                            str(fw_word_utt), str(bw_word_utt),next_pos,next_word_num,str(all_syl_len),str(len(word_list))))
    label_phone_fid.close()
file_id_canto.close()