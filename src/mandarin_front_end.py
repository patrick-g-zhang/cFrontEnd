import os
import glob
import logging
from pypinyin import pinyin,Style
from jieba import posseg
import re
from collections import OrderedDict
from linguistic_dict import Linguistic_DICT
ld = Linguistic_DICT()
class MFrontEnd(object):
    def __init__(self,text_file, file_id_path = './gen/file_id_list_kingtts.scp'):
        self.text_file = text_file
        self.file_id_path = file_id_path
        self.label_phone_align = './label_phone_align/'

        logger = logging.getLogger('creat front end for chinese')

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


    def create_utt_spk(wav_folder="/home/gyzhang/speech_database/1001", utt2spk="./gen/utt2spk"):
        """
        Create utt2spk file for kaldi project

        Arguments:
        wav_folder -- folder with wave files

        """

        f_out = open(utt2spk, 'w')

        for file in glob.glob(wav_folder + '/*.wav'):
            base = os.path.basename(file).split('.')[0]
            f_out.write(base + '\t' + 'tts' + '\n')

    def get_word_pos_list(self, raw_text):
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
        for word, flag in seg_list:
            # remove the punctuation
            if word in ['。', '？', '：', '、', '…','；','，']:
                continue
            word_list.append(word)
            pos_list.append(flag)
        return word_list, pos_list

    def get_word_phone_list(self, word_list):
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
        ipa_dict = ld.get_lexicon_dict(lexicon_path='./dicts/ipa_m.dict')
        for word in word_list:
            word_pinyin_initials = pinyin(word, style=Style.INITIALS)
            word_pinyin_finals = pinyin(word, style=Style.FINALS)
            word_pinyin_tone = pinyin(word,style=Style.TONE3)
            assert len(word_pinyin_initials) == len(word_pinyin_finals)
            for initial, final, all_tone in zip(word_pinyin_initials, word_pinyin_finals, word_pinyin_tone):
                char_phone_list = []
                assert len(initial) == 1
                assert  len(final) == 1
                if not initial[0] == "":
                    ipa_initial = ipa_dict[initial[0]]
                    char_phone_list.extend(ipa_initial)
                if not final[0] == "":
                    try:
                        ipa_final = ipa_dict[final[0]]
                        char_phone_list.extend(ipa_final)
                    except KeyError:
                        print(word)
                        exit(0)
                else:
                    print("error, the final is empty")
                    exit(0)
                phone_list.extend(char_phone_list)
                for my_phone in char_phone_list:
                    syl_map[phone_index] = char_index
                    phone_index = phone_index + 1
                if all_tone[0][-1].isdigit():
                    tone_list.append(all_tone[0][-1])
                else:
                    tone_list.append('5')
                word_map[char_index] = word_index
                char_index = char_index + 1
            word_index = word_index + 1
        #     logging.debug("phone_list:" + ' '.join(phone_list))
        return phone_list, tone_list, syl_map, word_map

    def pre_process(self, raw_text):
        """
        input raw text
        and output line
        """
        # remove the space or other symbols
        word_lists = re.split(r'\s+', raw_text.strip())
        if len(word_lists) < 2:
            print(word_lists)
            exit(1)
        sent_index = word_lists[0]
        word_lists = word_lists[1]
        word_lists = re.split(r'#', word_lists)
        sent_content = ''.join(word_lists)
        return sent_index, sent_content


    def create_file_for_kaldi(self, ):
        with open(self.text_file, 'r') as fid:
            textlines = fid.readlines()
        kaldi_text_no_tone = './gen/no_tone_kaldi_text'
        kfid = open(kaldi_text_no_tone, 'w')
        for text_line in textlines:
            sent_index, sent_content = self.pre_process(text_line)
            word_list, pos_list = self.get_word_pos_list(sent_content)
            # phone_list, tone_list, syl_map, word_map = self.get_word_phone_list(word_list)
            phone_list, tone_list, syl_map, word_map = self.get_word_phone_list(word_list)
            non_tone_line = sent_index + ' ' + ' '.join(phone_list)+'\n'
            kfid.write(non_tone_line)

    def create_phone_labels(self, training_data):
        with open(self.text_file, 'r') as fid:
            textlines = fid.readlines()
        file_id_canto = open(self.file_id_path, 'w')
        for text_line in textlines:
            sent_index, sent_content = self.pre_process(text_line)
            ## extract word and pos tagging
            word_list, pos_list = self.get_word_pos_list(sent_content)
            phone_list, tone_list, syl_map, word_map = self.get_word_phone_list(word_list)
            file_id_canto.write(sent_index + '\n')
            sil_phone_list = []
            start_time_list = []
            end_time_list = []
            if training_data:
                # extract alignment file
                ali_file = './align_phone/cup_phone_ali/' + sent_index + '.txt'
                with open(ali_file, 'r') as fid:
                    ali_file_lines = fid.readlines()
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
            label_phone_path = self.label_phone_align + sent_index + '.lab'
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
    mf = MFrontEnd(text_file='./inp/texts.txt',file_id_path = './gen/file_id_list_kingtts.scp')
    # mf.create_wav_file()
    # mf.create_file_for_kaldi()
    mf.create_phone_labels(True)
