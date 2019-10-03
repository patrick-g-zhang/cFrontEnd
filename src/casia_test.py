from corpus.casia import Casia
from FrontEnd import FrontEnd
import re
import os
from praatio import tgio
import pdb
from collections import OrderedDict
from linguistic_dict import Linguistic_DICT
ld = Linguistic_DICT()


class casia_front(FrontEnd):
    """docstring for ClassName"""

    def __init__(self, ):
        super(FrontEnd, self).__init__()
        # here we will only do with liuchanhg
        self.alignment_file_dir = "/home/gyzhang/Documents/aligned_liuchanhg"
        self.label_phone_align = "/home/gyzhang/projects/cFrontEnd/exp/casia/train/label_phone_align"
        os.makedirs(self.label_phone_align, exist_ok=True)
        self.file_id_path = "/home/gyzhang/projects/cFrontEnd/exp/casia/train/file_id_list.scp"
        self.chinese_dict = ld.get_lexicon_dict(
            lexicon_path="../data/dicts/mandarin_syl2phoneme.txt")

      #  ss = Casia()
      #  ss.process_same_text()

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

    def check_phone_consistent(self, ):
        """
            make sure phone same in 
        """
        with open("../exp/casia/train.txt") as fid:
            textlines = fid.readlines()
        for text_line in textlines:
            wav_path, sent_content = re.split("\|", text_line.strip())
            sent_index = re.split("\.", os.path.basename(wav_path))[0]
            spk = re.split('\-', sent_index)[0]
            if spk != "liuchanhg":
                continue
            # corresponding label file

            # extract word and pos tagging
            if "483" in sent_index:
                continue
            word_list, pos_list = self.get_word_pos_list(
                sent_content.strip(), "jieba")
            phone_list, tone_list, syl_map, word_map, _ = self.get_word_phone_list(
                word_list, True)
            phone_list_2 = []
            # extract alignment file
            alignment_file_path = os.path.join(
                self.alignment_file_dir, sent_index + ".TextGrid")
            tg = tgio.openTextgrid(alignment_file_path)
            wordTier = tg.tierDict['phones']
            for start, stop, label in wordTier.entryList:
                phone = label
                if phone == "sp" or phone == "sil":
                    continue
                phone_list_2.append(phone)
            assert phone_list_2 == phone_list
            # replace sp to sil

    def create_phone_labels(self, training_data):
        with open("../exp/casia/train.txt") as fid:
            textlines = fid.readlines()

        file_id_canto = open(self.file_id_path, 'w')
        for text_line in textlines:
            wav_path, sent_content = re.split("\|", text_line.strip())
            sent_index = re.split("\.", os.path.basename(wav_path))[0]
            spk = re.split('\-', sent_index)[0]
            if spk != "liuchanhg":
                continue
            if "483" in sent_index:
                continue
            # corresponding label file

            # extract word and pos tagging
            word_list, pos_list = self.get_word_pos_list(
                sent_content.strip(), "jieba")
            phone_list, tone_list, syl_map, word_map, _ = self.get_word_phone_list(
                word_list, True)
            file_id_canto.write(sent_index + '\n')
            sil_phone_list = []
            start_time_list = []
            end_time_list = []
            if training_data:
                # extract alignment file
                alignment_file_path = os.path.join(
                    self.alignment_file_dir, sent_index + ".TextGrid")
                tg = tgio.openTextgrid(alignment_file_path)
                wordTier = tg.tierDict['phones']
                for start, stop, label in wordTier.entryList:
                    start_time = str(int(start * 10000000))
                    end_time = str(int(stop * 10000000))
                    phone = label
                    sil_phone_list.append(phone)
                    start_time_list.append(start_time)
                    end_time_list.append(end_time)
                    # replace sp to sil
                replace_sil_phone_list = []
                for my_sil_phone in sil_phone_list:
                    if my_sil_phone == "sp":
                        replace_sil_phone_list.append("sil")
                    else:
                        replace_sil_phone_list.append(my_sil_phone)
                sil_phone_list = replace_sil_phone_list

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
            label_phone_path = os.path.join(
                self.label_phone_align, sent_index + '.lab')
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
                    fw_ph_syl, bw_ph_syl, syl_index, cur_syl_num = self.pos_phone_in_syl(
                        non_sil_index, syl_map)
                    fw_syl_word, bw_syl_word, word_index, cur_word_num = self.pos_syl_in_word(
                        syl_index, word_map)
                    fw_word_utt = word_index + 1
                    bw_word_utt = len(word_list) - word_index
                    cur_tone = tone_list[syl_index]
                    # position of phone in the syllabel forward and backward
                    if syl_index != 0:
                        prev_tone = tone_list[syl_index - 1]
                        pre_syl_num = self.get_phone_num_of_syl(
                            syl_index - 1, syl_map)
                        if word_index != 0:
                            pre_word_num = self.get_syl_num_of_word(
                                word_index - 1, word_map)
                            prev_pos = pos_list[word_index - 1]
                    if syl_index != all_syl_len - 1:
                        next_tone = tone_list[syl_index + 1]
                        next_syl_num = self.get_phone_num_of_syl(
                            syl_index + 1, syl_map)
                        if word_index != all_word_len - 1:
                            next_word_num = self.get_syl_num_of_word(
                                word_index + 1, word_map)
                            next_pos = pos_list[word_index + 1]
                    cur_pos = pos_list[word_index]
                else:
                    if index == 0:
                        # at the beginning of sent
                        _, _, next_syl_index, next_syl_num = self.pos_phone_in_syl(
                            0, syl_map)
                        next_tone = tone_list[next_syl_index]
                        _, _, next_word_index, next_word_num = self.pos_syl_in_word(
                            next_syl_index, word_map)
                        next_pos = pos_list[next_word_index]
                    elif index == iter_len - 1:
                        _, _, pre_syl_index, pre_syl_num = self.pos_phone_in_syl(
                            non_sil_len - 1, syl_map)
                        prev_tone = tone_list[pre_syl_index]
                        _, _, pre_word_index, pre_word_num = self.pos_syl_in_word(
                            pre_syl_index, word_map)
                        prev_pos = pos_list[pre_word_index]
                    else:
                        non_sil_prev_index = sil_nonsil_map[index - 1]
                        non_sil_next_index = sil_nonsil_map[index + 1]
                        _, _, pre_syl_index, pre_syl_num = self.pos_phone_in_syl(
                            non_sil_prev_index, syl_map)
                        _, _, next_syl_index, next_syl_num = self.pos_phone_in_syl(
                            non_sil_next_index, syl_map)
                        _, _, pre_word_index, pre_word_num = self.pos_syl_in_word(
                            pre_syl_index, word_map)
                        prev_pos = pos_list[pre_word_index]
                        _, _, next_word_index, next_word_num = self.pos_syl_in_word(
                            next_syl_index, word_map)
                        next_pos = pos_list[next_word_index]
                if training_data:
                    label_phone_fid.write(
                        "{25} {26} {0}^{1}-{2}+{3}={4}@{5}_{6}/A:{7}_{8}/B:{9}_{10}!{11}_{12}/C:{13}+{14}/D:{15}-{16}/E:{17}&{18}^{19}_{20}/F:{21}_{22}/G:{23}-{24}\n".format(
                            ll_phone, l_phone, c_phone, r_phone, rr_phone,
                            str(fw_ph_syl), str(bw_ph_syl), str(
                                prev_tone), str(pre_syl_num), str(cur_tone),
                            str(cur_syl_num), str(
                                fw_syl_word), str(bw_syl_word),
                            str(next_tone), str(
                                next_syl_num), prev_pos, pre_word_num, cur_pos, cur_word_num,
                            str(fw_word_utt), str(
                                bw_word_utt), next_pos, next_word_num, str(all_syl_len),
                            str(len(word_list)), start_time_list[index], end_time_list[index]))
                else:
                    label_phone_fid.write(
                        "{0}^{1}-{2}+{3}={4}@{5}_{6}/A:{7}_{8}/B:{9}_{10}!{11}_{12}/C:{13}+{14}/D:{15}-{16}/E:{17}&{18}^{19}_{20}/F:{21}_{22}/G:{23}-{24}\n".format(
                            ll_phone, l_phone, c_phone, r_phone, rr_phone,
                            str(fw_ph_syl), str(bw_ph_syl), str(
                                prev_tone), str(pre_syl_num), str(cur_tone),
                            str(cur_syl_num), str(
                                fw_syl_word), str(bw_syl_word),
                            str(next_tone), str(
                                next_syl_num), prev_pos, pre_word_num, cur_pos, cur_word_num,
                            str(fw_word_utt), str(
                                bw_word_utt), next_pos, next_word_num, str(all_syl_len),
                            str(len(word_list))))
            label_phone_fid.close()
        file_id_canto.close()


cf = casia_front()
cf.create_phone_labels(True)
# cf.process_same_text()
