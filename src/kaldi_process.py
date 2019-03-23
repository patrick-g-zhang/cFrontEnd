# %matplotlib inline
from collections import OrderedDict
import re
import os
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import pdb
sns.set(color_codes=True)
class Kaldi_Process(object):
    def __init__(self, frame_shift=0.005):
        self.frame_shift = frame_shift
        project_name="sameSen300"
        self.ctm_file = "../exp/"+ project_name +"/kaldi/phone_ctm.ctm"
        self.len_file = "../exp/"+ project_name +"/kaldi/feats.lengths"
        self.phones = "../exp/"+ project_name +"/kaldi/phones.txt"
        self.pdf_file = "../exp/"+ project_name +"/kaldi/pdf.ali"
        self.no_tone_text = "../exp/"+ project_name +"/kaldi/no_tone_kaldi_text"
        self.write_dir = "../exp/"+ project_name + "/train/aud_ali_tri1/"

    def gen_praat_textgrid(self, write_dir='../exp/namestory/train/praat_001/'):
        """
            This is new version of extraction duration of each phone from alignment.txt
            using len file to valid the length of duration
            Parameters:
            ali_file: alignment file path
            len_file: feat-to-len file path
        """
        # read frames file number of each file
        if os.path.exists(write_dir):
            shutil.rmtree(write_dir)
        os.mkdir(write_dir)
        phone_frames = dict()
        utt2framenum = OrderedDict()
        len_f_id = open(self.len_file, 'r')
        len_lines = len_f_id.readlines()
        len_f_id.close()
        for len_line in len_lines:
            utt, frame_num = re.split('\s+', len_line.strip())
            utt2framenum[utt] = int(frame_num)

        syllabel_file_id = open(self.no_tone_text,'r')
        syllabel_lines = syllabel_file_id.readlines()
        syllabel_file_id.close()
        utt2_syllabels = dict()
        for s_line in syllabel_lines:
            all_contents = re.split("\s+",s_line.strip())
            utt = all_contents[0]
            utt2_syllabels[utt] = all_contents[1:]
        id2phone = OrderedDict()
        phone_fid = open(self.phones, 'r')
        phones_ids = phone_fid.readlines()
        phone_fid.close()
        for phone_map in phones_ids:
            phone, id = re.split('\s+', phone_map.strip())
            id2phone[id] = phone

        # the frames of each phones
        utt_phone_frame_map = OrderedDict()
        ctm_fid = open(self.ctm_file, 'r')
        ctm_lines = ctm_fid.readlines()
        ctm_fid.close()
        up_utt = ""
        all_frames_utt = 0
        for ctm_line in ctm_lines:
            utt_id, _, _, duration, phone_id = re.split('\s+', ctm_line.strip())
            frame_number = int(round(round(float(duration), 3) / self.frame_shift))
            if utt_id != up_utt:
                utt_phone_frame_map[utt_id] = []
                # a utterance starts
                if up_utt != "":
                    assert abs(utt2framenum[up_utt] == all_frames_utt) <= 2
                all_frames_utt = frame_number
            else:
                all_frames_utt = all_frames_utt + frame_number
            # utt_phone_frame_map[utt_id][id2phone[phone_id]] = frame_number
            utt_phone_frame_map[utt_id].append((id2phone[phone_id], frame_number))
            up_utt = utt_id
        # if not state:
        # phone level alignment
        pdf_id = open(self.pdf_file, 'r')
        pdf_lines = pdf_id.readlines()
        pdf_id.close()
        for pdf_line in pdf_lines:
            frame_state = re.split('\s+', pdf_line.strip())
            utt_id = frame_state[0]
            aud_file_path = write_dir + utt_id + '.TextGrid'
            f_id = open(aud_file_path, 'w')
            f_id.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n\n")
            frame_index = 0
            start_end_list=[]
            phone_list=[]
            syllabels = utt2_syllabels[utt_id]
            syl_start_end_list = []
            syl_index = 0
            for phone_frame in utt_phone_frame_map[utt_id]:
                flag = 'W'
                (phone, frame_num) = phone_frame
                # phone level alignment
                # print(phone)
                if '_' in phone:
                    # here, flag should be B E I or S
                    flag = phone[-1]
                    print(phone)
                    phone = phone[:-2]
                # count the duration distribution of each phone which cutted by force alignment
                # check key in dictionary
                if phone in phone_frames:
                    phone_frames[phone].append(frame_num)
                else:
                    phone_frames[phone] = []
                    phone_frames[phone].append(frame_num)

                start_time = round(frame_index * self.frame_shift, 3)
                end_time = round((frame_index + frame_num) * self.frame_shift, 3)
                if flag == "B":
                    syl_start_time = start_time
                if flag == "E":
                    syl_end_time = end_time
                    syl_start_end_list.append((syl_start_time, syl_end_time,syllabels[syl_index]))
                    syl_index = syl_index +1
                if flag == "S":
                    syl_start_end_list.append((start_time, end_time, syllabels[syl_index]))
                    syl_index = syl_index + 1
                if phone == "sil":
                    syl_start_end_list.append((start_time, end_time, '#'))
                start_end_list.append((start_time, end_time))
                phone_list.append(phone)
                frame_index = frame_index + frame_num
            # pdb.set_trace()
            # assert len(syllabels) == len(syl_start_end_list)
            total_time = start_end_list[-1][1]
            f_id.write("xmin = 0\nxmax = {}\n tiers? <exists>\nsize = 2\nitem []:\n".format(total_time))
            f_id.write("item []:\n\titem [1]:\n\t\tclass = \"IntervalTier\"\n\t\tname = \"phone\"\n\t\txmin = 0\n\t\txmax = {0}\n\t\tintervals: size = {1}\n".format(total_time, len(start_end_list)))
            for num, start_end_tuple in enumerate(start_end_list):
                index = num+1
                f_id.write("\t\tintervals [{0}]:\n\t\t\txmin = {1}\n\t\t\txmax = {2}\n\t\t\ttext = \"{3}\"\n".format(
                    str(index), str(start_end_tuple[0]),str(start_end_tuple[1]),phone_list[num]
                ))
            f_id.write("\titem [2]:\n\t\tclass = \"IntervalTier\"\n\t\tname = \"syllable\"\n\t\txmin = 0\n\t\txmax = {0}\n\t\tintervals: size = {1}\n".format(total_time, len(syl_start_end_list)))
            for num, syl_start_end_tuple in enumerate(syl_start_end_list):
                syl_index = num+1
                f_id.write("\t\tintervals [{0}]:\n\t\t\txmin = {1}\n\t\t\txmax = {2}\n\t\t\ttext = \"{3}\"\n".format(
                    str(syl_index), str(syl_start_end_tuple[0]), str(syl_start_end_tuple[1]), syl_start_end_tuple[2]
                ))
            f_id.close()

    def extract_dur_from_ali(self,
                             write_dir,
                             wirte_aud=True, draw_pdf=False,state=False):
        """
            This is new version of extraction duration of each phone from alignment.txt
            using len file to valid the length of duration
            Parameters:
            ali_file: alignment file path
            len_file: feat-to-len file path
        """
        # read frames file number of each file
        if os.path.exists(write_dir):
            shutil.rmtree(write_dir)
        os.mkdir(write_dir)
        phone_frames = dict()
        utt2framenum = OrderedDict()
        len_f_id = open(self.len_file, 'r')
        len_lines = len_f_id.readlines()
        len_f_id.close()
        for len_line in len_lines:
            utt, frame_num = re.split('\s+', len_line.strip())
            utt2framenum[utt] = int(frame_num)

        id2phone = OrderedDict()
        phone_fid = open(self.phones, 'r')
        phones_ids = phone_fid.readlines()
        phone_fid.close()
        for phone_map in phones_ids:
            phone, id = re.split('\s+', phone_map.strip())
            id2phone[id] = phone

        # the frames of each phones
        utt_phone_frame_map = OrderedDict()
        ctm_fid = open(self.ctm_file, 'r')
        ctm_lines = ctm_fid.readlines()
        ctm_fid.close()
        up_utt = ""
        all_frames_utt = 0
        for ctm_line in ctm_lines:
            utt_id, _, _, duration, phone_id = re.split('\s+', ctm_line.strip())
            frame_number = int(round(round(float(duration), 3) / self.frame_shift))
            if utt_id != up_utt:
                utt_phone_frame_map[utt_id] = []
                # a utterance starts
                if up_utt != "":
                    assert abs(utt2framenum[up_utt] == all_frames_utt) <= 2
                all_frames_utt = frame_number
            else:
                all_frames_utt = all_frames_utt + frame_number
            # utt_phone_frame_map[utt_id][id2phone[phone_id]] = frame_number
            utt_phone_frame_map[utt_id].append((id2phone[phone_id], frame_number))
            up_utt = utt_id
        # if not state:
        # phone level alignment
        pdf_id = open(self.pdf_file, 'r')
        pdf_lines = pdf_id.readlines()
        pdf_id.close()
        for pdf_line in pdf_lines:
            frame_state = re.split('\s+', pdf_line.strip())
            utt_id = frame_state[0]
            aud_file_path = write_dir + utt_id + '.txt'
            f_id = open(aud_file_path, 'w')
            state_frames_list = frame_state[1:]
            frame_index = 0
            for phone_frame in utt_phone_frame_map[utt_id]:
                (phone, frame_num) = phone_frame
                if not state:
                    # phone level alignment
                    if '_' in phone:
                        phone = phone[:-2]
                    # count the duration distribution of each phone which cutted by force alignment 
                    # check key in dictionary
                    if phone in phone_frames:
                        phone_frames[phone].append(frame_num)
                    else:
                        phone_frames[phone] = []
                        phone_frames[phone].append(frame_num)
                    start_time = round(frame_index * self.frame_shift, 3)
                    end_time = round((frame_index + frame_num) * self.frame_shift, 3)
                    out_str = "{0}\t{1}\t{2}\n".format(str(start_time), str(end_time), phone)
                    f_id.write(out_str)
        # if not os.path.exists('out/'):
            # os.make makedirs('out/')
        #iterate the dictionary
                else:
                    # state level alignment
                    state_freq = OrderedDict()
                    for num in range(frame_num):
                        if state_frames_list[frame_index + num] in state_freq:
                            state_freq[state_frames_list[frame_index + num]] = state_freq[state_frames_list[
                                frame_index + num]] + 1
                        else:
                            state_freq[state_frames_list[frame_index + num]] = 1
                    state_index = frame_index
                    state_pos = 2
                    for key, value in state_freq.items():
                        start_time = round(state_index * self.frame_shift, 3)
                        end_time = round((state_index + value) * self.frame_shift, 3)
                        state_index = state_index + value
                        out_str = '{0} {1} {2} [{3}]\n'.format(start_time, end_time, phone, str(state_pos))
                        f_id.write(out_str)
                        state_pos = state_pos + 1
                frame_index = frame_index + frame_num
            f_id.close()
        # for key, value in phone_frames.items():
            # print(value)
            # sns.distplot(value)
            # plt.show()


if __name__ == '__main__':
    kp = Kaldi_Process()
    # kp.gen_praat_textgrid()
    kp.extract_dur_from_ali(write_dir=kp.write_dir)