import glob
import os
import re
from shutil import copyfile
import logging
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import librosa
# import pysndfile
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def hasNumbers(inputString):
    """
    determine whether this string contains any digital number
    :param inputString:
    :return:
    """
    return any(char.isdigit() for char in inputString)

def contains_letters(inputString):
    regexp = re.compile(r"[a-z]|[A-Z]")
    return regexp.search(inputString)

def text2entries(text_file, txt_dir):
    """
        You can use this code create entry for integrated text file.
        You can modify this file for your own need.
    """
    with open(text_file,'r') as f_txt:
        lines = f_txt.readlines()

    for each_line in lines:
        all_contents = re.split('\s+', each_line.strip())
        utt_id = all_contents[0]
        content = ' '.join(all_contents[1:])
        lab_path = os.path.join(txt_dir, utt_id+'.lab')
        with open(lab_path,'w') as f_lab:
            f_lab.write(content+'\n')

def text2list(text_file):
    """
    input a text file, return all element in this file as a list
    text file example:
    a
    b
    c
    return list:
    [a,b,c]
    :param text_file:
    :return:
    """
    with open(text_file,'r') as f_txt:
        lines = f_txt.readlines()
    text_list = []
    for each_line in lines:
        text_list.append(each_line.strip())
    return text_list


def get_duration_of_speech_corpus(wav_dir):
    """
    Given the file folder of speech, return the
    :param wav_dir:
    :return:
    """
    total_duration = 0
    for wav_file in glob.glob(wav_dir+'/*.wav'):
        sr, y = read(wav_file)
        duration = y.shape[0]/sr
        total_duration += duration
    return total_duration

def gen_std_wav(ori_wav_dir,target_wav_dir):
    logger = logging.getLogger("get std wav")
    logger.setLevel(level=logging.INFO)
    """
        generate standard waveform and scp file
        Args:
        ori_wav_dir: original waveform dir
    """
    text_name2 = target_wav_dir+'/text_name2.txt'
    fid_text = open(text_name2, 'w')
    txt_files = glob.glob(ori_wav_dir+'/**/*.txt',recursive=True)
    for txt_file in txt_files:
        wav_file = re.sub(r'(.*)txt', r'\1wav', txt_file)
        m = re.match(r'^.*(s[\d]{1,})\/out_([\d]{1,}).*$', wav_file)
        story_index = m.group(1)
        sent_index = m.group(2)
        target_wav_path = target_wav_dir+'/'+story_index+'_'+sent_index+'.wav'
        copyfile(wav_file, target_wav_path)
        with open(txt_file,'r') as ifid:
            text_line = ifid.read().strip()
            logger.info("out line {}".format(text_line))
        fid_text.write(story_index+'_'+sent_index+' '+text_line+'\n')
    fid_text.close()

def clean_move(clean_txt_path,wav_dir,target_wav_dir):
    """
    we create a clean sub-corpus for alignment
    :return:
    """
    # wav_scp_list = []
    with open(clean_txt_path,'r') as fid:
        txt_lines = fid.readlines()
    for txt_line in txt_lines:
        sent_name = re.split('\s+', txt_line.strip())[0]
        wav_path = wav_dir+'/'+sent_name+'.wav'
        target_wav_path = target_wav_dir+'/'+sent_name+'.wav'
        wav, fs = librosa.load(wav_path, sr=None)
        if len(wav.shape) == 2:
            wav = librosa.to_mono(wav)
        if fs != 16000:
            wav = librosa.resample(wav, fs, 16000)
        # librosa.output.write_wav(target_wav_path, wav, 16000)
        pysndfile.sndio.write(target_wav_path, wav, rate=16000, format='wav', enc='pcm16')


def valid_alignment(wav_dir, ali_dir,story_index,target_dir):
    """
    give a great interface for visualize alignment effect of namestory
    we need to concatenate two wav files and its label files respectively
    for our tools (for instance audacity) to observe all wave and alignment in one story
    Args:
    Suppose the wav name and label name are total equally and each time the output is a story
    Args:
        ali_dir:the folder of alignment result
        story_index: the target story index
    """
    all_wav = np.array([],dtype=np.int16)
    all_ali_txt = []
    for ali_txt_path in glob.glob(ali_dir+'/*.txt'):
        m = re.match(r'^.*s([\d]{1,})_.*$', ali_txt_path)
        m_story_index = m.group(1)
        if m_story_index == story_index:
            utt_index = re.match(r'^.*_([\d]{1,}.*?).txt$', ali_txt_path).group(1)
            with open(ali_txt_path,'r') as fid:
                ali_txts = fid.readlines()
            for ali_lin in ali_txts:
                start, end, phone = re.split('\s+', ali_lin.strip())
                start = float(start)
                end = float(end)
                duration = end - start
                if all_ali_txt == []:
                    all_ali_txt.append(ali_lin.strip())
                else:
                    _, pre_end, _ = re.split('\s+', all_ali_txt[-1])
                    pre_end = float(pre_end)
                    cur_start = round(pre_end, 3)
                    cur_end = round(cur_start+duration, 3)
                    phone = phone+utt_index
                    cur_str = "{}\t{}\t{}".format(str(cur_start),str(cur_end),phone)
                    all_ali_txt.append(cur_str)
            base_name = os.path.basename(ali_txt_path)
            r_wav_path = base_name.replace('txt','wav',1)
            full_wav_path = os.path.join(wav_dir,r_wav_path)
            fs, wave = read(full_wav_path)
            all_wav = np.concatenate((all_wav, wave[0:int(end*16000)]))
    with open(target_dir + '/s' + story_index + '.txt', 'w') as fid:
        for ali_lin in all_ali_txt:
            fid.write(ali_lin + '\n')
    write(target_dir+'/s'+story_index+'.wav',16000 ,all_wav)

if __name__ == '__main__':
    text2entries('../exp/namestory/kaldi/no_tone_kaldi_text','/home/gyzhang/speech_database/storyTelling/split_760')