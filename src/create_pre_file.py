"""
This python is created for preparing corresponding force alignment with kaldi
it consists of:
providing wav.scp and utt2spk files
"""
import jieba
import logging
import thulac
import glob
import os
thu1 = thulac.thulac(seg_only=True)
logging.basicConfig(level=logging.DEBUG)

def get_word_dict(dict_path='/home/gyzhang/PycharmProjects/kingtts/feature_extract/nc/tts_dict/cn_word.dict'):
    dict={}
    fo = open(dict_path,'r')
    lines = fo.readlines()
    fo.close()
    for line in lines:
        w = line.strip().split(' ')
        dict[w[0]]=w[1:]
    return dict


def get_char_dict(dict_path='/home/gyzhang/PycharmProjects/kingtts/feature_extract/nc/tts_dict/cn_char.dict'):
    dict={}
    fo = open(dict_path,'r')
    lines = fo.readlines()
    fo.close()
    for line in lines:
        w = line.strip().split(' ')
        dict[w[0]]=w[1:]
    return dict

def create_wav(wav_folder,wav_file):
    """
    Create wav.scp file for kaldi project

    Arguments:
    wav_folder -- folder with wave files with wave files like '10001.wav'
    wav_file -- our scp file
    
    """
    f_out = open(wav_file, 'w')

    for file in glob.glob(wav_folder+'/*.wav'):
        base=os.path.basename(file).split('.')[0]

        # write to scp file
        f_out.write('tts'+base+'\t'+file+'\n')

def create_scp_cusent(wav_folder, wav_file, utt2spk, ori_text, target_text):
    """
    create wav.scp and utt2spk and text file with cusent corpus for kaldi project

    """
    f_out = open(wav_file, 'w')
    u_out = open(utt2spk,'w')
    texts = open(ori_text,'r').readlines()
    utt_id_text = {}
    text_out = open(target_text,'w')
    for text in texts:
        text_list = text.strip().split(' ')
        utt_id = text_list[0]
        text_bank = ' '.join(text_list[1:])
        utt_id_text[utt_id] = text_bank

    for file in glob.glob(wav_folder+'/*f/*.wav'):
        base = os.path.basename(file).split('.')[0]
        dir_name = os.path.dirname(file).split('/')[-1]
        utt_id = dir_name+'_'+base
        # wav path with sph pip
        f_out.write(utt_id+'\t'+'/home/gyzhang/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav '+file +' |'+'\n')
        u_out.write(utt_id+'\t'+dir_name+'\n')
        text_out.write(utt_id+' '+utt_id_text[utt_id]+'\n')
    f_out.close()
    u_out.close()
    text_out.close()






def create_utt_spk(wav_folder, utt2spk):
    """
    Create utt2spk file for kaldi project

    Arguments:
    wav_folder -- folder with wave files

    """

    f_out = open(utt2spk, 'w')

    for file in glob.glob(wav_folder+'/*f/*.wav'):
        base = os.path.basename(file).split('.')[0]
        dir_name = os.path.dir_name(file).split('/')[-1]
        utt_id = dir_name+'_'+base
        f_out.write(utt_id+'\t'+file+'\n')


def create_text(input_text, text_file):
    """
    Create transcript text for kaldi with original chinese character as input

    Arguments:
    input_text -- our input text with Chinese character
    text_file -- text file of our output with pinyin

    Returns:

    """

    # get word and character dictionary
    word_dict = get_word_dict()
    char_dict = get_char_dict()

    # the file will be written in pinyin by one line
    f_output = open(text_file, 'w')

    f_input = open(input_text)
    trans = f_input.readlines()
    f_input.close()

    for line in trans:
        num, text = line.strip().split('\t')
        text = text.replace('#', '')

        # cut word with jieba
        word_seg = jieba.cut(text, HMM=False)

        # phone_list is pinyin list
        phone_list = []
        for word in word_seg:
            try:
                phone_of_word = word_dict[word]
                phone_list.extend(phone_of_word)

            # jieba always give big chunk and need to cut further
            except KeyError:
                t_words = thu1.cut(word)
                for e_word in t_words:
                    p_word, _ = e_word
                    try:
                        phone_of_word = word_dict[p_word]
                        phone_list.extend(phone_of_word)
                    except KeyError:
                        for char in p_word:
                            phone_list.extend(char_dict[char])
        tran_text = 'tts'+num + '\t' + ' '.join(phone_list) + '\n'
        f_output.write(tran_text)
    f_output.close()


def create_align_aud(ctm_file,phone_dict):
    """
    Create label file for audacity

    Arguments:
    :param ctm_file:
    :param phone_dict:
    :return:
    """
    f_phone = open(phone_dict, 'r')
    phones = f_phone.readlines()
    f_phone.close()

    phone_map = {}
    for phone in phones:
        phone_ele, phone_id = phone.strip().split(' ')
        phone_map[phone_id]=phone_ele

    f_ctm = open(ctm_file, 'r')
    ctms = f_ctm.readlines()
    f_ctm.close()
    if not os.path.exists('./align_for_aud'):
        os.mkdir('./align_for_aud')
    cur_utt_id='tts100001'
    f_w = open('./align_for_aud/'+cur_utt_id+'.txt', 'w')
    for ctm_line in ctms:
        utt_id, _, start_time, dur_time, phone_id =ctm_line.strip().split(' ')
        if cur_utt_id != utt_id:
            f_w.close()
            cur_utt_id = utt_id
            f_w = open('./align_for_aud/'+cur_utt_id+'.txt','w')
        start_time = float(start_time)
        end_time = round(start_time + float(dur_time), 2)
        phone_ele = phone_map[phone_id]
        ali = str(start_time) + '\t' + str(end_time) + '\t' + phone_ele + '\n'
        f_w.write(ali)

    f_w.close()
if __name__ == '__main__':
    # create_wav(wav_folder='/home/gyzhang/TARGET_WAV', wav_file='/home/gyzhang/PycharmProjects/kingtts/align_phone/kingtts/data/train/wav.scp')
    # create_text(input_text='/home/gyzhang/texts.txt', text_file='/home/gyzhang/PycharmProjects/kingtts/align_phone/kingtts/data/train/text')
    # create_utt_spk(wav_folder='/home/gyzhang/TARGET_WAV', utt2spk='/home/gyzhang/PycharmProjects/kingtts/align_phone/kingtts/data/train/utt2spk')
    # create_align_aud(ctm_file="/home/gyzhang/PycharmProjects/kingtts/align_phone/align_tts/exp/tri2a_ali/king_tts.ctm", phone_dict="/home/gyzhang/PycharmProjects/kingtts/align_phone/align_tts/data/lang/phones.txt")
    cu_sent_wav_folder = '/home/gyzhang/speech_database/CUSENT/CUSENT_wav/train'
    cu_sent_wav_file = '../exp/cu_sent/kaldi/wav.scp' 
    cu_sent_u2s_file = '../exp/cu_sent/kaldi/utt2spk'
    cu_sent_ori_text = '/home/gyzhang/speech_database/CUSENT/train.text'
    cu_sent_target_text = '../exp/cu_sent/kaldi/no_tone_kaldi_text'
    create_scp_cusent(cu_sent_wav_folder, cu_sent_wav_file,cu_sent_u2s_file,cu_sent_ori_text,cu_sent_target_text)