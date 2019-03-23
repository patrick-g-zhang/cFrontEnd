import pdb
import re
import pycantonese as pc
# import pycantonese as pc
# corpus = pc.hkcancor()
import jieba
from linguistic_dict import Linguistic_DICT
jieba.set_dictionary('./dicts/freq_merged.dict')
mld = Linguistic_DICT()
word_dict = mld.get_phone_dict(dict_file='./dicts/word2jyut.lex')
word_dict_fid = open('./word2jyut.lex','a')
# kaldi_text_no_tone = './no_tone_kaldi_text'
# kfid = open(kaldi_text_no_tone, 'w')
with open('./raw_text.txt') as fid:
    textlines = fid.readlines()

for line in textlines:
    # remove the space and
    word_lists = re.split(r'\s+', line.strip())
    sent_index = word_lists[0]
    word_lists = word_lists[1:]
    word2str = ''.join(word_lists)
    seg_list = jieba.cut(word2str, HMM=False)  # 默认是精确模式
    # will store the phone list of
    non_tone_line_phones=[]
    for word in seg_list:
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
            word_dict_fid.write('{0} {1}\n'.format(word, word_phone))
        # try:
            # jp = pc.parse_jyutping(word_phone)
        # except Exception as e:
            # pdb.set_trace()
        # for char_phone in jp:
            # char_phone = list(char_phone)
            # char_phone = [e_phone for e_phone in char_phone if e_phone != '']
            # assert char_phone[-1].isdigit()
            # non_tone_line_phones.append(''.join(char_phone[:-1]))
    # non_tone_line = sent_index + ' ' + ' '.join(non_tone_line_phones)+'\n'
    # kfid.write(non_tone_line)
# kfid.close()
word_dict_fid.close()


                    # foov = open("./oovchar.txt", 'a', encoding='utf-8')
                    # foov.write(char + ' ' + phone + '\n')
                    # foov.close()