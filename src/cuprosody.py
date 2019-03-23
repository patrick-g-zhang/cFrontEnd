# coding: utf-8
"""
usage: preprocess the cuprosody text to standard format
"""
import re
text_file = "../exp/cuprosody/train/cuprosody.txt"
recon_text_file = "../exp/cuprosody/train/cn_text.txt"
with open(text_file, 'r') as fid:
	text_lines = fid.readlines()
rfid = open(recon_text_file,'w')
for line in text_lines:
    word_list = re.split('\s+', line.strip())
    sent_index = word_list[0]
    sent_content = ''.join(word_list[1:])
    # combine sent index and content as whole sentence
    whole_sent = sent_index + ' ' +sent_content+'\n'
    rfid.write(whole_sent)



