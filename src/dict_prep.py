"""
	This program is for dict preparation
"""
import re 
with open("/home/gyzhang/projects/cFrontEnd/data/dicts/dict_name.dict","r") as f_dict:
	dict_lines = f_dict.readlines()
f_dict = open("/home/gyzhang/projects/cFrontEnd/data/dicts/dict_name.dict","w")
for dict_line in dict_lines:
	if dict_line == "\n":
		continue
	_,_, pos = re.split('\s+', dict_line.strip())
	if 'x' not in pos:
		f_dict.write(dict_line)
f_dict.close()

