# This is program is about create the dict for text
import pycantonese as pc
import re
class Linguistic_DICT(object):
    """
        Read different kinds of dict
    """
    def get_POS_dict(self, dict_file):
        _dict = {}
        try:
            with open(dict_file, 'r') as f:
                for line in f:
                    (num, pos) = line.strip().split(' ')
                    _dict[pos] = num
        except:
            print("Error: Fail to open %s" % dict_file)
        return _dict

    def get_phone_dict(self, dict_file):
        # pdb.set_trace()
        # pdb.set_trace()
        _dict = {}
        try:
            with open(dict_file, 'r') as f:
                for line in f:
                    (phone, num) = line.strip().split(' ')
                    _dict[phone] = num
        except:
            print("Error: Fail to open %s" % dict_file)
        return _dict

    def search_single_char(self, word_dict, m_char):
        # search single char in word dict
        for key, value in word_dict.items():
            if m_char in key:
                m_index = list(key).index(m_char)
                jp = pc.parse_jyutping(value)
                jp_index = ''.join(list(jp[m_index]))
                return jp_index
    def get_lexicon_dict(self, lexicon_path):
        with open(lexicon_path, 'r') as fid:
            lex_lines = fid.readlines()
        _dict = {}
        for tline in lex_lines:
            tline = tline.strip()
            lex_list = re.split('\s+',tline)
            _dict[lex_list[0]] = lex_list[1:]
        return _dict