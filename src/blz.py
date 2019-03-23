from pypinyin import pinyin,Style, style
from linguistic_dict import Linguistic_DICT
ld = Linguistic_DICT()
import urllib, sys
import ssl
import time
import re
from aip import AipNlp
APP_ID= '15716974'
API_KEY='QZ4ee5tvyLrKCZ5FZib1eDFN'
SECRET_KEY='Zhe7VieQlGeSvGoPbeHdfLLeDF78KOYO'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
text_file_path = "/Users/patrickzhang/Dropbox/blizzard_release_2019_v1/text.txt"
ipa_dict = ld.get_lexicon_dict(lexicon_path='../data/dicts/ipa_m.dict')
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
with open(text_file_path, 'r') as fid:
    text_lines = fid.readlines()
for num, line in enumerate(text_lines):
    index = line[0:7]
    text_line = line.strip()[7:]
    # text_line="最近张潇雨老师跟我说，他在香港股市抄底中兴通讯公司，一个月就赚了百分之六十。这本来可喜可贺。不过，张潇雨反思说，其实这件事不该做。为啥？因为这违反了他自己一直主张的，也是巴菲特主张的，“不懂的东西就不要做”的原则。事实上，在这次中兴事件中，别说中兴公司了，就是美国政府，甚至特朗普本人，也未必能够操控事件的发展。能赚到钱，完全是运气。可是好运气会给一个长期投资者造成很多假象。比如，难免就会认为“以后应该尽量趁乱抄底”“判断宏观经济、政治形势很容易”“一个月赚百分之六十很正常”等等。和赚到一小笔钱的收益相比，这种心智模式带来的长期风险更可怕。所以张潇雨总结了一句话：用正确的方法得到一个失败的结果，其实要远远好于用错误的方法得到一个胜利的结果。"
    # print(index, text_line)
    """ 调用词法分析 """
    if num%5==4:
        time.sleep(1)
    return_list = client.lexer(text_line)['items']
    pinyin_line = []
    lexicon_syl = []
    for item in return_list:
        # determine current token is punc or not
        if item['pos'] != 'w':
            if hasNumbers(item['item']):
                print(item['item'], item['pos'], index)
    #         for character in pinyin(item['item'],style=Style.NORMAL):
    #             if character not in lexicon_syl:
    #                 lexicon_syl.append(character)
    #             initial = style.convert(character[0], Style.INITIALS, True)
    #             final = style.convert(character[0], Style.FINALS, True)
    #             if initial != "":
    #                 ipa_initial = ipa_dict[initial]
    #                 # syl_phone_list.extend(ipa_initial)
    #                 print(initial,final)
    #             else:
    #                 print(final)
    #             pinyin_line.extend(character)
    # print(lexicon_syl)
    # break