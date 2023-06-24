from difflib import SequenceMatcher
import re


class LawTypeObj(object):
    def __init__(self, key='', lawlist=None):
        if lawlist is None:
            lawlist = []
        self.key = key
        self.lawlist = lawlist


with open('../Resources/law_types.txt') as f:
    law_types_list = []

    regex = re.compile(r'\d|\.')
    for line in f:
        line = re.sub(regex, '', line)
        law_types_list.append(line.strip())
        # for law_type in law_types_list:
        #     if SequenceMatcher(None, law_type.key, line).ratio() > 0.8:
        #         law_type.lawlist.append(line)
        # newnode = LawTypeObj(line, [line])
        # law_types_list.append(newnode)
    law_set = set(law_types_list)
    law_set = sorted(law_set)
    # law_set = sorted(law_set)

    with open('../Resources/law_typesNEW_reg.txt', 'w') as newf:
        type_freq = []
        final_list = []
        for type in law_set:
            type_freq.append(law_types_list.count(type))
        for type, freq in zip(law_set, type_freq):
            final_list.append([type, freq])
        final_list.sort(key=lambda x: x[1], reverse=True)
        for row in final_list:
            newf.write(str(row[1]) + ' || ' + row[0] + '\n')

    # for node in law_types_list:
    #     print(node.key + '->' + str(node.lawlist.__len__()))
