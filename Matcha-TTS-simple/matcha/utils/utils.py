import os
import sys

def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def dict_to_attrdic(dict_obj):
    attrdict_obj = AttrDict(dict_obj)
    for k in attrdict_obj:
        if type(attrdict_obj[k]) is dict:
            attrdict_obj[k] = dict_to_attrdic(attrdict_obj[k])
    return attrdict_obj