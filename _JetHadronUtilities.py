from collections import defaultdict


def return_none():
    return None

def return_true():
    return True

def init_none_dict():
    return defaultdict(return_none)

def init_bool_dict():
    return defaultdict(return_true)