import torch 
import random


def tokens_enhance(tokens):
    l = len(tokens)
    
    enhance_id = random.randint(1, 2)
    if enhance_id == 1:
        # 随机交换两个词的位置
        b = random.randint(1, l-1)
        c = random.randint(1, l-1)
        d = tokens[b]
        tokens[b] = tokens[c]
        tokens[c] = d

    if enhance_id == 2:
        # mask 某个token
        b = random.randint(1, l-1)
        tokens[b] = 103
    return tokens


    