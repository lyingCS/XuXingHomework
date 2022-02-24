# -*- coding: utf-8 -*-
import json
from collections import Counter

poetry_tang = "dataset/poetryTang.txt"
max_len = 50

with open(poetry_tang, 'r', encoding='utf8') as l:
    lines = l.readlines()


def remove_punc(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_ ~《》【】（）'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char  # space is also a character
    return no_punct


pairs = []
for line in lines:
    objects = line.split("::")
    first = remove_punc(objects[0]).strip()
    second = remove_punc(objects[-1]).strip()
    first = [s for s in first]
    second = [s for s in second]
    if len(second) > max_len or len(first) > max_len:
        continue
    pairs.append([first, second])

word_freq = Counter()
for pair in pairs:
    word_freq.update(pair[0])
    word_freq.update(pair[1])

min_word_freq = 5
words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

print("Total words are: {}".format(len(word_map)))

with open('WORDMAP_poetry.json', 'w') as j:
    json.dump(word_map, j)


def encode_question(words, word_map):
    enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c


def encode_reply(words, word_map):
    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<end>']] + [
        word_map['<pad>']] * (max_len - len(words))
    return enc_c


pairs_encoded = []
for pair in pairs:
    qus = encode_question(pair[0], word_map)
    ans = encode_reply(pair[1], word_map)
    pairs_encoded.append([qus, ans])

with open('pairs_encoded.json', 'w') as p:
    json.dump(pairs_encoded, p)
