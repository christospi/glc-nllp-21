import glob
import json
from collections import Counter
import logging
import os
import numpy as np
import tqdm

train_path = '/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/TRAIN_DATA/train'
dev_path = '/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/TRAIN_DATA/dev'
test_path = '/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/TRAIN_DATA/test'

class_test = 'subject'

# Load train dataset and count labels
train_files = glob.glob('/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/TRAIN_DATA/train/*.json')
train_counts = Counter()
for filename in tqdm.tqdm(train_files):
    with open(filename) as file:
        data = json.load(file)
        train_counts[data[class_test]] += 1

train_concepts = set(list(train_counts))

frequent, few = [], []
for i, (label, count) in enumerate(train_counts.items()):
    if count > 10:
        frequent.append(label)
    else:
        few.append(label)

# Load dev/test datasets and count labels
rest_files = glob.glob('/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/TRAIN_DATA/dev/*.json')
rest_files += glob.glob('/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/TRAIN_DATA/test/*.json')
rest_concepts = set()
for filename in tqdm.tqdm(rest_files):
    with open(filename) as file:
        data = json.load(file)
        rest_concepts.add(data[class_test])

# Compute zero-shot group
zero = list(rest_concepts.difference(train_concepts))

for label in zero:
    countme = 0
    for filename in tqdm.tqdm(rest_files):
        with open(filename) as file:
            data = json.load(file)
            if data[class_test] == label:
                countme += 1
    print('\nZero label: {} - instances found in dev/test: {}\n'.format(label, countme))

label_ids = dict()
# margins = [(0, len(frequent) + len(few) + len(zero))]
# k = 0
# for group in [frequent, few, zero]:
#     margins.append((k, k + len(group)))
#     for concept in group:
#         label_ids[concept] = k
#         k += 1
# margins[-1] = (margins[-1][0], len(frequent) + len(few) + len(zero))
#
# print('just a pause plz')