import collections
import json
import os
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = '/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/data'
TRAIN_PATH = '/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/data/train'
DEV_PATH = '/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/data/dev'
TEST_PATH = '/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_EXP/data/test'
RAP_UNIQUE_PATH = '/home/chris/DATACENTER/Raptarchis/LATEST/RAPTARCHIS_UNIQUE'

'''
STATISTICS
'''


def count_classes():
    vol_set = set()
    ch_set = set()
    subj_set = set()

    for root, _, f_names in os.walk(DATA_PATH):
        for jsonfile in f_names:
            with open(os.path.join(root, jsonfile)) as file:
                data = json.load(file)
                vol_set.add(data['volume'])
                ch_set.add(data['chapter'])
                subj_set.add(data['subject'])

    print('Volumes: {}\tChapters: {}\tSubjects: {}'.format(len(vol_set), len(ch_set), len(subj_set)))


def duplicate_statistics():
    same_volume = 0
    same_chapter = 0
    same_subject = 0
    not_same = 0

    with open('/home/chris/Desktop/duplicate_data.json') as file:
        data = json.load(file)
        for law in data['data']:
            volset = set()
            chset = set()
            subjset = set()
            for entry in law['duplicates']:
                volset.add(entry.split('/')[-5])
                chset.add(entry.split('/')[-4])
                subjset.add(entry.split('/')[-3])
            same_volume += len(volset)
            same_chapter += len(chset)
            same_subject += len(subjset)

    print('\n' + 'Duplicates on:' + '\n' +
          '-Same volume: ' + str(same_volume) + '\n'
                                                '-Same chapter: ' + str(same_chapter) + '\n'
                                                                                        '-Same subject: ' + str(
        same_subject) + '\n')


def docs_per_class(class_check='volume'):
    total = 0
    total_check = 0
    target_class = []
    target_class_counter = collections.Counter()

    for root, _, f_names in os.walk(DATA_PATH):
        for jsonfile in f_names:
            with open(os.path.join(root, jsonfile)) as file:
                data = json.load(file)
                total += 1
                target_class.append(data[class_check])
                target_class_counter[data[class_check]] += 1

    # # Method 1
    # counter = collections.Counter(target_class)
    # target_class = sorted(set(target_class))
    # counts = []
    #
    # for target in target_class:
    #     counts.append(counter[target])
    #     total_check += counter[target]
    #     print(target + ': ' + str(counter[target]))
    #
    # nsize = len(target_class)
    # ind = np.arange(nsize)
    # width = 0.5
    # fig, ax = plt.subplots()
    # fig.set_size_inches(40, 15)
    # ax.bar(ind, counts, width, color='b')
    #
    # # add some text for labels, title and axes ticks
    # ax.set_ylabel('Count')
    # ax.set_title('Docs per Class')
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(target_class, rotation='vertical')

    # # Method 2
    # # Used for some metrics like SUBJECT
    # plt.figure(figsize=(6, 6))
    #
    # plt.hist(sorted(target_class, key=collections.Counter(target_class).get, reverse=True),
    #          rwidth=5, orientation="vertical")
    # plt.title('Legal Resources Per {}'.format(class_check.casefold()))
    # plt.ylabel('Legal Resources')
    # plt.xlabel("Volumes")
    # # plt.xticks(range(0, len(set(target_class))))
    # plt.xticks(np.linspace(5, 50, 10))

    # # Method 3 - VERTICAL CHART
    # plt.rcdefaults()
    #
    # fig, ax = plt.subplots(figsize=(10, 15))
    #
    # y_pos = np.arange(len(target_class_counter))
    #
    # ax.barh(y_pos, [val[1] for val in target_class_counter.most_common()], align='center')
    #
    # # Limit X axis values
    # # ax.set_xlim(1000, 3000)
    #
    # '''
    # basically plt.xticks and plt.yticks accept lists as input and use them as markers on the x axis and y axis
    # respectively, np.linspace generates an array with start,stop and number of points.
    # '''
    # # ax.set_xticks(np.linspace(200, 4000, 3))
    #
    # ax.set_yticks(y_pos)
    # ylabels = [val[0] for val in target_class_counter.most_common()]
    # ylabels = ['\n'.join(wrap(la, 30)) for la in ylabels]
    # ax.set_yticklabels(ylabels)
    # ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('Total docs')
    # ax.set_title('Docs per class')
    # for i, v in enumerate([val[1] for val in target_class_counter.most_common()]):
    #     ax.text(v + 3, i + .25, str(v), color='gray', fontweight='bold')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    # # FINAL Method 4
    fig, ax = plt.subplots(1, 1)
    a = np.array(list(target_class_counter.values()))
    # ax.hist(a)
    ax.hist(a, bins='auto')
    plt.axvline(a.mean(), color='k', linestyle='dashed', linewidth=1, label='Mean: {}'.format(int(a.mean())))
    plt.legend(loc='upper right')
    ax.set_title('Legal Resources in {}s'.format(class_check.capitalize()))
    # ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Number of Legal Resources')
    ax.set_ylabel('Number of Classes')

    # plt.tight_layout()
    plt.savefig('Resources/per_{}_stats.png'.format(class_check))
    plt.show()

    # print('\nTotal: ' + str(total) + ' aka ' + str(total_check))


def token_statistics():
    lengths = []
    file_count = 0
    low_tokens_counter = 0
    vol_counter = collections.Counter()
    ch_counter = collections.Counter()
    subj_counter = collections.Counter()

    lawsfile = open('Resources/low_tokens_laws.txt', 'w', encoding='utf8')
    for root, _, f_names in os.walk(DATA_PATH):
        for jsonfile in f_names:
            with open(os.path.join(root, jsonfile)) as file:
                data = json.load(file)
                file_count += 1
                # If law has articles then merge with header in order to count tokens
                if len(data['articles']) > 0:
                    arts = '\n'.join(data['articles'])
                    cur_length = len(data['header'].split()) + len(arts.split())
                else:
                    cur_length = len(data['header'].split())

                lengths.append(cur_length)

                # Check documents under a specific token count
                if cur_length < 100:
                    low_tokens_counter += 1
                    lawsfile.write(str(os.path.join(root, jsonfile)) + '\n\n')
                    json.dump(data, lawsfile, ensure_ascii=False)
                    lawsfile.write('\n\n-----------------------------------------------------------\n\n')
                # Counter for each volume, chapter, subject
                vol_counter[root.split('/')[-4]] += 1
                ch_counter[root.split('/')[-3]] += 1
                subj_counter[root.split('/')[-2]] += 1

    print('TOTAL DOCUMENTS: {}'.format(file_count))
    print('LOW TOKENS DOCUMENTS: {}'.format(low_tokens_counter))
    print(vol_counter.most_common())
    lengths = sorted(lengths)
    for i in range(20):
        end = int(((i + 1) / 10) * len(lengths) / 2)
        print(
            'DATASET ({0:3}%): DOCS: {1:5} LENGTH: MAX: {2:3} MEAN: {3:.2f}'.format(
                (i + 1) * 5, end, max(lengths[:end]), np.mean(lengths[:end])))

    plt.hist(lengths, bins=50, range=(0, 2000))
    plt.axvline(np.array(lengths).mean(), color='k', linestyle='dashed', linewidth=1,
                label='Mean: {}'.format(int(np.array(lengths).mean())))
    plt.legend(loc='upper right')
    plt.title('Tokens Count')
    plt.ylabel('Legal Resources')
    plt.xlabel("Number of Tokens")
    plt.savefig('Resources/tokens_stats.png')
    plt.show()


def per_set_token_statistics():
    total_lengths = []
    total_docs = 0

    for exp_set in [TRAIN_PATH, DEV_PATH, TEST_PATH]:
        lengths = []
        file_count = 0
        low_tokens_counter = 0
        vol_counter = collections.Counter()
        ch_counter = collections.Counter()
        subj_counter = collections.Counter()

        print(exp_set.split('/')[-1], '\n', 30 * '-')

        for root, _, f_names in os.walk(exp_set):
            for jsonfile in f_names:
                with open(os.path.join(root, jsonfile)) as file:
                    data = json.load(file)
                    file_count += 1
                    total_docs += 1
                    # If law has articles then merge with header in order to count tokens
                    if len(data['articles']) > 0:
                        arts = '\n'.join(data['articles'])
                        cur_length = len(data['header'].split()) + len(arts.split())
                    else:
                        cur_length = len(data['header'].split())

                    lengths.append(cur_length)
                    total_lengths.append(cur_length)

                    # Check documents under a specific token count
                    low_tokens_counter += 1 if cur_length < 100 else 0

                    # Counter for each volume, chapter, subject
                    vol_counter[data['volume']] += 1
                    ch_counter[data['chapter']] += 1
                    subj_counter[data['subject']] += 1

        print('TOTAL DOCUMENTS IN {} SET: {}'.format(exp_set.split('/')[-1], file_count))
        print('LOW TOKENS DOCUMENTS IN {} SET: {}'.format(exp_set.split('/')[-1], low_tokens_counter))
        print('MEAN OF TOKENS PER DOC IN {} SET: {}'.format(exp_set.split('/')[-1], np.mean(lengths)))
        # print(vol_counter.most_common())
    print('\n', 30 * '=')
    print('MEAN OF TOKENS IN {} DOCS: {}'.format(total_docs, np.mean(total_lengths)))


def cli_statistics(root_node):
    with open('./Resources/statistics.txt', 'w') as f:
        for vol in root_node.children:
            f.write(vol.title + '\n')
            for ch in vol.children:
                f.write('\t' + ch.title + '\n')
                for subj in ch.children:
                    f.write('\t\t' + subj.title + '\n')
                    wart = 0
                    noart = 0
                    for law in subj.children:
                        if law.children:
                            wart += 1
                        else:
                            noart += 1
                    f.write('\t\t\tTOTAL:' + str(len(subj.children))
                            + '\n\t\t\tWITH_ARTICLES:' + str(wart)
                            + '\n\t\t\tNO_ARTICLES:' + str(noart)
                            + '\n\t\t\tFAILS:' + str(len(subj.children) - wart - noart) + '\n')


# TODO: Need to be refactored to read from JSON files not from mem
def create_article_charts(root_node):
    vol_data = []
    no_art = []
    with_art = []
    for vol in root_node.children:
        vol_data.append(vol.title)
        no_art.append(0)
        with_art.append(0)
        for ch in vol.children:
            for subj in ch.children:
                for law in subj.children:
                    if law.children:
                        with_art[-1] += 1
                    else:
                        no_art[-1] += 1
    x = []
    for id, item in enumerate(vol_data):
        x.append(id)

    # fig = plt.figure(frameon=False)
    # w = 20
    # h = 10
    # fig.set_size_inches(w, h)
    # ax = fig.add_subplot(111)

    N = len(vol_data)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15)
    rects1 = ax.bar(ind, with_art, width, color='g')

    rects2 = ax.bar(ind + width, no_art, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Laws')
    ax.set_title('Laws per Volume')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(vol_data, rotation='vertical')
    ax.legend((rects1[0], rects2[0]), ('With Articles', 'No Articles'))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.savefig('Resources/articles_chart.png')


def create_year_charts():
    no_year_count = 0
    file_count = 0
    years = []
    for root, _, f_names in os.walk(DATA_PATH):
        for jsonfile in f_names:
            with open(os.path.join(root, jsonfile)) as file:
                data = json.load(file)
                file_count += 1
                if data['year'] is not None:
                    years.append(int(data['year']))
                else:
                    no_year_count += 1

    plt.hist(years, bins=50)
    plt.title('Legal Resources Per Year')
    plt.ylabel('Legal Resources')
    plt.xlabel("Year")
    # plt.axvline(np.array(years).mean(), color='k', linestyle='dashed', linewidth=1, label=int(np.array(years).mean()))
    # plt.legend(loc='upper right')
    plt.savefig('/home/chris/PycharmProjects/RaptarchisData/Resources/years_chart.png')
    plt.show()


# RATIO OF TRAIN-DEV-TEST SAMPLES (60-20-20)
def data_ratio_tdt(level='chapter'):
    train_counter = collections.Counter()
    dev_counter = collections.Counter()
    test_counter = collections.Counter()
    total_counter = collections.Counter()

    for exp_set in [TRAIN_PATH, DEV_PATH, TEST_PATH]:

        print(exp_set.split('/')[-1].upper(), '\n', 30 * '-')

        for root, _, f_names in os.walk(exp_set):
            for jsonfile in f_names:
                with open(os.path.join(root, jsonfile)) as file:
                    data = json.load(file)
                    if 'train' in exp_set:
                        train_counter[data[level]] += 1
                    elif 'dev' in exp_set:
                        dev_counter[data[level]] += 1
                    else:
                        test_counter[data[level]] += 1
                    total_counter[data[level]] += 1

    for item in total_counter.most_common():
        print('{:>25} -> TOTAL: {} TRAIN: {} ({:.2f}%) DEV: {} ({:.2f}%) TEST: {} ({:.2f}%)'.format(
            item[0], total_counter[item[0]], train_counter[item[0]],
            100 * float(train_counter[item[0]]) / float(total_counter[item[0]]),
            dev_counter[item[0]], 100 * float(dev_counter[item[0]]) / float(total_counter[item[0]]),
            test_counter[item[0]], 100 * float(test_counter[item[0]]) / float(total_counter[item[0]]),
        ))


def how_many_freq_few_zero(class_check='subject'):
    test_counter = collections.Counter()
    train_counter = collections.Counter()
    dev_counter = collections.Counter()
    set_of_cls = set()

    few = 0
    freq = 0
    zero = 0
    resid = 0

    for exp_set in [TRAIN_PATH, DEV_PATH, TEST_PATH]:
        for root, _, f_names in os.walk(exp_set):
            for jsonfile in f_names:
                with open(os.path.join(root, jsonfile)) as file:
                    data = json.load(file)
                    set_of_cls.add(data[class_check])
                    if exp_set == TRAIN_PATH:
                        train_counter[data[class_check]] += 1
                    elif exp_set == DEV_PATH:
                        dev_counter[data[class_check]] += 1
                    else:
                        test_counter[data[class_check]] += 1

    for entry in set_of_cls:
        train_counter[entry] += 0
        dev_counter[entry] += 0
        test_counter[entry] += 0

    # for entry in set_of_cls:
    #     if train_counter[entry] > 0 and dev_counter[entry] + test_counter[entry] == 0:
    #         print('Only in train: {} found: {}'.format(entry, train_counter[entry]))
    #     if dev_counter[entry] + test_counter[entry] > 0 and train_counter[entry] == 0:
    #         print('Only in dev/test: {} found: {}'.format(entry, dev_counter[entry] + test_counter[entry]))

    for entry in set_of_cls:
        if train_counter[entry] >= 10 and dev_counter[entry] > 0 and test_counter[entry] > 0:
            freq += train_counter[entry] + dev_counter[entry] + test_counter[entry]
        elif train_counter[entry] > 0:
            few += train_counter[entry] + dev_counter[entry] + test_counter[entry]
        elif train_counter[entry] == 0 and dev_counter[entry] + test_counter[entry] > 0:
            zero += train_counter[entry] + dev_counter[entry] + test_counter[entry]
        else:
            resid += train_counter[entry] + dev_counter[entry] + test_counter[entry]

    total = freq + few + zero + resid
    print('LEVEL TO CHECK: {} - TOTAL DOCS: {}\nFREQUENT: {} = {}   FEW: {} = {}  ZERO: {} = {}\n'
          .format(class_check, total, freq, freq / total * 100, few, few / total * 100, zero, zero / total * 100))
    print('RESID: {} = {}'.format(resid, resid / total * 100))


# ######################################################################################################################
# Executing methods
# ######################################################################################################################


docs_per_class('volume')
# per_set_token_statistics()
# create_year_charts()
# data_ratio_tdt()
# create_year_charts()
# docs_per_class()
# how_many_freq_few_zero()
# token_statistics()
