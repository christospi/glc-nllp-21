import json
import os
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
from Levenshtein import distance
from random import shuffle
import requests

import properties as pr


'''
DATA UTILITIES
'''

date_rgx = re.compile(r'(της|Της|τη|την).*((1[89]\d\d)|(20[01]\d))')
basic_date_rgx = re.compile(r'((1[89]\d\d)|(20[01]\d))\.?(\s+|$)')
law_id_rgx = re.compile(r'[Α-Ωα-ω.\s]([\d]+)')
law_id_next_line_rgx = re.compile(r'^([\d]+)')

leg_template = 'http://legislation.di.uoa.gr/eli/{}/{}/{}'

nomothesia_law_types = {'ΣΥΜΦΩΝΙΑ': 'agr',
                        'ΑΝΑΚΟΙΝΩΣΗ': 'ann',
                        'ΠΡΑΞΗ ΥΠΟΥΡΓΙΚΟΥ ΣΥΜΒΟΥΛΙΟΥ': 'amc',
                        'ΑΠΟΦΑΣΗ': 'dec',
                        'ΠΡΑΞΗ ΝΟΜΟΘΕΤΙΚΟΥ ΠΕΡΙΕΧΟΜΕΝΟΥ': 'la',
                        'ΠΡΟΕΔΡΙΚΟ ΔΙΑΤΑΓΜΑ': 'pd',
                        'ΚΑΝΟΝΙΣΜΟΣ': 'gocreg',
                        'ΝΟΜΟΣ': 'law'}

# Major law types to match the laws extracted from Raptarchis
final_law_types = {
    'ΑΝΑΓΚΑΣΤΙΚΟΣ ΝΟΜΟΣ': ['ΑΝΑΓΚΑΣΤΙΚΟΣ ΝΟΜΟΣ', 'ΑΝΑΓΚ ΝΟΜΟΣ', 'ΑΝΑΓΚ. ΝΟΜΟΣ', 'ΑΝΑΓΚΝΟΜΟΣ'],
    'ΝΟΜΟΣ': ['ΝΟΜΟΣ'],
    'ΑΠΟΦΑΣΗ': ['ΑΠΟΦΑΣΗ', 'ΑΠΟΦΑΣΙΣ'],
    'ΒΑΣΙΛΙΚΟ ΔΙΑΤΑΓΜΑ': ['ΒΑΣΙΛΙΚΟ ΔΙΑΤΑΓΜΑ'],
    'ΑΝΑΓΚΑΣΤΙΚΟ ΔΙΑΤΑΓΜΑ': ['ΑΝΑΓΚΑΣΤΙΚΟ ΔΙΑΤΑΓΜΑ'],
    'ΝΟΜΟΘΕΤΙΚΟ ΔΙΑΤΑΓΜΑ': ['ΝΟΜΟΘΕΤΙΚΟ ΔΙΑΤΑΓΜΑ', 'ΝΟΜΟΘΕΤ ΔΙΑΤΑΓΜΑ', 'ΝΟΜΟΘΕΤΔΙΑΤΑΓΜΑ'],
    'ΚΑΝΟΝΙΣΤΙΚΟ ΔΙΑΤΑΓΜΑ': ['ΚΑΝΟΝΙΣΤΙΚΟ ΔΙΑΤΑΓΜΑ'],
    'ΠΟΛΕΜΙΚΟ ΔΙΑΤΑΓΜΑ': ['ΠΟΛΕΜΙΚΟ ΔΙΑΤΑΓΜΑ'],
    'ΠΟΛΙΤΙΚΟ ΔΙΑΤΑΓΜΑ': ['ΠΟΛΙΤΙΚΟ ΔΙΑΤΑΓΜΑ'],
    'ΠΡΟΕΔΡΙΚΟ ΔΙΑΤΑΓΜΑ': ['ΠΡΟΕΔΡΙΚΟ ΔΙΑΤΑΓΜΑ'],
    'ΔΙΑΤΑΓΜΑ': ['ΔΙΑΤΑΓΜΑ'],
    'ΕΓΚΥΚΛΙΟΣ': ['ΕΓΚΥΚΛΙΟΣ'],
    'ΣΥΝΤΑΝΤΙΚΗ ΠΡΑΞΗ': ['ΣΥΝΤΑΝΤΙΚΗ ΠΡΑΞΗ'],
    'ΚΑΝΟΝΙΜΟΣ': ['ΚΑΝΟΝΙΜΟΣ'],
    'ΑΓΟΡΑΝΟΜΙΚΗ ΔΙΑΤΑΞΗ': ['ΑΓΟΡΑΝΟΜΙΚΗ ΔΙΑΤΑΞΗ'],
    'ΥΓΕΙΟΝΟΜΙΚΗ ΔΙΑΤΑΞΗ': ['ΥΓΕΙΟΝΟΜΙΚΗ ΔΙΑΤΑΞΗ'],
    'ΑΣΤΥΝΟΜΙΚΗ ΔΙΑΤΑΞΗ': ['ΑΣΤΥΝΟΜΙΚΗ ΔΙΑΤΑΞΗ'],
    'ΚΑΝΟΝΙΣΤΙΚΗ ΔΙΑΤΑΞΗ': ['ΚΑΝΟΝΙΣΤΙΚΗ ΔΙΑΤΑΞΗ'],
    'ΔΙΑΤΑΞΗ': ['ΔΙΑΤΑΞΗ'],
    'ΑΝΑΚΟΙΝΩΣΗ': ['ΑΝΑΚΟΙΝΩΣΗ'],
    'ΔΙΑΓΓΕΛΜΑ': ['ΔΙΑΓΓΕΛΜΑ'],
    'ΔΙΑΚΗΡΥΞΗ': ['ΔΙΑΚΗΡΥΞΗ', 'ΔΙΑΚΗΡΥΞΙΣ'],
    'ΠΡΑΞΗ ΥΠΟΥΡΓΙΚΟΥ ΣΥΜΒΟΥΛΙΟΥ': ['ΠΡΑΞΗ ΥΠΟΥΡΓΙΚΟΥ ΣΥΜΒΟΥΛΙΟΥ', 'ΠΡΑΞΙΣ ΥΠΟΥΡΓΙΚΟΥ ΣΥΜΒΟΥΛΙΟΥ'],
    'ΠΡΑΞΗ ΝΟΜΟΘΕΤΙΚΟΥ ΠΕΡΙΕΧΟΜΕΝΟΥ': ['ΠΡΑΞΗ ΝΟΜΟΘΕΤΙΚΟΥ ΠΕΡΙΕΧΟΜΕΝΟΥ', 'ΠΡΑΞΙΣ ΝΟΜΟΘΕΤΙΚΟΥ ΠΕΡΙΕΧΟΜΕΝΟΥ'],
    'ΠΡΑΞΗ': ['ΠΡΑΞΗ', 'ΠΡΑΞΙΣ'],
    'ΨΗΦΙΣΜΑ': ['ΨΗΦΙΣΜΑ'],
    'ΚΑΝΟΝΙΣΜΟΣ': ['ΚΑΝΟΝΙΣΜΟΣ']
}


def normalize(string):
    return ''.join(
        (c for c in unicodedata.normalize('NFD', string.upper().strip(' ')) if unicodedata.category(c) != 'Mn'))


def classify_law(line):
    maxsim = 0
    lawtype = ''

    for key, value in final_law_types.items():
        for law_type in value:
            sim = SequenceMatcher(None, law_type, line).ratio()
            if sim > maxsim:
                maxsim = sim
                lawtype = key
        # sims.append(SequenceMatcher(None, cls, line).ratio())
        # sims.append(distance(cls, line))
    f = open('Resources/classmatch.txt', "a")
    # f = open('../Resources/classmatch.txt', "a")
    f.write(line + ' | ' + lawtype + ' | ' + str(maxsim) + '\n')

    return lawtype


# Capture law's year - if found in first lines
def fetch_year(rn):
    for vol in rn.children:
        for ch in vol.children:
            for subj in ch.children:
                for law in subj.children:
                    if law.text:
                        # Check in the first 3 lines if any year is mentioned, according to year_rgx
                        splits = law.text.splitlines()
                        limit = 6 if len(splits) > 6 else len(splits)
                        found = False
                        for line in splits[:limit]:
                            found = date_rgx.search(line)
                            if found:
                                law.year = found.group(2).strip(' .,\n')
                                break
                        if not found:
                            for line in splits[:limit]:
                                found = basic_date_rgx.search(line)
                                if found:
                                    law.year = found.group(0).strip(' .,\n')
                                    break


# Capture law's ID
def fetch_law_id(rn):
    for vol in rn.children:
        for ch in vol.children:
            for subj in ch.children:
                for law in subj.children:
                    if law.title:
                        found = law_id_rgx.search(law.title)
                        if found:
                            law.law_id = found.group(1)
                        # If law_id not found in title (i.e. ΝΟΜΟΣ υπ' αριθ 134) then check for law_id in the next line
                        # aka the 1st line of the law's text
                        else:
                            if law.text:
                                found = law_id_next_line_rgx.search(''.join(law.text.splitlines()[0:2]))
                                if found:
                                    law.law_id = found.group(1)
                                else:
                                    if law.type:
                                        found = law_id_rgx.search(law.type)
                                        if found:
                                            law.law_id = found.group(1)


# Generate legislation URI
def generate_leg_uri(rn):
    for vol in rn.children:
        for ch in vol.children:
            for subj in ch.children:
                for law in subj.children:
                    if law.law_id and law.year:
                        year = int(law.year)
                        if 1989 < year < 2019:
                            type2check = nomothesia_law_types.get(law.type)
                            if type2check:
                                uri2check = leg_template.format(type2check, law.year, law.law_id)
                                fetched = requests.get(uri2check)
                                if fetched.status_code == 200:
                                    law.leg_uri = uri2check


def export_contents(root_node):
    with open('Resources/contents.txt', 'w') as f:
        for vol in root_node.children:
            f.write(vol.id + ' - ' + vol.title + '\n')
            for ch in vol.children:
                f.write('\t' + ch.id + ' - ' + ch.title + '\n')
                for subj in ch.children:
                    f.write('\t\t' + subj.id + ' - ' + subj.title + '\n')
                    # for law in subj.children:
                    #     f.write('\t\t\t' + law.title + '\n')


def show_headers(rn, year_flag=False):
    if year_flag:
        f = open('Resources/year_headers.txt', "w")
    else:
        f = open('Resources/headers.txt', "w")

    for vol in rn.children:
        for ch in vol.children:
            for subj in ch.children:
                for law in subj.children:
                    # If u want to check for year in header content
                    if year_flag:
                        # If year is found, then do nothing. Otherwise, print the header
                        if law.year is None:
                            f.write(vol.title + '/' + ch.title + '/' + subj.title + '/' + law.title + '\n')
                            f.write(law.text + '\n\n')
                    # Otherwise it only prints the header if NO article is found inside law
                    else:
                        if not law.children:
                            f.write(vol.title + '/' + ch.title + '/' + subj.title + '/' + law.title + '\n')
                            f.write(law.text + '\n\n')


def dump_to_json(root_node, json_path):
    json_dict = {'title': root_node.title,
                 'type': root_node.type,
                 'year': root_node.year,
                 'law_id': root_node.law_id,
                 'leg_uri': root_node.leg_uri,
                 'volume': json_path.split('/')[-5],
                 'chapter': json_path.split('/')[-4].split('_')[-1].strip(' .,').upper(),
                 'subject': json_path.split('/')[-3].split('_')[-1].strip(' .,').upper(),
                 'header': root_node.title + '\n' + root_node.text,
                 'articles': []
                 }

    for art in root_node.children:
        json_dict['articles'].append(art.text)

    with open(json_path + '.json', "w") as jsf:
        json.dump(json_dict, jsf, ensure_ascii=False)


def dump_to_files(root_node, start_year=1000):
    # Call with different start_year to adjust range, e.g. start from 1989
    file_root = pr.output_folder
    for vol in root_node.children:
        os.makedirs(file_root + vol.title, exist_ok=True)
        for ch in vol.children:
            os.makedirs(file_root + vol.title + '/' + ch.id + '_' + ch.title, exist_ok=True)
            for subj in ch.children:
                no_law_text_id = 1
                subj.title = subj.title.replace('/', '')

                lawcation = file_root + vol.title + '/' + ch.id + '_' + ch.title + '/' + subj.id.upper() + '_' + subj.title
                os.makedirs(lawcation, exist_ok=True)
                os.makedirs(lawcation + '/WITH_ARTICLES', exist_ok=True)
                os.makedirs(lawcation + '/NO_ARTICLES', exist_ok=True)

                # One file per subject
                # f = open(file_root + vol.title + '/' + ch.title + '/' + subj.title + '/' + subj.title + '.txt', "w")

                for law in subj.children:
                    # NOTE: In cases where no year found, the law is not dumped to .json file
                    if law.year is not None:
                        if int(law.year) > start_year:

                            law.title = (law.title[:75] + '..') if len(law.title) > 75 else law.title

                            if law.children:
                                jsonpath = lawcation + '/WITH_ARTICLES/' + law.id + '-' + law.title.replace('/', '_')
                            else:
                                jsonpath = lawcation + '/NO_ARTICLES/' + law.id + '-' + law.title.replace('/', '_')

                            dump_to_json(law, jsonpath)


# DEPRICATED - Find laws/pd++ that exist more than once in dataset. (Same leg_uri (type/year/id)
# BUT (probably) different part of law's text)
def expose_duplicates():
    # duplicates = dict()
    duplicates = set()
    list_of_files = []
    for root, _, f_names in os.walk(pr.output_folder):
        for jsonfile in f_names:
            list_of_files.append(os.path.join(root, jsonfile))

    for i, json1 in enumerate(list_of_files):
        for j, json2 in enumerate(list_of_files):
            if i <= j:
                continue
            with open(json1) as jsf1:
                with open(json2) as jsf2:
                    data1 = json.load(jsf1)
                    data2 = json.load(jsf2)
                    if data1['leg_uri'] is not None and data1['leg_uri'] == data2['leg_uri']:
                        duplicates.add(data1['leg_uri'])
                        print(data1['leg_uri'])

    with open('/home/chris/Desktop/duplicates_final.txt', 'w') as outf:
        for item in duplicates:
            outf.write(item + '\n')

    return duplicates


# BETTER duplicates exposure
def tuned_expose_duplicates():
    list_of_files = []
    catalogue = dict()
    full_dupli_set = set()
    total_unique_laws = 0

    for root, _, f_names in os.walk(pr.output_folder_enhanced):
        for jsonfile in f_names:
            list_of_files.append(os.path.join(root, jsonfile))

    with open('/home/chris/Desktop/duplicate_logs.txt', 'w') as logs:
        for i, json1 in enumerate(list_of_files):
            for j, json2 in enumerate(list_of_files):
                if i <= j:
                    continue
                with open(json1) as jsf1:
                    with open(json2) as jsf2:
                        data1 = json.load(jsf1)
                        data2 = json.load(jsf2)
                        if data1['type'] == data2['type'] \
                                and data1['law_id'] == data2['law_id'] and data1['law_id'] is not None \
                                and data1['year'] == data2['year'] and data1['year'] is not None:

                            key = '{}/{}/{}'.format(data1['type'], data1['year'], data1['law_id'])

                            if key not in catalogue:
                                catalogue[key] = set()

                            catalogue[key].add(json1)
                            catalogue[key].add(json2)
                            full_dupli_set.add(json1)
                            full_dupli_set.add(json2)

                            print(json1 + '\n' + json2 + '\n' + '--------------------')
                            logs.write(json1 + '\n' + json2 + '\n' + '--------------------' + '\n')

    with open('/home/chris/Desktop/duplicate_data.json', 'w') as outf:
        full_list = []
        for k, v in catalogue.items():
            entry = {
                'law_uri': k,
                'duplicates': list(sorted(v))
            }
            full_list.append(entry)

        total_unique_laws = len(full_list)
        final_json = {'data': full_list}
        json.dump(final_json, outf, ensure_ascii=False)

    with open('/home/chris/Desktop/duplicate_entries.json', 'w') as allf:
        entries_json = {'data': list(full_dupli_set)}
        json.dump(entries_json, allf, ensure_ascii=False)

    print('\n\n' + 'Total duplicate files: ' + str(len(full_dupli_set)) + '\n')
    print('Total unique duplicate laws: ' + str(total_unique_laws) + '\n')


def remove_duplicates():
    duplicates = set()
    total_count = 0
    unique_count = 0
    with open('/home/chris/Desktop/duplicate_entries.json') as file:
        data = json.load(file)
        duplicates.update(data['data'])

    for root, _, f_names in os.walk(pr.output_folder_enhanced):
        for jsonfile in f_names:
            total_count += 1
            if os.path.join(root, jsonfile) not in duplicates:
                unique_count += 1
                unique_path = root.replace('_ENHANCED', '_UNIQUE')
                os.makedirs(unique_path, exist_ok=True)
                shutil.copy(os.path.join(root, jsonfile), unique_path)

    print('Total Docs: ' + str(total_count))
    print('Unique Docs: ' + str(unique_count))
    print('Duplicate Docs (Removed): ' + str(total_count - unique_count))


def remove_useless_text():
    useless_text = re.compile(r'([Κκ]αταργ|[Αα]ντικα[τθ]|[Ττ]ροποπ).*')
    for root, _, f_names in os.walk('/home/chris/Desktop/RAPTARCHIS_COPY'):
        for jsonfile in f_names:
            with open(os.path.join(root, jsonfile)) as file:
                data = json.load(file)
                articles = data['articles']
                if len(articles) > 0:
                    for art in articles:
                        found = useless_text.search(art)
                        if found:
                            # TODO something
                            continue
                else:
                    found = useless_text.search(data['header'])
                    if found:
                        # TODO something
                        continue


'''
EXPERIMENT FUNCTIONS

After finalizing the dataset (duplicates, URIs, removals etc.) use these functions to create the final data to be used
on NN experiments
'''


def create_experiment_sets():
    # limits = [TRAIN, DEV, TEST]
    limits = [0.6, 0.2, 0.2]
    os.makedirs(pr.output_folder_train, exist_ok=True)
    os.makedirs(pr.output_folder_dev, exist_ok=True)
    os.makedirs(pr.output_folder_test, exist_ok=True)

    train_counter = 1
    dev_counter = 1
    test_counter = 1

    vol_label = sorted(os.listdir(pr.output_folder_unique))

    vol_path = sorted([os.path.join(pr.output_folder_unique, vol)
                       for vol in os.listdir(pr.output_folder_unique)
                       if os.path.isdir(os.path.join(pr.output_folder_unique, vol))])

    for label, path in zip(vol_label, vol_path):
        vol_files = []
        for root, _, f_names in os.walk(path):
            for jsf in f_names:
                vol_files.append(os.path.join(root, jsf))

        # NOTE: shuffle works IN PLACE
        shuffle(vol_files)

        train_n = int(round(limits[0] * len(vol_files)))
        dev_n = int(round(limits[1] * len(vol_files)))
        # test_n = len(vol_files) - train_n - dev_n

        # print('TOTAL: {} -> [{}, {}, {}] = {}'.format(len(vol_files), train_n, dev_n, test_n, train_n+dev_n+test_n))

        train_list = vol_files[:train_n]
        dev_list = vol_files[train_n:train_n + dev_n]
        test_list = vol_files[train_n + dev_n:]

        # print('-----> {} -> [{}, {}, {}] = {}'.format(len(vol_files), len(train_list), len(dev_list), len(test_list),
        #                                               len(train_list) + len(dev_list) + len(test_list)))

        for file in train_list:
            # V.1: Keep hierarchy in experiment files
            # exp_file = file.replace('_UNIQUE', '_EXP/TRAIN')
            # os.makedirs(exp_file, exist_ok=True)
            # shutil.copy(file, exp_file)

            # V.2: Ignore hierarchy and put everything in same folder
            shutil.copy(file, pr.output_folder_train + str(train_counter) + '.json')
            train_counter += 1

        for file in dev_list:
            # exp_file = file.replace('_UNIQUE', '_EXP/DEV')
            # os.makedirs(exp_file, exist_ok=True)
            # shutil.copy(file, exp_file)

            shutil.copy(file, pr.output_folder_dev + str(dev_counter) + '.json')
            dev_counter += 1

        for file in test_list:
            # exp_file = file.replace('_UNIQUE', '_EXP/TEST')
            # os.makedirs(exp_file, exist_ok=True)
            # shutil.copy(file, exp_file)

            shutil.copy(file, pr.output_folder_test + str(test_counter) + '.json')
            test_counter += 1


def create_experiment_class_json():
    final_json = dict()
    vol_label = sorted(os.listdir(pr.output_folder_unique))

    for i, label in enumerate(vol_label):
        final_json[i+1] = {
            'concept_id': str(i+1),
            'label': label,
            'alt_labels': [],
            'parents': []
        }

    with open(pr.output_folder_exp + 'raptarchis_el.json', 'w') as jsf:
        json.dump(final_json, jsf, ensure_ascii=False)


'''
MAIN FUNCTION
'''
if __name__ == "__main__":
    # expose_duplicates()
    # tuned_expose_duplicates()
    # duplicate_statistics()
    # remove_duplicates()
    # docs_per_class()
    # create_experiment_sets()
    # create_experiment_class_json()
    # count_classes()
    exit(0)
