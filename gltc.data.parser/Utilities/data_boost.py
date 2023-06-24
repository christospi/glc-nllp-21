import os
import re
import json
import shutil
import requests

import properties as pr
import utils as ut


def legislation_boost():

    total_docs = 0
    enhanced_docs = 0

    # 1st RUN
    # duplicates = ut.expose_duplicates()

    # 2nd RUN
    duplicates = set()
    with open('/home/chris/Desktop/duplicates_final.txt') as dupl:
        for line in dupl:
            duplicates.add(line.strip())

    for root, _, f_names in os.walk(pr.output_folder):
        for jsonfile in f_names:
            with open(os.path.join(root, jsonfile)) as file:
                enhanced = False
                law_data = json.load(file)
                if law_data['leg_uri'] is not None:
                    if law_data['leg_uri'] not in duplicates:
                        # print(law_data['leg_uri'])
                        leg_uri = law_data['leg_uri']
                        law_tokens = 0
                        total_docs += 1
                        new_law_data = law_data

                        if len(law_data['articles']) > 0:
                            arts = '\n'.join(law_data['articles'])
                            law_tokens = len(law_data['header'].split()) + len(arts.split())
                        else:
                            law_tokens = len(law_data['header'].split())

                        leg_uri_json = fetch_legislation_json(leg_uri)

                        # if False:
                        if leg_uri_json is not None:
                            doc_arts = []
                            leg_tokens = 0
                            doc_title = json_key_extract(leg_uri_json, 'dc:title')
                            articles = json_key_extract(leg_uri_json, 'Article')

                            if not articles:
                                articles = json_key_extract(leg_uri_json, 'Container')

                            for arts in articles:
                                if isinstance(arts, list):
                                    for art in arts:
                                        title = json_key_extract(art, 'Title')
                                        content = json_key_extract(art, 'content')
                                        if title and title != '':
                                            content = title + content
                                        doc_arts.append('\n'.join(content))
                                        leg_tokens += len(('\n'.join(content)).split())
                                else:
                                    title = json_key_extract(arts, 'Title')
                                    content = json_key_extract(arts, 'content')
                                    if title and title != '':
                                        content = title + content
                                    doc_arts.append('\n'.join(content))
                                    leg_tokens += len(('\n'.join(content)).split())

                            if leg_tokens > law_tokens:
                                eng_text = False
                                new_law_data['header'] = ''.join(doc_title)
                                new_law_data['articles'] = []

                                for art in doc_arts:
                                    if re.search(r'[A-Za-z]{4,}', art):
                                        # print('English words found: ', leg_uri)
                                        eng_text = True
                                        break
                                    new_law_data['articles'].append(art)

                                if not eng_text:
                                    enhanced = True
                                    # with open(os.path.join(pr.output_folder_enhanced, jsonfile), 'w') as outjsf:
                                    #     json.dump(new_law_data, outjsf, ensure_ascii=False)
                            # else:
                            #     print('Less tokens in leg: ' + os.path.join(root, jsonfile))
                            #     continue

                # If enhanced then write the new one in the new hierarchy (same name, new data), otherwise copy the old
                enhanced_path = root.replace('RAPTARCHIS', 'RAPTARCHIS_ENHANCED')
                os.makedirs(enhanced_path, exist_ok=True)
                if enhanced:
                    with open(os.path.join(enhanced_path, jsonfile), 'w') as outjsf:
                        json.dump(new_law_data, outjsf, ensure_ascii=False)
                    enhanced_docs += 1
                    print('ENHANCED: ', os.path.join(enhanced_path, jsonfile))
                else:
                    shutil.copy(os.path.join(root, jsonfile), enhanced_path)

    print('SUMMARY -> Docs enhanced: ', str(enhanced_docs), '/', str(total_docs))


def fetch_legislation_json(leg_uri):
    try:
        out_json = '{}_{}_{}.json'.format(leg_uri.split('/')[-3], leg_uri.split('/')[-2], leg_uri.split('/')[-1])
        fetched = requests.get(leg_uri + '/data/json')
        if fetched.status_code == 200:
            with open(pr.legislation_json + out_json, "w") as jsf:
                json.dump(fetched.json(), jsf, ensure_ascii=False)
            return fetched.json()
        else:
            raise Exception(leg_uri + ' | STATUS CODE NOT 200')
    except:
        return None


def json_key_extract(obj, key):

    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    if k == key:
                        arr.append(v)
                    extract(v, arr, key)
                elif k == key:
                    arr.append(str(v))
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results


# legislation_boost()
