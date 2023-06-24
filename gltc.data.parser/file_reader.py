import re
import os

import data_node as dn
import properties as pr
import utils


def parse_file():
    volume_rgx = re.compile(r'Τ[ΟΌ]ΜΟΣ\s*\n')
    chapter_rgx = re.compile(
        r'ΚΕΦ[ΑΆ]ΛΑΙΟ[NΝ]?\s*\n')  # Not sure if N occurs in Raptarchis hierarchy or only in context
    subject_rgx = re.compile(r'Θ[ΕΈ]ΜΑ\s*\n')
    article_rgx = re.compile(r'^[΄‘]?[αάΑΆA](ρθρο|ΡΘΡΟ|ρθρ|ΡΘΡ|ρθρον|ΡΘΡΟΝ).*\s*\n')

    # general_law_rgx = re.compile(r'^([0-9]+[Α-Ωα-ωA-Za-z]{0,1})\.[ ]*([Α-Ω0-9. ]+)((υπ|υπ΄|υπ\'|υπ’)[ ]*αρ.*)?\n')
    general_law_rgx = re.compile(
        r'^([0-9]+[Α-Ωα-ωA-Za-z]?)\.[ ]*([Α-ΩETYIOPAHKBNM]{2,}[Α-Ω0-9΄. ETYIOPAHKBNM]+)((υπ|υπ΄|υπ\'|υπ’)[ ]*[αάΑ]ρ.*)?\n')

    art_id = 1
    law_header = ''

    root_node = dn.DataNode()
    volume_node = None
    chapter_node = None
    subject_node = None
    law_node = None
    article_node = None

    for root, _, f_names in os.walk(pr.data_folder):
        for txt_file in f_names:
            with open(os.path.join(root, txt_file)) as f:
                # with open('Resources/law_types.txt', 'a') as law_types_file:
                for line in f:
                    '''
                    Start checking for structure words like: ΤΟΜΟΣ, ΚΕΦΑΛΑΙΟ, ΘΕΜΑ
                    If none found, then start checking for law content like: ΝΟΜΟΣ, ΔΙΑΤΑΓΜΑ etc.
                    '''
                    # VOLUME
                    # If it matches Volume Regex then we search for Volume's Number and Title
                    if volume_rgx.match(line):
                        next_line = next(f)
                        # Get Volume Number
                        if re.match(r'\d{1,4}[Α-ΩA-Z]{0,2}\s*\n', next_line):
                            volume_no = next_line.rstrip()
                            next_line = next(f)
                            # Get(?) Volume subNumber
                            if re.match(r'(\d{1,2}|[Α-ΩA-Z]{1,2})\s*\n', next_line):
                                volume_no = volume_no + "." + next_line.rstrip()
                                next_line = next(f)

                            # Search if Volume with the same name already exists in the hierarchy
                            volume_node = root_node.search_vol_by_label(next_line.rstrip())
                            if volume_node is None:
                                volume_node = dn.DataNode('ΤΟΜΟΣ', volume_no, next_line.rstrip())
                                root_node.children.append(volume_node)

                            chapter_node = None
                            subject_node = None
                            law_node = None
                            article_node = None
                            art_id = 1

                    # CHAPTER
                    # If it matches Chapter Regex then we search for Chapter's Number and Title
                    elif chapter_rgx.match(line):
                        next_line = next(f)
                        # Chapter Number
                        if re.match(r'[Α-ΩA-Z]{1,3}\s*\n', next_line):
                            chapter_no = next_line.rstrip()
                            next_line = next(f)
                            # Get(?) Chapter subNumber
                            if re.match(r'(\d{1,2}|[Α-ΩA-Z]{1,2})\s*\n', next_line):
                                chapter_no = chapter_no + "." + next_line.rstrip()
                                next_line = next(f)

                            # print(line.rstrip(), chapter_no, next_line.rstrip())
                            chapter_node = dn.DataNode('ΚΕΦΑΛΑΙΟ', chapter_no, next_line.rstrip())
                            volume_node.children.append(chapter_node)
                            subject_node = None
                            law_node = None
                            article_node = None
                            art_id = 1

                    # SUBJECT
                    elif subject_rgx.match(line):
                        next_line = next(f)
                        if re.match(r'[α-ω]{1,3}\s*\n', next_line):
                            subject_no = next_line.rstrip()
                            next_line = next(f)
                            subj_title = next_line.replace('"', '').replace('«', '').replace('»', '').rstrip()
                            subj_title = utils.normalize(subj_title)
                            subject_node = dn.DataNode('ΘΕΜΑ', subject_no, subj_title)
                            if chapter_node is not None:
                                chapter_node.children.append(subject_node)
                            else:
                                volume_node.children.append(subject_node)

                            if law_node is not None and article_node is None and law_header != '':
                                law_node.text = law_header

                            law_node = None
                            article_node = None
                            art_id = 1

                    # CHECK FOR CONTENT
                    elif general_law_rgx.match(line):

                        # CLASSIFIER - writes the line that mentions the law type in a separate file
                        # law_types_file.write(line+'\n')

                        if law_node is not None and article_node is None and law_header != '':
                            law_node.text = law_header

                        m = general_law_rgx.match(line)
                        lawtype = utils.classify_law(m.group(2).rstrip())
                        # law_node = dn.DataNode(m.group(2).rstrip(), m.group(1), line.strip())
                        law_node = dn.DataNode(lawtype, m.group(1), line.strip())
                        article_node = None
                        law_header = ''
                        art_id = 1

                        if subject_node is not None:
                            subject_node.children.append(law_node)
                        elif chapter_node is not None:
                            chapter_node.children.append(law_node)
                        elif volume_node is not None:
                            volume_node.children.append(law_node)

                    elif article_rgx.match(line):

                        article_node = dn.DataNode('ΑΡΘΡΟ', str(art_id))
                        article_node.text = line
                        art_id += 1

                        if law_node is not None:
                            law_node.children.append(article_node)
                            # If it finds at least one article inside law, then flush header text in law.text
                            if law_header != '':
                                law_node.text = law_header
                                law_header = ''
                        elif subject_node is not None:
                            subject_node.children.append(article_node)
                        elif chapter_node is not None:
                            chapter_node.children.append(article_node)

                    else:
                        if article_node is None:
                            law_header += line
                        else:
                            article_node.text += line

                # Reset everything for new file
                volume_node = None
                chapter_node = None
                subject_node = None
                law_node = None
                article_node = None
                art_id = 1

    root_node.sort_tree()
    return root_node


if __name__ == "__main__":
    rn = parse_file()
    utils.fetch_year(rn)
    utils.fetch_law_id(rn)
    utils.generate_leg_uri(rn)
    # utils.dump_to_files(rn)

    # utils.dump_to_files(rn, 1985)
    # utils.token_statistics()

    # utils.show_headers(rn, True)
    # utils.export_contents(rn)
    # utils.cli_statistics(rn)
    # rn.print_tree()
