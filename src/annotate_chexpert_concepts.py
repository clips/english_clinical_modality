import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import sys
import json


def generate_annotated_sentences(negbio_output, pathto_bioscope, modality_type, outfile=''):

    assert modality_type in ['negation', 'speculation']

    concept_idxs, text = extract_chexpert_concepts_and_text(negbio_output)
    with open(pathto_bioscope, 'r') as f:
        annotated_sentences = json.load(f)
    concept_annotations = annotate_chexpert_concepts(annotated_sentences, concept_idxs, text, modality_type=modality_type)
    if outfile:
        with open(outfile, 'w') as f:
            json.dump(concept_annotations, f)

    return concept_annotations


def extract_chexpert_concepts_and_text(negbio_infile):
    # extract chexpert concept spans together with the raw text from the NegBio output
    tree = ET.parse(negbio_infile)
    root = tree.getroot()
    children = root.getchildren()
    document = children[3].getchildren()
    passage = document[1].getchildren()
    text = passage[1].text
    annotations = passage[2:]

    concept_idxs = {}
    for annotation in annotations:
        for element in annotation.getchildren():
            if element.tag == 'location':
                data = element.attrib
                start = int(data['offset'])
                end = start + int(data['length'])
                concept_idxs[start] = text[start:end]

    return concept_idxs, text


def annotate_chexpert_concepts(annotated_sentences, concept_idxs, text, modality_type):
    # tagged sentences input: {sentence_id: [tagged_versions_of_sentence]}

    assert modality_type in ['negation', 'speculation']

    concept_ids = defaultdict(lambda: defaultdict(list))
    for sentence_index, tagged_sentences in annotated_sentences.items():
        # each separate tagged sentence contains a single cue!
        for tagged_sentence in tagged_sentences:
            # search for cue
            cue_tokens, cue_idxs = [], []
            token_index_surplus = 0
            for token_index, (token, label) in enumerate(tagged_sentence):
                if label == 'CUE':
                    cue_tokens.append(token)
                    cue_idxs = []
                    for i in range(len(token.split())):
                        cue_idx = token_index + i + token_index_surplus
                        cue_idxs.append(cue_idx)
                    token_index_surplus += (len(token.split()) - 1)

            cue = ' '.join(cue_tokens)
            concept_count = 1
            token_index_surplus = 0
            for token_index, (token, tag) in enumerate(tagged_sentence):
                text_position = text.index(token)
                if text_position in concept_idxs:

                    # collect concept tokens
                    concept_identifier = 'C{}'.format(concept_count)
                    matching_concept = concept_idxs[text_position]
                    split_concept = matching_concept.split()
                    if len(split_concept) > 1:
                        new_concept_idx = text_position + matching_concept.index(split_concept[1])
                        concept_idxs[new_concept_idx] = ' '.join(split_concept[1:])
                    else:
                        concept_count += 1

                    # assign modality
                    if tag in ['I', 'B', 'A']:
                        cue_data = {'cue': cue,
                                    'cue_idxs': cue_idxs}
                    else:
                        cue_data = None

                    # bundle data
                    token_dict = {'token_idx': token_index + token_index_surplus,
                                  'cue_data': cue_data}
                    concept_ids[sentence_index][concept_identifier].append(token_dict)

                token_index_surplus += (len(token.split()) - 1)

    # fuse data into single organized concept annotations
    concept_annotations = defaultdict(lambda: defaultdict(dict))
    for sent_id, concept_data in concept_ids.items():
        for concept_id, data in concept_data.items():
            collected_cues = {}
            token_idxs = []
            for token_dict in data:
                token_idx = token_dict['token_idx']
                token_idxs.append(token_idx)
                cue_data = token_dict['cue_data']
                if cue_data:
                    collected_cues[tuple(cue_data['cue_idxs'])] = cue_data['cue']

            organized_cues = {}
            for cue_index, (cue_idxs, cue) in enumerate(sorted(collected_cues.items())):
                cue_id = 'cue_{}'.format(cue_index)
                organized_cues[cue_id] = {'cue': cue,
                                          'cue_idxs': cue_idxs}
            if organized_cues:
                modality = True
            else:
                modality = False

            # convert collected cues to cue identifiers!
            concept_annotations[sent_id][concept_id][modality_type] = modality
            concept_annotations[sent_id][concept_id]['token_idxs'] = token_idxs
            concept_annotations[sent_id][concept_id]['cue_data'] = organized_cues

    return concept_annotations
