# read in BioScope corpus in a tagged format that is easily digestable for our models
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pattern_tokenizer import pattern_tokenize


class BioScopeReader:

    def __init__(self, modality):

        assert modality in ['negation', 'speculation']
        self.modality = modality

        self.reference_modality_id = None
        self.reference_xcope = None
        self.parent_map = {}
        self.within_scope = False
        self.cue_observed = False

    def __call__(self, infile, simple_tags=False, include_neutral=True, annotated=True, outfile=''):
        """Collects all relevant sentences and annotations from a BioScope xml file"""
        # BEWARE: one sentence outputted per separate cue present, so duplicates can occur!
        tagged_sentences = {}
        tree = ET.parse(infile)
        root = tree.getroot()
        documentset = root[0]
        sentence_identifier = 0
        for document in documentset:
            for documentpart in document[1:]:
                for sentence in documentpart:
                    modality_ids = self.extract_modality_ids(sentence)
                    if modality_ids:
                        tagged = self.tag_sentence_bao(sentence, modality_ids)
                        if simple_tags:
                            tagged = self.convert_bao_to_io(tagged)
                        if annotated:
                            tagged_sentences[sentence_identifier] = tagged
                        else:
                            tagged_sentences[sentence_identifier] = ' '.join(x[0] for x in tagged[0])
                    elif include_neutral:
                        tagged = [self.tag_without_modality(sentence)]
                        if annotated:
                            tagged_sentences[sentence_identifier] = tagged
                        else:
                            tagged_sentences[sentence_identifier] = ' '.join(x[0] for x in tagged[0])
                    sentence_identifier += 1

        if outfile:
            with open(outfile, 'w') as f:
                json.dump(tagged_sentences, f)

        return tagged_sentences

    def collect_modality_cues(self, infile, modality='negation'):
        assert modality in ['negation', 'speculation']
        self.modality = modality
        collected_modality_cues = Counter()
        tree = ET.parse(infile)
        root = tree.getroot()
        documentset = root[0]
        for document in documentset:
            for documentpart in document[1:]:
                for sentence in documentpart:
                    modality_cues = self.extract_modality_cues(sentence)
                    collected_modality_cues.update(modality_cues)

        return collected_modality_cues

    def tag_sentence_bao(self, sentence, modality_ids):
        """Tags a sentence using the BAO scheme from Qian et al. 2016.
        Returns a tagged sentence per separate modality cue and scope"""

        tagged_sentences = []
        # extract unmarked outer parts of sentence, independent from modality cues inside sentence
        begin_of_sentence, end_of_sentence = self.tag_unmarked_outer_parts(sentence)

        # first extract ids of all modality scopes and cues, then do the extraction for each separate id
        self.parent_map = {c: p for p in sentence.iter() for c in p}
        for reference_modality_id in modality_ids:
            inside_sentence_tags = self.tag_inside_sentence_bao(sentence, reference_modality_id)
            tagged_sentence = begin_of_sentence + inside_sentence_tags + end_of_sentence
            tagged_sentences.append(tagged_sentence)

        return tagged_sentences

    def tag_inside_sentence_bao(self, sentence, reference_modality_id):
        """Tags a sentence for a single specified modality cue using the BAO annotation scheme"""
        self.reference_modality_id = reference_modality_id
        self.cue_observed = False

        inside_sentence_tags = []
        children = sentence.getchildren()
        self.iterate_and_tag_bao(children, inside_sentence_tags)
        self.reference_xcope = None

        return inside_sentence_tags

    def iterate_and_tag_bao(self, children, inside_sentence_tags):
        """Iterates over all child nodes of a sentence and tags every token within the sentence"""
        for child in children:
            self.child_within_reference_xcope(child)
            if child.tag == 'xcope':
                self.tag_xcope_bao(child, inside_sentence_tags, text='text')
            elif child.tag == 'cue':
                self.tag_cue_bao(child, inside_sentence_tags)
            else:
                raise ValueError('Something is not right, tag of child is not xcope or cue but {}'.format(child.tag))
            # recursion
            new_children = child.getchildren()
            self.iterate_and_tag_bao(new_children, inside_sentence_tags)
            if child.tag == 'xcope':
                self.tag_xcope_bao(child, inside_sentence_tags, text='tail')

    def child_within_reference_xcope(self, child):
        """Checks if a current child node has the reference neg xcope as ancestor"""
        self.within_scope = False
        if not self.reference_xcope:
            return
        else:
            # iterate over parent map until xcope found
            reference_child = child
            while True:
                try:
                    ancestor = self.parent_map[reference_child]
                except KeyError:
                    return
                if ancestor == self.reference_xcope:
                    self.within_scope = True
                    return
                else:
                    reference_child = ancestor

    def tag_xcope_bao(self, xcope, inside_sentence_tags, text='text'):
        """Tags an xcope with the BAO tag scheme"""
        if xcope.attrib['id'] == self.reference_modality_id:
            self.reference_xcope = xcope
            self.within_scope = True

        if text == 'text':
            xcope_text = self.tokenize(xcope.text)
        elif text == 'tail':
            xcope_text = self.tokenize(xcope.tail)
        else:
            raise ValueError('text argument has to be text or tail, not {}'.format(text))

        if self.within_scope:
            tag = 'A' if self.cue_observed else 'B'
        else:
            tag = 'O'

        for xcope_token in xcope_text:
            inside_sentence_tags.append((xcope_token, tag))

    def tag_cue_bao(self, cue, inside_sentence_tags):
        """Tags a cue with the BAO tag scheme"""
        cue_text = self.tokenize(cue.text)
        cue_tail = self.tokenize(cue.tail)

        # if reference modality cue
        if cue.attrib['ref'] == self.reference_modality_id:
            inside_sentence_tags.append((cue.text.strip(), 'CUE'))
            self.cue_observed = True
            tag = 'A' if self.within_scope else 'O'
            for cue_token in cue_tail:
                inside_sentence_tags.append((cue_token, tag))
            return

        # if not reference modality cue
        if self.within_scope:
            tag = 'A' if self.cue_observed else 'B'
        else:
            tag = 'O'
        for cue_token in cue_text + cue_tail:
            inside_sentence_tags.append((cue_token, tag))

    def tag_without_modality(self, sentence):
        """Assigns O (outside) labels for all tokens"""
        begin_of_sentence, end_of_sentence = self.tag_unmarked_outer_parts(sentence)
        children = sentence.getchildren()
        inside_sentence_tags = []
        self.iterate_and_tag_outside(children, inside_sentence_tags)
        tagged_sentence = begin_of_sentence + inside_sentence_tags + end_of_sentence

        return tagged_sentence

    def iterate_and_tag_outside(self, children, inside_sentence_tags):
        """Tags all tokens within a sentence with the O (outside) label"""
        for child in children:
            if child.tag == 'xcope':
                for xcope_text_token in self.tokenize(child.text):
                    inside_sentence_tags.append((xcope_text_token, '0'))
            elif child.tag == 'cue':
                for cue_token in self.tokenize(child.text) + self.tokenize(child.tail):
                    inside_sentence_tags.append((cue_token, '0'))
            else:
                raise ValueError('Something is not right, tag of child is not xcope or cue but {}'.format(child.tag))
            # recursion
            new_children = child.getchildren()
            self.iterate_and_tag_outside(new_children, inside_sentence_tags)
            if child.tag == 'xcope':
                for xcope_text_token in self.tokenize(child.tail):
                    inside_sentence_tags.append((xcope_text_token, '0'))

    @staticmethod
    def tokenize(text):
        if not text:
            return []

        tokens = []
        for line in pattern_tokenize(text.strip()):
            tokens += line.split()

        return tokens

    @staticmethod
    def convert_bao_to_io(tagged):
        converted_sentences = []
        for tagged_sentence in tagged:
            converted_sentence = []
            for token, tag in tagged_sentence:
                if tag in ['B', 'A']:
                    converted_sentence.append((token, 'I'))
                else:
                    converted_sentence.append((token, tag))
            converted_sentences.append(converted_sentence)

        return converted_sentences

    def extract_modality_cues(self, sentence):
        """Extracts all modality cues from a sentence"""
        children = sentence.getchildren()
        modality_cues = set()
        self.scrape_modality_cues(children, modality_cues)

        return modality_cues

    def scrape_modality_cues(self, children, modality_cues):
        """Iterates over all children nodes of a sentence to scrape all modality cues"""
        for child in children:
            if child.tag == 'cue':
                if child.attrib['type'] == self.modality:
                    modality_cues.add(child.text.lower())
            new_children = child.getchildren()
            self.scrape_modality_cues(new_children, modality_cues)

    def extract_modality_ids(self, sentence):
        """Extracts all modality ids from a sentence"""
        children = sentence.getchildren()
        modality_ids = set()
        self.scrape_modality_ids(children, modality_ids)

        return modality_ids

    def scrape_modality_ids(self, children, modality_ids):
        """Iterates over all children nodes of a sentence to scrape all modality ids"""
        for child in children:
            if child.tag == 'cue':
                if child.attrib['type'] == self.modality:
                    modality_ids.add(child.attrib['ref'])
            new_children = child.getchildren()
            self.scrape_modality_ids(new_children, modality_ids)

    def tag_unmarked_outer_parts(self, sentence):
        """Extracts the unmarked beginning and ending of the sentence, if present."""
        begin_of_sentence = []
        for token in self.tokenize(sentence.text):
            begin_of_sentence.append((token, 'O'))

        end_of_sentence = []
        for token in self.tokenize(sentence.tail):
            end_of_sentence.append((token, 'O'))

        return begin_of_sentence, end_of_sentence
