from collections import defaultdict
import random
from math import ceil
import json
import pickle


def general_results():
    with open('data/dependencies.p', 'rb') as f:
        trees = pickle.load(f)
    with open('data/negation_concepts.json', 'r') as f:
        annotated_concepts = json.load(f)
    conjunct_concepts = NegationDetection.extract_conjunct_concepts(trees, annotated_concepts)

    Detector = NegationDetection(linked_concepts=False)
    data = Detector.traintest_split(annotated_concepts)
    test = data['test']

    results = {}
    results['forward'], _ = Detector.lookforward_baseline(trees, test, conjunct_concepts)
    results['punctuation'], _ = Detector.punctuation_baseline(trees, test, conjunct_concepts)
    Detector.linked_concepts = True
    results['forward_conj'], _ = Detector.lookforward_baseline(trees, test, conjunct_concepts)
    results['punctuation_conj'], _ = Detector.punctuation_baseline(trees, test, conjunct_concepts)

    with open('negation_results.json', 'w') as f:
        json.dump(results, f)


def cue_level_results():
    with open('data/dependencies.p', 'rb') as f:
        trees = pickle.load(f)
    with open('data/negation_concepts.json', 'r') as f:
        annotated_concepts = json.load(f)
    conjunct_concepts = NegationDetection.extract_conjunct_concepts(trees, annotated_concepts)

    Detector = NegationDetection(linked_concepts=False, cue_level_evaluation=True)
    data = Detector.traintest_split(annotated_concepts)
    test = data['test']

    bundled_results = []

    cue_level_results, _ = Detector.punctuation_baseline(trees, test, conjunct_concepts)
    for cue, results in cue_level_results.items():
        precision, recall = results['positive_precision'], results['positive_recall']
        try:
            F1 = 2 * ((precision * recall) / (precision + recall))
        except:
            F1 = 0
        bundled_results.append((cue, (precision, recall, F1)))

    with open('negation_cue_results.json', 'w') as f:
        json.dump(bundled_results, f)

    # pretty printing
    for cue, res in bundled_results:
        print(cue, '&', ' & '.join([str(round(x * 100, 2)) for x in res]), r'\\')


class NegationDetection:

    def __init__(self, modality='negation', linked_concepts=True, cue_level_evaluation=False):

        self.modality_cues = {'negation': ['no', 'without', 'not']}

        assert modality in ['negation', 'speculation']
        self.modality = modality

        self.linked_concepts = linked_concepts

        self.seed = 1993

        self.cue_level_evaluation = cue_level_evaluation

    def traintest_split(self, annotated_sentences):
        # splits annotated sentences in train and test sentences
        train_amount = ceil(len(annotated_sentences) / 2)
        sentence_ids = sorted(annotated_sentences.keys())

        random.seed(self.seed)
        train_sentence_ids = random.sample(sentence_ids, train_amount)

        train_sentences, test_sentences = {}, {}
        for sentence_id, sentence_data in annotated_sentences.items():
            if sentence_id in train_sentence_ids:
                train_sentences[sentence_id] = sentence_data
            else:
                test_sentences[sentence_id] = sentence_data

        # also output statistics!
        numtrain, numtrainmodal = 0, 0
        numtest, numtestmodal = 0, 0
        for sentence_id, sentence_data in train_sentences.items():
            for concept_id, concept_data in sentence_data.items():
                numtrain += 1
                if concept_data[self.modality]:
                    numtrainmodal += 1
        for sentence_id, sentence_data in test_sentences.items():
            for concept_id, concept_data in sentence_data.items():
                numtest += 1
                if concept_data[self.modality]:
                    numtestmodal += 1
        statistics = {'# of train sentences': len(train_sentences),
                      '# of test sentences': len(test_sentences),
                      '# of train concepts': numtrain,
                      '# of test concepts': numtest,
                      '# of {} train concepts'.format(self.modality): numtrainmodal,
                      '# of {} test concepts'.format(self.modality): numtestmodal}

        data = {'train': train_sentences,
                'test': test_sentences,
                'statistics': statistics}

        return data

    @staticmethod
    def evaluate_confusion_matrix(confusion_matrix):

        true_pos = len(confusion_matrix['true_pos'])
        true_neg = len(confusion_matrix['true_neg'])
        false_pos = len(confusion_matrix['false_pos'])
        false_neg = len(confusion_matrix['false_neg'])

        try:
            positive_precision = true_pos / (true_pos + false_pos)
        except ZeroDivisionError:
            positive_precision = None
        try:
            positive_recall = true_pos / (true_pos + false_neg)
        except ZeroDivisionError:
            positive_recall = None
        try:
            negative_precision = true_neg / (true_neg + false_neg)
        except ZeroDivisionError:
            negative_precision = None
        try:
            negative_recall = true_neg / (true_neg + false_pos)
        except ZeroDivisionError:
            negative_recall = None

        try:
            accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        except ZeroDivisionError:
            accuracy = None

        try:
            majority_baseline = max(true_pos + false_neg, true_neg + false_pos) / (true_pos + false_neg + true_neg + false_pos)
        except ZeroDivisionError:
            majority_baseline = None

        results = {'accuracy': accuracy,
                   'majority_baseline': majority_baseline,
                   'positive_precision': positive_precision,
                   'positive_recall': positive_recall,
                   'negative_precision': negative_precision,
                   'negative_recall': negative_recall}

        return results

    def evaluate_confusion_matrix_cue_level(self, confusion_matrix):

        cue_level_confusion_matrix = defaultdict(lambda: defaultdict(list))

        # true pos
        for conf in confusion_matrix['true_pos']:
            evaluation_data = {'sentence_id': conf['sentence_id'],
                               'concept_id': conf['concept_id']}
            predicted_cues = {predicted_cue.lower() for predicted_cue in conf['predicted_cues']}
            for true_cue in conf['true_cues']:
                if true_cue in predicted_cues:
                    cue_level_confusion_matrix[true_cue]['true_pos'].append(evaluation_data)
                else:
                    if true_cue in self.modality_cues[self.modality]:
                        cue_level_confusion_matrix[true_cue]['false_neg'].append(evaluation_data)

        # false pos
        for conf in confusion_matrix['false_pos']:
            evaluation_data = {'sentence_id': conf['sentence_id'],
                               'concept_id': conf['concept_id']}
            predicted_cues = {predicted_cue.lower() for predicted_cue in conf['predicted_cues']}
            for predicted_cue in predicted_cues:
                if predicted_cue in self.modality_cues[self.modality]:
                    cue_level_confusion_matrix[predicted_cue]['false_pos'].append(evaluation_data)

        # false neg
        for conf in confusion_matrix['false_neg']:
            evaluation_data = {'sentence_id': conf['sentence_id'],
                               'concept_id': conf['concept_id']}
            for true_cue in conf['true_cues']:
                if true_cue in self.modality_cues[self.modality]:
                    cue_level_confusion_matrix[true_cue]['false_neg'].append(evaluation_data)

        # combine into cue-level results
        cue_level_results = {}
        for cue in self.modality_cues[self.modality]:

            true_pos = len(cue_level_confusion_matrix[cue]['true_pos'])
            false_pos = len(cue_level_confusion_matrix[cue]['false_pos'])
            false_neg = len(cue_level_confusion_matrix[cue]['false_neg'])

            try:
                positive_precision = true_pos / (true_pos + false_pos)
            except ZeroDivisionError:
                positive_precision = None
            try:
                positive_recall = true_pos / (true_pos + false_neg)
            except ZeroDivisionError:
                positive_recall = None

            results = {'positive_precision': positive_precision,
                       'positive_recall': positive_recall}

            cue_level_results[cue] = results

        return cue_level_results, cue_level_confusion_matrix

    def fill_confusion_matrix(self, confusion_matrix, sentence_id, sentence_concepts, concept_predictions):

        for concept_id, concept_data in sentence_concepts.items():
            cue_data = concept_data['cue_data']
            true_cues = {cue_id_data['cue'].lower() for cue_id_data in cue_data.values()}
            predicted_cues = concept_predictions[concept_id]

            # check for matches
            evaluation_data =  {'sentence_id': sentence_id,
                                'concept_id': concept_id,
                                'predicted_cues': predicted_cues,
                                'true_cues': true_cues}

            if concept_data[self.modality]:  # if ground truth is modality
                if predicted_cues:
                    confusion_matrix['true_pos'].append(evaluation_data)
                else:
                    confusion_matrix['false_neg'].append(evaluation_data)
            else:
                if predicted_cues:
                    confusion_matrix['false_pos'].append(evaluation_data)
                else:
                    confusion_matrix['true_neg'].append(evaluation_data)

    def lookforward_baseline(self, trees, annotated_concepts, conjunct_concepts):

        confusion_matrix = defaultdict(list)

        for sentence_id, sentence_concepts in sorted(annotated_concepts.items()):
            tree = trees[sentence_id]
            concept_predictions = defaultdict(list)
            for token in tree:
                if token.form.lower() in self.modality_cues[self.modality]:
                    cue = token.form.lower()
                    # assign all following concept_ids negation status
                    start_index = token.index - 1
                    for concept_id, concept_data in sentence_concepts.items():
                        concept_idxs = concept_data['token_idxs']
                        if max(concept_idxs) > start_index:
                            concept_predictions[concept_id].append(cue)

            # link predictions for linked concepts
            if self.linked_concepts:
                conjunct_cs = conjunct_concepts[sentence_id]
                if conjunct_cs:
                    conjunct_cs = defaultdict(list, conjunct_cs)
                    for concept_id, cues in list(concept_predictions.items()):
                        linked_concepts = conjunct_cs[concept_id]
                        for linked_concept in linked_concepts:
                            concept_predictions[linked_concept] += cues

            # compare concept predictions to ground truth and fill confusion matrix accordingly
            self.fill_confusion_matrix(confusion_matrix, sentence_id, sentence_concepts, concept_predictions)

        if not self.cue_level_evaluation:
            results = self.evaluate_confusion_matrix(confusion_matrix)
        else:
            results, confusion_matrix = self.evaluate_confusion_matrix_cue_level(confusion_matrix)

        print(results)

        return results, confusion_matrix

    def punctuation_baseline(self, trees, annotated_concepts, conjunct_concepts):
        # FIXED!
        confusion_matrix = defaultdict(list)

        for sentence_id, sentence_concepts in sorted(annotated_concepts.items()):
            tree = trees[sentence_id]
            concept_predictions = defaultdict(list)
            for token in tree:
                if token.form.lower() in self.modality_cues[self.modality]:
                    cue = token.form.lower()
                    # assign all following concept_ids negation status if they appear before first following punctuation
                    start_index = token.index - 1
                    next_punctuation_index = None
                    for t in tree[start_index:]:
                        if t.form in '!?.;,:':
                            next_punctuation_index = t.index - 1
                            break
                    start_index = token.index - 1
                    for concept_id, concept_data in sentence_concepts.items():
                        concept_idxs = concept_data['token_idxs']
                        # skip concept if it appears past the first following punctuation
                        if next_punctuation_index:
                            if min(concept_idxs) > next_punctuation_index:
                                continue
                        if max(concept_idxs) > start_index:
                            concept_predictions[concept_id].append(cue)

            # link predictions for linked concepts
            if self.linked_concepts:
                conjunct_cs = conjunct_concepts[sentence_id]
                if conjunct_cs:
                    conjunct_cs = defaultdict(list, conjunct_cs)
                    for concept_id, cues in list(concept_predictions.items()):
                        linked_concepts = conjunct_cs[concept_id]
                        for linked_concept in linked_concepts:
                            concept_predictions[linked_concept] += cues

            # compare concept predictions to ground truth and fill confusion matrix accordingly
            self.fill_confusion_matrix(confusion_matrix, sentence_id, sentence_concepts, concept_predictions)

        if not self.cue_level_evaluation:
            results = self.evaluate_confusion_matrix(confusion_matrix)
        else:
            results, confusion_matrix = self.evaluate_confusion_matrix_cue_level(confusion_matrix)

        print(results)

        return results, confusion_matrix

    def cue_specific_dependency_rules(self, token, tree):

        affected_token_idxs = []
        """
        # example of a possible simple rule:
        cue = token.form.lower()
        if cue == 'no':
            if token.deprel == 'neg':
                head_index = token.head
                affected_token_idxs.append(head_index - 1)
        """

        return affected_token_idxs

    def dependency_model(self, trees, annotated_concepts, conjunct_concepts):

        # insert rules into this dependency model using the function self.cue_specific_dependency_rules

        confusion_matrix = defaultdict(list)

        for sentence_id, sentence_concepts in sorted(annotated_concepts.items()):
            tree = trees[sentence_id]
            concept_predictions = defaultdict(list)
            for token in tree:
                if token.form.lower() in self.modality_cues[self.modality]:
                    cue = token.form.lower()
                    affected_token_idxs = self.cue_specific_dependency_rules(token, tree)
                    # assign all affected concept_ids negation status
                    for concept_id, concept_data in sentence_concepts.items():
                        concept_idxs = concept_data['token_idxs']
                        for concept_idx in concept_idxs:
                            if concept_idx in affected_token_idxs:
                                concept_predictions[concept_id].append(cue)

            # link predictions for linked concepts
            if self.linked_concepts:
                conjunct_cs = conjunct_concepts[sentence_id]
                if conjunct_cs:
                    conjunct_cs = defaultdict(list, conjunct_cs)
                    for concept_id, cues in list(concept_predictions.items()):
                        linked_concepts = conjunct_cs[concept_id]
                        for linked_concept in linked_concepts:
                            concept_predictions[linked_concept] += cues

            # compare concept predictions to ground truth and fill confusion matrix accordingly
            self.fill_confusion_matrix(confusion_matrix, sentence_id, sentence_concepts, concept_predictions)

        if not self.cue_level_evaluation:
            results = self.evaluate_confusion_matrix(confusion_matrix)
        else:
            results, confusion_matrix = self.evaluate_confusion_matrix_cue_level(confusion_matrix)

        print(results)

        return results, confusion_matrix

    @staticmethod
    def extract_conjunct_concepts(trees, annotated_concepts):
        # FINISHED! extracts conjunct concepts for each sentence

        conjunct_concepts = {}
        for sentence_id, sentence_concepts in sorted(annotated_concepts.items()):
            # check if concepts are conjunct by seeing whether items have a 'conj' deprelation to other tokens
            tree = trees[sentence_id]
            all_concept_ids = set()
            for concept_data in sentence_concepts.values():
                all_concept_ids.update(concept_data['token_idxs'])

            # linked concepts
            linked_concepts = defaultdict(set)
            for token in tree:
                if token.index - 1 not in all_concept_ids:
                    continue
                if token.deprel == 'conj':
                    head_token_idx = token.head - 1
                    head_concepts = set()
                    if head_token_idx in all_concept_ids:
                        # extract all concepts containing the head index
                        for concept, concept_data in sentence_concepts.items():
                            if head_token_idx in concept_data['token_idxs']:
                                head_concepts.add(concept)
                        # extract all concepts containing the dependent index
                        for concept, concept_data in sentence_concepts.items():
                            if token.index - 1 in concept_data['token_idxs']:
                                for head_concept in head_concepts:
                                    linked_concepts[head_concept].add(concept)

            # fuse linked_concepts (iterate 2 times for this)
            for c, linked_cs in list(linked_concepts.items()):
                for linked_c in linked_cs:
                    linked_concepts[linked_c].add(c)
                    linked_concepts[linked_c].update(linked_cs)
            linked_concepts = {k: {x for x in v if x != k} for k, v in linked_concepts.items()}

            conjunct_concepts[sentence_id] = linked_concepts

        return conjunct_concepts
