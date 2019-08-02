from bllipparser import RerankingParser
import StanfordDependencies
import json
import pickle

print('Loading bllipparser...')
rrp = RerankingParser.fetch_and_load('GENIA+PubMed', verbose=True)


def extract_dependency_data(infile, outfile=''):
    # reads in the BioScope sentences and returns the dependency data

    with open(infile, 'r') as f:
        sentences = json.load(f)

    print('Loading...')
    sd = StanfordDependencies.get_instance(backend='subprocess')

    print('Extracting...')
    trees = {}
    for sentence_id, sentence in sorted(sentences.items()), total=len(sentences):
        try:
            parse = rrp.simple_parse(sentence)
            tree = sd.convert_tree(parse)
            trees[sentence_id] = tree
        except:
            print('Skipped sentence {}'.format(sentence_id))
            trees[sentence_id] = []

    if outfile:
        with open(outfile, 'wb') as f:
            pickle.dump(trees, f)

    return trees
