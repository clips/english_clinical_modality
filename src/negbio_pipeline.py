"""
This script runs a NegBio pipeline on extracted BioScope sentences for CheXpert concept extraction.

Ideally, this could be done in one go using the main_chexpert command, but this led to problems with the Python apt_pkg,
which we're circumventing in this script.

Don't forget to allow negbio_pipeline to be used by the command line!
# export PATH=~/.local/bin:$PATH
"""
import os
import sys


# we need to carry out step 1, 4, 5 and 9 of the pipeline
def negbio_pipeline(infile, start_outfile):

    with open(infile, 'r') as f:
        bioscope_sentences = json.load(f)
    start_infile = '{}.txt'.format(infile)
    with open(start_infile, 'w') as f:
        for _, sentence in sorted(bioscope_sentences.items()):
            f.write(sentence + '\n')

    # this step creates a file
    outfile = '{}_step1'.format(start_outfile)
    print('Converting to bioc format...')
    os.system('negbio_pipeline text2bioc --output={} {}'.format(outfile, start_infile))

    # the following steps create a directory, we need to get the file out of that directory each time
    new_infile = outfile
    outfile = '{}_step4'.format(start_outfile)
    print('Splitting in sentences...')
    os.system('negbio_pipeline ssplit --output={} {}'.format(outfile, new_infile))

    new_indirectory = outfile
    new_infile = '{}/{}'.format(new_indirectory, os.listdir(new_indirectory)[0])
    outfile = '{}_step5'.format(start_outfile)
    print('Extracting CheXpert concepts...')
    os.system('negbio_pipeline dner_chexpert --output={} {}'.format(outfile, new_infile))

    new_indirectory = outfile
    new_infile = '{}/{}'.format(new_indirectory, os.listdir(new_indirectory)[0])
    outfile = '{}_final'.format(start_outfile)
    print('Converting to final form...')
    os.system('negbio_pipeline dner_chexpert --output={} {}'.format(outfile, new_infile))
    new_indirectory = outfile
    final_outfile = '{}/{}'.format(new_indirectory, os.listdir(new_indirectory)[0])

    os.system('mv {} {}'.format(final_outfile, start_outfile+'.xml'))
