# Negation and speculation detection of concepts in English clinical text

This repository contains the source code for negation and speculation detection of concepts extracted from the English clinical texts in the [BioScope corpus](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2586758/). This software was developed in the scope of the  [ACCUMULATE](https://github.com/clips/accumulate) project.


## Requirements

* Python 3
* [BLLIP parser](https://github.com/BLLIP/bllip-parser)
* [StanfordDependencies](https://pypi.org/project/PyStanfordDependencies/)
* [NegBio](https://github.com/ncbi-nlp/NegBio)

## Usage

### Reading in BioScope

To process the BioScope data, you need to provide the path to one of the three BioScope corpus files. We developed and tested the code on the clinical records present in the **clinical_rm.xml** file.

```
from bioscope_reader import BioScopeReader
# infile = path to BioScope corpus file
# outfile = file to write tagged or untagged BioScope sentences to
# modality = negation or speculation

reader = BioScopeReader(modality)
# extract tagged BioScope sentences
reader(infile, annotated=True, outfile=outfile)
# extract the text of the same sentences without any tags
reader(infile, annotated=False, outfile=outfile)
```

### Extracting clinical concepts using CheXpert

To extract CheXpert concepts from the BioScope data using NegBio, run

```
from negbio_pipeline import negbio_pipeline
# infile = path to untagged BioScope sentences
# outfile = file to write NegBio output to
negbio_pipeline(infile, outfile)
```

To generate concept annotations fusing the processed BioScope data and the CheXpert concepts extracted by NegBio, run

```
from annotate_chexpert_concepts import generate_annotated_sentences
# negbio_output = file containing the output of NegBio on the BioScope sentences
# pathto_bioscope = path to tagged BioScope sentences
# modality_type = negation or speculation
# outfile = file to write generated concept annotations to
generate_annotated_sentences(negbio_output, pathto_bioscope, modality_type, outfile)
```

### Extracting dependency parses using BLLIP parser and StanfordDependencies

To extract dependency parses from the untagged BioScope sentences, run

```
from extract_dependency import extract_dependency_data
# infile = path to untagged BioScope sentences
# outfile = path to write dependency parses to
extract_dependency_data(infile, outfile)
```

### Negation and speculation detection

The scripts **detect_negation.py** and **detect_speculation.py** contain all code for running negation and speculation detection on the data prepared in the previous steps. This code is contained in the **NegationDetection** and **SpeculationDetection** classes. The **general_results()** and **cue_level_results** give a demonstration and evaluation of highly effective simple baseline models. A dependency parse model is also provided for customization, in which rules can be easily inserted.
.