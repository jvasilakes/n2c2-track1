import os
from os.path import isfile, join
import argparse
from glob import glob

import spacy
from pymusas.spacy_api.taggers import rule_based
from pymusas.pos_mapper import UPOS_TO_USAS_CORE
nlp = spacy.load('en_core_sci_scibert', exclude=['parser', 'ner'])
custom_config = {'usas_tags_token_attr': 'semantic_tags'}
usas_tagger = nlp.add_pipe('usas_tagger', config=custom_config)
# Adds the tagger to the pipeline and returns the tagger 
# usas_tagger = nlp.add_pipe('usas_tagger')
# ##
# portuguese_usas_lexicon_url = 'https://raw.githubusercontent.com/UCREL/Multilingual-USAS/master/Portuguese/semantic_lexicon_pt.tsv'
# portuguese_usas_lexicon_file = download_url_file(portuguese_usas_lexicon_url)
# # Includes the POS information
# portuguese_lexicon_lookup = LexiconCollection.from_tsv(portuguese_usas_lexicon_file)
# # excludes the POS information
# portuguese_lemma_lexicon_lookup = LexiconCollection.from_tsv(portuguese_usas_lexicon_file, 
#                                                              include_pos=False)

# # Add the lexicon information to the USAS tagger within the pipeline
# usas_tagger.lexicon_lookup = portuguese_lexicon_lookup
# usas_tagger.lemma_lexicon_lookup = portuguese_lemma_lexicon_lookup
# Maps from the POS model tagset to the lexicon POS tagset
usas_tagger.pos_mapper = UPOS_TO_USAS_CORE

text = 'The car started speeding.'

tokens = nlp(text)

for tok in tokens:
    print(tok._.semantic_tags )