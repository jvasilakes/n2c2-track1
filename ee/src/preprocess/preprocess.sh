#!/bin/sh
# Applying scispacy
python scispacy.py --datadir ../../data/default/brat/ --outdir ../../data/default/spacy/
python scispacy.py --datadir ../../data/split0/brat/ --outdir ../../data/split0/spacy/
python scispacy.py --datadir ../../data/split1/brat/ --outdir ../../data/split1/spacy/
python scispacy.py --datadir ../../data/split2/brat/ --outdir ../../data/split2/spacy/
python scispacy.py --datadir ../../data/split3/brat/ --outdir ../../data/split3/spacy/
python scispacy.py --datadir ../../data/split4/brat/ --outdir ../../data/split4/spacy/
# Finalizing the input
python make_data.py --datadir ../../data/default/brat/ --spacydir ../../data/default/spacy/
python make_data.py --datadir ../../data/split0/brat/ --spacydir ../../data/split0/spacy/
python make_data.py --datadir ../../data/split1/brat/ --spacydir ../../data/split1/spacy/
python make_data.py --datadir ../../data/split2/brat/ --spacydir ../../data/split2/spacy/
python make_data.py --datadir ../../data/split3/brat/ --spacydir ../../data/split3/spacy/
python make_data.py --datadir ../../data/split4/brat/ --spacydir ../../data/split4/spacy/
# Data are good to go