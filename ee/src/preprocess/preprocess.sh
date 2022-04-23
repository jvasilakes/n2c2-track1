#!/bin/sh
# Applying scispacy
python scispacy.py --datadir ../../data/default/train/ --outdir ../../data/default/spacy/train/
python scispacy.py --datadir ../../data/default/dev/ --outdir ../../data/default/spacy/dev/
python scispacy.py --datadir ../../data/split0/train/ --outdir ../../data/split0/spacy/train/
python scispacy.py --datadir ../../data/split0/dev/ --outdir ../../data/split0/spacy/dev/
python scispacy.py --datadir ../../data/split1/train/ --outdir ../../data/split1/spacy/train/
python scispacy.py --datadir ../../data/split1/dev/ --outdir ../../data/split1/spacy/dev/
python scispacy.py --datadir ../../data/split2/train/ --outdir ../../data/split2/spacy/train/
python scispacy.py --datadir ../../data/split2/dev/ --outdir ../../data/split2/spacy/dev/
python scispacy.py --datadir ../../data/split3/train/ --outdir ../../data/split3/spacy/train/
python scispacy.py --datadir ../../data/split3/dev/ --outdir ../../data/split3/spacy/dev/
python scispacy.py --datadir ../../data/split4/train/ --outdir ../../data/split4/spacy/train/
python scispacy.py --datadir ../../data/split4/dev/ --outdir ../../data/split4/spacy/dev/
#
python scispacy.py --datadir ../../data/ensemble/train/ --outdir ../../data/ensemble/spacy/train/
python scispacy.py --datadir ../../data/ensemble/dev/ --outdir ../../data/ensemble/spacy/dev/
# Finalizing the input
python preprocess_spacy_words.py --txt_files ../../data/default/train/ --ann_files ../../data/default/train/ --spacy_files ../../data/default/spacy/train/
python preprocess_spacy_words.py --txt_files ../../data/default/dev/ --ann_files ../../data/default/dev/ --spacy_files ../../data/default/spacy/dev/
python preprocess_spacy_words.py --txt_files ../../data/split0/train/ --ann_files ../../data/split0/train/ --spacy_files ../../data/split0/spacy/train/
python preprocess_spacy_words.py --txt_files ../../data/split0/dev/ --ann_files ../../data/split0/dev/ --spacy_files ../../data/split0/spacy/dev/
python preprocess_spacy_words.py --txt_files ../../data/split1/train/ --ann_files ../../data/split1/train/ --spacy_files ../../data/split1/spacy/train/
python preprocess_spacy_words.py --txt_files ../../data/split1/dev/ --ann_files ../../data/split1/dev/ --spacy_files ../../data/split1/spacy/dev/
python preprocess_spacy_words.py --txt_files ../../data/split2/train/ --ann_files ../../data/split2/train/ --spacy_files ../../data/split2/spacy/train/
python preprocess_spacy_words.py --txt_files ../../data/split2/dev/ --ann_files ../../data/split2/dev/ --spacy_files ../../data/split2/spacy/dev/
python preprocess_spacy_words.py --txt_files ../../data/split3/train/ --ann_files ../../data/split3/train/ --spacy_files ../../data/split3/spacy/train/
python preprocess_spacy_words.py --txt_files ../../data/split3/dev/ --ann_files ../../data/split3/dev/ --spacy_files ../../data/split3/spacy/dev/
python preprocess_spacy_words.py --txt_files ../../data/split4/train/ --ann_files ../../data/split4/train/ --spacy_files ../../data/split4/spacy/train/
python preprocess_spacy_words.py --txt_files ../../data/split4/dev/ --ann_files ../../data/split4/dev/ --spacy_files ../../data/split4/spacy/dev/
#
python preprocess_spacy_words.py --txt_files ../../data/ensemble/train/ --ann_files ../../data/ensemble/train/ --spacy_files ../../data/ensemble/spacy/train/
python preprocess_spacy_words.py --txt_files ../../data/ensemble/dev/ --ann_files ../../data/ensemble/dev/ --spacy_files ../../data/ensemble/spacy/dev/