import os
import json
import argparse
import warnings
from glob import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from brat_reader import BratAnnotations


FEATURE_TYPES = ["bow", "bow_bin", "tfidf"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_dir", type=str,
                        help="n2c2TrainingData/tokenized/")
    parser.add_argument("ann_dir", type=str,
                        help="Directory containing brat ann files.")
    parser.add_argument("out_dir", type=str,
                        help="Where to save the predictions.")
    parser.add_argument("--feature_type", type=str,
                        choices=FEATURE_TYPES, default="bow")
    parser.add_argument("--window_size", type=int, default=0,
                        help="""Positive int of the number of sentences to use
                                +/- the sentence containing the target.""")
    return parser.parse_args()


def main(args):
    os.makedirs(args.out_dir, exist_ok=False)

    all_sentences = {"train": [], "dev": [], "test": []}
    all_dispositions = {"train": [], "dev": [], "test": []}
    ordered_doc_ids = {"train": [], "dev": [], "test": []}

    # Preprocess each dataset
    datasets_to_process = []
    for dataset in ["train", "dev", "test"]:
        dataset_dir = os.path.join(args.ann_dir, dataset)
        text_dir = os.path.join(args.text_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"No dataset directory found at {dataset_dir}. Skipping...")
            continue
        datasets_to_process.append(dataset)
        if not os.path.exists(text_dir):
            print(f"No tokenized texts found at {text_dir}. Skipping...")
            continue
        # dict of document IDs (str) to disposiions (tuple)
        docid_to_dispositions = find_dispositions(dataset_dir)
        for (docid, dispositions) in docid_to_dispositions.items():
            ordered_doc_ids[dataset].append(docid)
            text_file = os.path.join(text_dir, f"{docid}.txt.json")
            # list of sentences, which are lists of tokens
            tokenized_sents = get_sentences(text_file)
            # list of indices in tokenized_sents (int), one per disposition
            sent_idxs = match_sents_to_dispositions(
                    dispositions, tokenized_sents)
            # list of sentences, one per disposition.
            #   If window_size > 0, sentences in the window are concatenated.
            sents_for_features = get_sentences_in_window(
                    sent_idxs, tokenized_sents, window_size=args.window_size)
            all_sentences[dataset].extend(sents_for_features)
            all_dispositions[dataset].extend(dispositions)

    # Compute features for each dataset
    kwargs = get_vectorizer_kwargs(args)
    vectorizer = None
    label_encoders = {}
    all_X = {}
    all_y_by_task = {}
    for dataset in datasets_to_process:
        X, vectorizer = get_features(all_sentences[dataset],
                                     feature_type=args.feature_type,
                                     vectorizer_kwargs=kwargs,
                                     vectorizer=vectorizer)
        print(vectorizer)
        all_X[dataset] = X
        # y_by_task: dict of task name (str) to encoded label (int)
        y_by_task, label_encoders = encode_all_labels(
                all_dispositions[dataset], encoders=label_encoders)
        all_y_by_task[dataset] = y_by_task
        print(f"{dataset} size: {X.shape}")

    # Train an SVM on each task
    clf_by_task = {}
    for task in y_by_task.keys():
        print(f"Training on {task}")
        y = all_y_by_task["train"][task]
        clf = make_pipeline(StandardScaler(with_mean=False),
                            LinearSVC(random_state=0, tol=1e-5))
        clf.fit(all_X["train"], y)
        clf_by_task[task] = clf

    # Predict on each dataset/task
    for dataset in datasets_to_process:
        preds_by_task = {}
        for (task, clf) in clf_by_task.items():
            preds = clf.predict(all_X[dataset])
            preds_by_task[task] = preds
        pred_dispositions = assign_predictions_to_events(
                all_dispositions[dataset], preds_by_task, label_encoders)
        dataset_outdir = os.path.join(args.out_dir, dataset)
        os.makedirs(dataset_outdir)
        save_predictions(pred_dispositions, dataset_outdir)


def find_dispositions(ann_dir):
    glob_path = os.path.join(ann_dir, "*.ann")
    ann_files = glob(glob_path)

    all_dispositions = {}
    for ann_file in ann_files:
        anns = BratAnnotations.from_file(ann_file)
        dispositions = anns.get_events_by_type("Disposition")
        doc_id = os.path.basename(ann_file).replace(".ann", '')
        all_dispositions[doc_id] = dispositions
    return all_dispositions


def get_sentences(filepath):
    with open(filepath, 'r') as inF:
        sentences = [json.loads(line.strip()) for line in inF]
    tokenized_sents = []
    for sent in sentences:
        tokenized_sents.append(sent["tokens"])
    return tokenized_sents


def match_sents_to_dispositions(dispositions, sentences):
    # Each character index in the full document maps to an index in sentences
    # I use a dict because I don't trust the token indices to be contiguous.
    sent_index_lookup = {}
    for (i, sent) in enumerate(sentences):
        for tok in sent:
            start_j, end_j, tok_text = tok
            for j in range(start_j, end_j+1):
                sent_index_lookup[j] = i

    sent_idxs = []
    for dis in dispositions:
        try:
            sent_i = sent_index_lookup[dis.span.start_index]
        except KeyError:
            print(f"MISSING {dis}")
            continue
        sent_idxs.append(sent_i)
    return sent_idxs


def get_sentences_in_window(sent_idxs, tokenized_sents, window_size=0):
    all_windowed_sents = []
    for sent_i in sent_idxs:
        # sent = tokenized_sents[sent_i]
        start_i = max([0, sent_i - window_size])
        # Add 1 because range is not inclusive of the end index
        end_i = min([len(tokenized_sents), sent_i + window_size + 1])
        # tok[2] gets the token by itself without start/end indices
        window_sents = [tok[2] for i in range(start_i, end_i)
                        for tok in tokenized_sents[i]]
        all_windowed_sents.append(window_sents)
    return all_windowed_sents


def get_vectorizer_kwargs(args):
    if args.feature_type == "bow":
        return {}
    elif args.feature_type == "bow_bin":
        return {"binary": True}
    else:
        return {}


def dummy_tokenizer(doc):
    return doc


def get_features(tokenized_sentences, feature_type="bow",
                 vectorizer_kwargs={}, vectorizer=None):
    if vectorizer is None:
        if "bow" in feature_type:
            vectorizer = CountVectorizer(tokenizer=dummy_tokenizer,
                                         preprocessor=dummy_tokenizer,
                                         **vectorizer_kwargs)
        elif "tfidf" in feature_type:
            vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer,
                                         preprocessor=dummy_tokenizer,
                                         **vectorizer_kwargs,
                                         max_df=50)
        else:
            raise NotImplementedError(f"Unsupported feature_type '{feature_type}'")  # noqa
        # Suppresses a warning that token_filter wont be used since we have
        #  pre-tokenized sentences.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vectorizer.fit(tokenized_sentences)
    X = vectorizer.transform(tokenized_sentences)
    return X, vectorizer


def encode_all_labels(dispositions, encoders={}):
    y_by_task = {}
    attr_names = set([key for d in dispositions
                      for key in d.attributes.keys()])
    for attr_name in attr_names:
        try:
            encoder = encoders[attr_name]
        except KeyError:
            encoder = None
        y, encoder = encode_one_label(dispositions, attr_name, encoder=encoder)
        y_by_task[attr_name] = y
        encoders[attr_name] = encoder
    return y_by_task, encoders


def encode_one_label(dispositions, attr_name, encoder=None):
    if attr_name != "Negation":
        values = [d.attributes[attr_name].value for d in dispositions]
    else:
        values = []
        for d in dispositions:
            attr = d.attributes[attr_name]
            val = attr.value
            values.append(val)
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(values)
    y = encoder.transform(values)
    return y, encoder


def assign_predictions_to_events(dispositions, preds_by_task, label_encoders):
    all_decoded_preds = {}
    for (task, preds) in preds_by_task.items():
        enc = label_encoders[task]
        decoded_preds = enc.inverse_transform(preds)
        all_decoded_preds[task] = decoded_preds

    dispositions_copy = [d.copy() for d in dispositions]
    for (i, d) in enumerate(dispositions_copy):
        for (attr_name, attr) in d.attributes.items():
            dec_pred = all_decoded_preds[attr_name][i]
            attr.update("value", dec_pred)

    return dispositions_copy


def save_predictions(dispositions, outdir):
    anns = BratAnnotations.from_events(dispositions)
    anns.save_brat(outdir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
