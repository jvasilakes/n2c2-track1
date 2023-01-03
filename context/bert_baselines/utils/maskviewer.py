import os
import re
import json
import random
import argparse
from hashlib import md5
from collections import OrderedDict

from flask import Flask
from flask import request, redirect, url_for
from dominate import document
from dominate.tags import *
from matplotlib import cm
from matplotlib import colors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath", type=str,
        help="""Path to JSON lines file containing mask info,
                e.g., output by run.py validate --output_token_masks""")
    parser.add_argument(
        "--num-examples", "-N", type=int, default=1000,
        help="Number of examples to load.")
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Localhost port on which to serve the application.")
    parser.add_argument(
        "--annotation-spec", type=str, default=None,
        help="""Path to a JSON file containing the annotation specifications.
                If None, no annotation interface is rendered.""")
    parser.add_argument(
        "--annotation-answers", type=str, default="anns.json",
        help="""Path to JSON file containing annotation answers to render.""")
    parser.add_argument(
        "--shuffle", default=False, action="store_true",
        help="If specified, shuffle the order of examples.")
    parser.add_argument(
        "--random-state", type=int, default=0,
        help="Set the random seed.")
    return parser.parse_args()


def create_app(args):
    app = Flask(__name__)
    app.config["filename"] = args.filepath
    app.config["data"] = load_data(args.filepath, shuffle=args.shuffle,
                                   random_state=args.random_state)
    app.config["state"] = OrderedDict({
        "collapse wordpiece": True,
        "rm_special_tokens": True,
        "correct": True,
        "incorrect": True,
        "max_examples": args.num_examples,
    })
    ann_spec = None
    ann_answers = None
    if args.annotation_spec is not None:
        ann_spec = json.load(open(args.annotation_spec))
        ann_answers = {}
        if os.path.isfile(args.annotation_answers):
            ann_answers = json.load(open(args.annotation_answers))
    app.config["annotation_spec"] = ann_spec
    app.config["annotation_answers"] = ann_answers

    # Define and register the label matchers.
    filters = Filters()
    labels = sorted(set([d["label"] for d in app.config["data"]]))
    for lab in labels:
        filters.register_match_fn("label", lab, name=f"label_{lab}")
        filters.register_match_fn("prediction", lab, name=f"prediction_{lab}")
        app.config["state"][f"label_{lab}"] = True
        app.config["state"][f"prediction_{lab}"] = True

    # The main page
    @app.route("/", methods=("GET", "POST"))
    def view_file():
        state = app.config["state"]
        data = app.config["data"]
        annotation_spec = app.config["annotation_spec"]
        annotation_answers = app.config["annotation_answers"]

        if request.method == "POST":
            if "annotation_task" in list(request.form.keys())[0]:
                annotation_answers = save_annotations_as_json(
                    request.form, answers_file=args.annotation_answers)
                print(f"Saved to {args.annotation_answers}")
            else:
                print("FILTER")
                print(request.form)
                for (inp, value) in state.items():
                    input_val = request.form.get(inp)
                    if inp == "max_examples":
                        try:
                            val = int(input_val)
                        except ValueError:
                            val = state[inp]
                    else:
                        val = input_val == "on"
                    state[inp] = val

        d = document(title="Stochastic Mask Viewer")
        d += h1("Stochastic Mask Viewer")
        d += p(f"Viewing: {app.config['filename']}")
        d += h3("Filters")
        f = form(method="post")
        for (key, val) in state.items():
            if key == "max_examples":
                continue
            inp = input_(_type="checkbox", _id=key, name=key,
                         checked=state.get(key) or False),
            lab = label(key.title(), for_=key)
            f += inp
            f += lab
            f += br()
        lab = label("Max Examples (-1 for all)", for_=key)
        inp = input_(_type="text", _id="max_examples", name="max_examples",
                     placeholder=str(state.get("max_examples")))
        f += lab
        f += inp
        f += br()
        f += input_(_type="submit", value="Apply")
        f += br()
        f += br()
        d += f

        filtered_data = apply_filters(filters, data, state)
        n_shown = min(int(state.get("max_examples")), len(filtered_data))
        d += p(f"Showing {n_shown}/{len(filtered_data)} examples")
        d += br()

        for (i, example) in enumerate(filtered_data):
            if i == int(state.get("max_examples")):
                break
            d += example2html(
                example, collapse_wordpiece=state.get("collapse wordpiece"),
                rm_special_tokens=state.get("rm_special_tokens")
            )
            if app.config["annotation_spec"] is not None:
                d += br()
                # Changes d in place
                example_hash = md5(str(example).encode()).hexdigest()
                render_annotation_task(d, app.config["annotation_spec"],
                                       example_hash,
                                       answers=annotation_answers)
                d += br()
        return d.render()

    return app


def load_data(filepath, shuffle=False, random_state=0):
    data = [json.loads(line) for line in open(filepath)]
    if shuffle is True:
        random.seed(random_state)
        random.shuffle(data)
    return data


def example2html(example, collapse_wordpiece=False, rm_special_tokens=False):
    docid = example["docid"]
    gold = example["label"]
    pred = example["prediction"]

    head = b(f"Doc ID: {docid} | Gold: '{gold}' | Predicted: '{pred}'")

    if gold == pred:
        txt = " ✓ "
        background_color = "#00ff00"
    else:
        txt = " X "
        background_color = "#ff0000"
    sign = span(txt, _class="highlight",
                style=f"background-color:{background_color}")

    highlighted_tokens = get_highlighted_tokens(
        example["tokens_with_masks"],
        collapse_wordpiece=collapse_wordpiece,
        rm_special_tokens=rm_special_tokens)
    text = p(highlighted_tokens)
    return div(sign, head, text)


def get_highlighted_tokens(tokens_with_masks, collapse_wordpiece=False,
                           rm_special_tokens=False):
    spans = []
    if collapse_wordpiece is True:
        tokens_with_masks = collapse_wordpiece_tokens(tokens_with_masks)

    curr_tok_is_subj = False
    for (tok, z) in tokens_with_masks:
        if tok in ["[unused0]", "[unused1]", "[unused2]"]:
            curr_tok_is_subj = not curr_tok_is_subj
            continue

        if rm_special_tokens is True:
            if is_special_token(tok):
                continue

        if curr_tok_is_subj is True:
            style = "color:blue;font-weight:bold"
            title = "Medication"
        else:
            color = z2color(z)
            style = f"background-color:{color}"
            title = f"{z:.3f}"
        token = span(f" {tok} ", _class="highlight",
                     style=style, title=title)
        spans.append(token)
    return spans


def markup_subject_token(tokens_with_masks):
    subj_start_idx = None
    subj_end_idx = None
    for start_i in range(len(tokens_with_masks)):
        end_i = start_i + 3
        triple = tokens_with_masks[start_i:end_i]
        triple_toks = [t for (t, z) in triple]
        if re.match(r'\[unused[0-9]+\]', ''.join(triple_toks)):
            if subj_start_idx is None:
                subj_start_idx = start_i
                continue
            if subj_end_idx is None:
                subj_end_idx = start_i
                break
    marker = ("[S]", 0.0)
    output = tokens_with_masks[:subj_start_idx] + [marker] + \
        tokens_with_masks[subj_start_idx+3:subj_end_idx] + [marker] + \
        tokens_with_masks[subj_end_idx+3:]
    return output


def is_special_token(token):
    specials = [re.escape(r'[CLS]'),
                re.escape(r'[SEP]'),
                r'\[\s?unused[0-9]+\s?\]']
    for special in specials:
        if re.match(special, token) is not None:
            return True
    return False


def z2color(z):
    # Cap the colormap to make the highlighting more readable.
    norm = colors.Normalize(vmin=0, vmax=1.5)
    return colors.rgb2hex(cm.PuRd(norm(z)))


def collapse_wordpiece_tokens(tokens_with_masks):
    output = []
    current_tok = ''
    current_zs = []
    for (tok, z) in tokens_with_masks:
        if tok.startswith("##"):
            current_tok += tok.lstrip("##")
            current_zs.append(z)
        else:
            if len(current_zs) > 0:
                output.append((current_tok, sum(current_zs) / len(current_zs)))
                current_tok = ''
                current_zs = []
            current_tok = tok
            current_zs.append(z)
    output.append((current_tok, sum(current_zs) / len(current_zs)))
    return output


def render_annotation_task(doc, annotation_spec, example_id, answers={}):
    """
    annotation_spec is a python dict with the structure

    {"task": str,
     "fields": list}
    """

    example_answers = answers.get(str(example_id)) or {}

    f = form(method="post", _id="annform")
    f += b(annotation_spec["task"].title())
    f += br()
    for field in annotation_spec["fields"]:
        key = f"{example_id}:annotation_task:"
        key += '_'.join(annotation_spec["task"].split())
        checked = example_answers.get(key) == field
        f += input_(_type="radio", _id=key, name=key,
                    value=field, checked=checked)
        f += label(field.title(), for_=key)
        comment_key = key + "_comment_" + field
        if checked is True:
            comment_text = example_answers.get(comment_key) or ''
            f += input_(value=comment_text, _type="text", _id=comment_key,
                        name=comment_key, placeholder="Optional comment")
        else:
            f += input_(_type="text", _id=comment_key,
                        name=comment_key, placeholder="Optional comment")
        f += br()
    f += button("Save")
    doc += f


def save_annotations_as_json(request_form, answers_file="anns.json"):
    # First remove any unused comment forms.
    form = dict(request_form)
    to_rm = []
    for (key, val) in form.items():
        if val == '':
            to_rm.append(key)
    for key in to_rm:
        form.pop(key)
    example_id = list(form.keys())[0].split(':')[0]

    try:
        all_answers = json.load(open(answers_file))
    except FileNotFoundError:
        all_answers = {}
    all_answers[example_id] = form
    with open(answers_file, 'w') as outF:
        json.dump(all_answers, outF)
    return all_answers


def apply_filters(filters, data, state):
    filtered_data = []
    for d in data:
        filter_results = []
        for (group_name, filter_group) in filters.items():
            group_results = []
            for (key, filt) in filter_group.items():
                if state[key] is True:
                    group_results.append(filt(d))
            filter_results.append(any(group_results))
        if all(filter_results) is True:
            filtered_data.append(d)
    return filtered_data


def register(group, name):
    def assign_name(func):
        func._tag = (group, name)
        return func
    return assign_name


class Filters(object):
    """
    Filters are functions that test a datapoint for a condition.
    Filter functions are organized in groups.
    Generally, filter functions within a group should apply to
    mutually exclusive attributes of the datapoints. E.g., whether
    a datapoint is correctly or incorrectly predicted.
    """

    def __init__(self):
        # Initialize the filters.
        _ = self.filters

    @property
    def filters(self):
        if "_filter_registry" in self.__dict__.keys():
            return self._filter_registry
        else:
            self._filter_registry = {}
            for name in dir(self):
                var = getattr(self, name)
                if hasattr(var, "_tag"):
                    group, fn_name = var._tag
                    if group not in self._filter_registry:
                        self._filter_registry[group] = {}
                    self._filter_registry[group][fn_name] = var
            return self._filter_registry

    def register_match_fn(self, key, label_to_match, name=None):
        """
        `key` is a dict key in a datapoint.
        `label_to_match` is the desired value of datapoint[key].
        That is, the match function will return True if
        datapoint[key] == label_to_match
        """
        reg_fn = register(key, label_to_match)
        match_fn = reg_fn(lambda example: example[key] == label_to_match)
        if name is None:
            name = label_to_match
        if key not in self._filter_registry.keys():
            self._filter_registry[key] = {}
        self._filter_registry[key][name] = match_fn

    def __getitem__(self, group, fn_name):
        return self.filters[group][fn_name]

    def __setitem__(self, *args, **kwargs):
        raise AttributeError(f"{self.__class__} does not support item assignment.")  # noqa 

    def keys(self):
        return self.filters.keys()

    def values(self):
        return self.filters.values()

    def items(self):
        return self.filters.items()

    @register("answers", "correct")
    def correct(self, example):
        return example["label"] == example["prediction"]

    @register("answers", "incorrect")
    def incorrect(self, example):
        return example["label"] != example["prediction"]


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    app.run(port=args.port)
