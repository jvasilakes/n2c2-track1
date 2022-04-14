import json
import argparse
from collections import OrderedDict

from flask import Flask
from flask import request
from dominate import document
from dominate.tags import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath", type=str,
        help="""Path to JSON lines file containing mask info,
                e.g., output by run.py validate --output_token_masks""")
    return parser.parse_args()


def create_app(args):
    app = Flask(__name__)
    app.config["data"] = [json.loads(line.strip())
                          for line in open(args.filepath)]
    app.config["state"] = OrderedDict({
        "collapse wordpiece": True,
        "correct": True,
        "incorrect": True,
        "max_examples": 1000,
    })

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

        if request.method == "POST":
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

        d = document(title="Predictions Viewer")
        d += h1("Predictions Viewer")
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

        for (i, example) in enumerate(apply_filters(filters, data, state)):
            if i == int(state.get("max_examples")):
                break
            d += example2html(
                example, collapse_wordpiece=state.get("collapse wordpiece"))
        return d.render()

    return app


def example2html(example, collapse_wordpiece=False):
    docid = example["docid"]
    gold = example["label"]
    pred = example["prediction"]

    head = b(f"ID: {docid} | Gold: '{gold}' | Predicted: '{pred}'")

    if gold == pred:
        txt = " ✓ "
        background_color = "#00ff00"
    else:
        txt = " X "
        background_color = "#ff0000"
    sign = span(txt, _class="highlight",
                style=f"background-color:{background_color}")

    tokens = get_tokens(example["tokens"],
                        collapse_wordpiece=collapse_wordpiece)
    text = p(tokens)
    return div(sign, head, text)


def get_tokens(tokens, collapse_wordpiece=False):
    spans = []
    if collapse_wordpiece is True:
        tokens = collapse_wordpiece_tokens(tokens)
    for tok in tokens:
        token = span(f" {tok} ")
        spans.append(token)
    return spans


def collapse_wordpiece_tokens(tokens):
    output = []
    current_tok = ''
    for tok in tokens:
        if tok.startswith("##"):
            current_tok += tok.lstrip("##")
        else:
            if current_tok != '':
                output.append(current_tok)
                current_tok = ''
            current_tok = tok
    output.append(current_tok)
    return output


def apply_filters(filters, data, state):
    for d in data:
        filter_results = []
        for (group_name, filter_group) in filters.items():
            group_results = []
            for (key, filt) in filter_group.items():
                if state[key] is True:
                    group_results.append(filt(d))
            filter_results.append(any(group_results))
        if all(filter_results) is True:
            yield d


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
    app.run()
